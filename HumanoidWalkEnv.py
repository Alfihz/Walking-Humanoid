"""
HumanoidWalkEnv V24 - ALL GAUSSIAN REWARDS
==========================================
Every reward component uses a Gaussian distribution peaked at the optimal value.
NO PENALTIES - only positive rewards that guide toward optimal behavior.

Philosophy:
- Don't punish bad behavior → Reward optimal behavior
- Robot learns "where's the sweet spot?" not "what should I avoid?"

Gaussian Formula: reward = weight * exp(-(x - target)² / (2 * sigma²))
- x = current value
- target = optimal value (peak of Gaussian)
- sigma = tolerance (how strict - smaller = stricter)
- weight = importance of this component

REWARD COMPONENTS:
==================
TIER 1 - Survival (Always Active):
  - Height:     peak at 1.4m
  - Upright:    peak at quat_w = 1.0
  - Alive:      constant bonus

TIER 2 - Velocity (Always Active):
  - Speed:      peak at 0.5 m/s
  
TIER 3 - Gait Quality (When Walking):
  - Pitch:      peak at 0 (no lean)
  - Arm Pos:    peak at 0 (arms at sides)
  - Airborne:   peak at 20% (walking, not running)
  - Stride:     peak at 0.4m
  - Step Bonus: discrete reward for alternating steps
"""

from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.utils import EzPickle
import numpy as np
import os

DEFAULT_XML_PATH = os.path.join(os.path.dirname(__file__), 'assets', 'humanoid_180_75.xml')


def gaussian_reward(x, target, sigma, weight):
    """
    Gaussian reward function peaked at target.
    
    Args:
        x: current value
        target: optimal value (center of Gaussian)
        sigma: standard deviation (smaller = stricter)
        weight: maximum reward when x == target
    
    Returns:
        reward in range [0, weight], maximum when x == target
    """
    return weight * np.exp(-((x - target) ** 2) / (2 * sigma ** 2))


class HumanoidWalkEnv(MujocoEnv, EzPickle):
    """V24 - All Gaussian rewards humanoid walking environment."""
    
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 100,
    }

    def __init__(self, xml_file=DEFAULT_XML_PATH, frame_skip=5, training_phase="standing", **kwargs):
        MujocoEnv.__init__(
            self, xml_file, frame_skip,
            observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            **kwargs
        )

        # Body/Sensor IDs
        try:
            self.torso_id = self.model.body('torax').id
            self.pelvis_id = self.model.body('pelvis').id
            self.left_foot_site_id = self.model.site('foot_left_site').id
            self.right_foot_site_id = self.model.site('foot_right_site').id
            self.left_foot_geoms = ['foot1_left', 'foot2_left']
            self.right_foot_geoms = ['foot1_right', 'foot2_right']
        except KeyError as e:
            raise KeyError(f"Required body/site not found in XML: {e}") from e

        obs_sample = self._get_obs()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_sample.shape[0],), dtype=np.float32
        )

        self.training_phase = training_phase
        self.walking_progress = 0.0
        print(f"[V24-Gaussian] Initialized in '{training_phase}' phase")
        
        EzPickle.__init__(self, xml_file=xml_file, frame_skip=frame_skip, 
                         training_phase=training_phase, **kwargs)

        # ================================================================
        # GAUSSIAN REWARD PARAMETERS
        # Format: (target, sigma, weight)
        # ================================================================
        
        # TIER 1: SURVIVAL
        self.height_params = (1.4, 0.15, 8.0)      # Peak at 1.4m, ±15cm tolerance
        self.upright_params = (1.0, 0.1, 5.0)      # Peak at quat_w=1.0
        self.alive_reward = 3.0                     # Constant survival bonus
        
        # TIER 2: VELOCITY
        self.velocity_params = (0.5, 0.2, 10.0)    # Peak at 0.5 m/s, ±0.2 tolerance
        
        # TIER 3: GAIT QUALITY
        self.pitch_params = (0.0, 0.15, 4.0)       # Peak at 0 (upright), ±0.15 rad
        self.arm_position_params = (0.0, 0.4, 3.0) # Peak at 0 (arms down), ±0.4 rad
        self.airborne_params = (0.20, 0.15, 3.0)   # Peak at 20% airborne, ±15%
        self.stride_params = (0.40, 0.15, 4.0)     # Peak at 40cm stride, ±15cm
        self.lateral_params = (0.0, 0.2, 2.0)      # Peak at 0 drift, ±20cm
        
        # TIER 3: DISCRETE STEP REWARD (not Gaussian - binary event)
        self.step_reward = 12.0                     # Bonus per alternating step
        
        # Small control cost (keep actions smooth)
        self.ctrl_cost_weight = 0.001
        
        # Joint indices
        self.shoulder2_right_idx = 25
        self.shoulder2_left_idx = 28

        # Tracking variables
        self.last_left_contact = False
        self.last_right_contact = False
        self.last_swing_foot = None
        self.steps_taken = 0
        self.last_step_x = 0.0
        self.episode_start_x = 0.0
        self.episode_start_y = 0.0
        self.velocity_history = []
        self.velocity_history_maxlen = 50
        self.airborne_history = []
        self.airborne_history_maxlen = 20

    def set_training_phase(self, phase: str, progress: float = 0.0):
        self.training_phase = phase
        self.walking_progress = np.clip(progress, 0.0, 1.0)

    @property
    def healthy_z_range(self):
        return (0.9, 2.0)  # Slightly wider for learning

    @property
    def is_healthy(self):
        z = self.data.qpos[2]
        min_z, max_z = self.healthy_z_range
        healthy_height = min_z < z < max_z
        torso_quat_w = self.data.xquat[self.torso_id][0]
        torso_upright = torso_quat_w > 0.6  # Slightly more lenient
        return healthy_height and torso_upright

    @property
    def contact_forces(self):
        return np.clip(self.data.cfrc_ext, -1.0, 1.0)

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()
        com_inertia = self.data.xipos[self.torso_id].copy()
        com_velocity = self.data.cvel[self.torso_id].copy()
        actuator_forces = self.data.qfrc_actuator.flat.copy()
        return np.concatenate((position, velocity, com_inertia, com_velocity.flat, actuator_forces))

    def reset_model(self):
        noise_low, noise_high = -0.01, 0.01
        qpos = self.init_qpos + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nv)
        qpos[2] = self.height_params[0]  # Start at target height
        self.set_state(qpos, qvel)
        
        self.last_left_contact = False
        self.last_right_contact = False
        self.last_swing_foot = None
        self.steps_taken = 0
        self.last_step_x = self.data.qpos[0]
        self.episode_start_x = self.data.qpos[0]
        self.episode_start_y = self.data.qpos[1]
        self.velocity_history = []
        self.airborne_history = []
        
        return self._get_obs()

    def _check_foot_contact(self, foot_geom_names):
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_name = self.model.geom(contact.geom1).name
            geom2_name = self.model.geom(contact.geom2).name
            if 'floor' in [geom1_name, geom2_name]:
                for foot_geom in foot_geom_names:
                    if foot_geom in [geom1_name, geom2_name]:
                        return True
        return False

    def step(self, action):
        """Execute one timestep with ALL GAUSSIAN rewards."""
        
        x_before = self.data.qpos[0]
        y_before = self.data.qpos[1]
        
        self.do_simulation(action, self.frame_skip)
        
        x_after = self.data.qpos[0]
        y_after = self.data.qpos[1]
        height_z = self.data.qpos[2]
        
        # Velocities
        forward_velocity = (x_after - x_before) / self.dt
        
        # Track velocity history
        self.velocity_history.append(forward_velocity)
        if len(self.velocity_history) > self.velocity_history_maxlen:
            self.velocity_history.pop(0)
        avg_velocity = np.mean(self.velocity_history[-10:]) if len(self.velocity_history) >= 10 else forward_velocity
        
        # Contacts
        left_contact = self._check_foot_contact(self.left_foot_geoms)
        right_contact = self._check_foot_contact(self.right_foot_geoms)
        no_contact = not left_contact and not right_contact
        both_contact = left_contact and right_contact
        single_support = (left_contact != right_contact)
        
        # Track airborne ratio
        self.airborne_history.append(1.0 if no_contact else 0.0)
        if len(self.airborne_history) > self.airborne_history_maxlen:
            self.airborne_history.pop(0)
        airborne_ratio = np.mean(self.airborne_history) if self.airborne_history else 0.0
        
        terminated = not self.is_healthy
        
        # Get torso orientation
        torso_quat = self.data.xquat[self.torso_id]
        torso_quat_w = torso_quat[0]
        
        # Calculate torso pitch (forward lean)
        torso_pitch = 2.0 * np.arcsin(np.clip(torso_quat[2], -1.0, 1.0))
        
        # Get arm positions
        qpos = self.data.qpos
        shoulder2_right = qpos[self.shoulder2_right_idx]
        shoulder2_left = qpos[self.shoulder2_left_idx]
        avg_arm_position = (abs(shoulder2_right) + abs(shoulder2_left)) / 2.0
        
        # Lateral drift from start
        lateral_drift = abs(y_after - self.episode_start_y)
        
        # ================================================================
        # TIER 1: SURVIVAL REWARDS (Gaussian)
        # ================================================================
        
        # Height reward - peak at 1.4m
        height_reward = gaussian_reward(height_z, *self.height_params)
        
        # Upright reward - peak at quat_w = 1.0
        upright_reward = gaussian_reward(torso_quat_w, *self.upright_params)
        
        # Alive bonus - constant while healthy
        alive_reward = self.alive_reward if not terminated else 0.0
        
        tier1_total = height_reward + upright_reward + alive_reward
        
        # ================================================================
        # TIER 2: VELOCITY REWARD (Gaussian)
        # ================================================================
        
        # Velocity reward - peak at 0.5 m/s
        velocity_reward = gaussian_reward(avg_velocity, *self.velocity_params)
        
        tier2_total = velocity_reward
        
        # ================================================================
        # TIER 3: GAIT QUALITY REWARDS (Gaussian + Step Bonus)
        # ================================================================
        
        # Scale gait rewards by curriculum progress
        gait_scale = 0.3 + 0.7 * self.walking_progress  # 0.3 to 1.0
        
        # Pitch reward - peak at 0 (upright torso)
        pitch_reward = gaussian_reward(torso_pitch, *self.pitch_params) * gait_scale
        
        # Arm position reward - peak at 0 (arms at sides)
        arm_reward = gaussian_reward(avg_arm_position, *self.arm_position_params) * gait_scale
        
        # Airborne ratio reward - peak at 20% (walking gait)
        airborne_reward = gaussian_reward(airborne_ratio, *self.airborne_params) * gait_scale
        
        # Lateral position reward - peak at 0 drift (walking straight)
        lateral_reward = gaussian_reward(lateral_drift, *self.lateral_params) * gait_scale
        
        # Step detection and stride reward
        step_bonus = 0.0
        stride_reward = 0.0
        
        if avg_velocity > 0.05:  # Only count steps when moving
            left_touchdown = left_contact and not self.last_left_contact
            right_touchdown = right_contact and not self.last_right_contact
            current_x = self.data.qpos[0]
            
            # Left foot touchdown after right was swinging
            if left_touchdown and self.last_swing_foot == 'right':
                stride_length = current_x - self.last_step_x
                if stride_length > 0.02:  # At least 2cm forward
                    self.steps_taken += 1
                    step_bonus = self.step_reward
                    # Stride length reward - peak at 40cm
                    stride_reward = gaussian_reward(stride_length, *self.stride_params)
                self.last_step_x = current_x
                self.last_swing_foot = 'left'
            
            # Right foot touchdown after left was swinging
            elif right_touchdown and self.last_swing_foot == 'left':
                stride_length = current_x - self.last_step_x
                if stride_length > 0.02:
                    self.steps_taken += 1
                    step_bonus = self.step_reward
                    stride_reward = gaussian_reward(stride_length, *self.stride_params)
                self.last_step_x = current_x
                self.last_swing_foot = 'right'
            
            # Initialize swing foot
            elif left_touchdown and self.last_swing_foot is None:
                self.last_swing_foot = 'left'
                self.last_step_x = current_x
            elif right_touchdown and self.last_swing_foot is None:
                self.last_swing_foot = 'right'
                self.last_step_x = current_x
        
        tier3_total = pitch_reward + arm_reward + airborne_reward + lateral_reward + step_bonus + stride_reward
        
        # ================================================================
        # SMALL CONTROL COST (only non-Gaussian component)
        # ================================================================
        ctrl_cost = -self.ctrl_cost_weight * np.sum(np.square(action))
        
        # ================================================================
        # TOTAL REWARD
        # ================================================================
        
        if self.training_phase == "standing" and self.walking_progress < 0.3:
            # Ultra-simple: just survival rewards
            total_reward = tier1_total + ctrl_cost
            phase_name = "ultra_simple"
        else:
            # Full walking: all tiers
            total_reward = tier1_total + tier2_total + tier3_total + ctrl_cost
            phase_name = "walking"
        
        # Update contact tracking
        self.last_left_contact = left_contact
        self.last_right_contact = right_contact
        
        # ================================================================
        # METRICS (compatible with previous versions)
        # ================================================================
        
        left_foot_z = self.data.site_xpos[self.left_foot_site_id][2]
        right_foot_z = self.data.site_xpos[self.right_foot_site_id][2]
        
        all_metrics = {
            # Base rewards
            'base_reward/healthy': alive_reward,
            'base_reward/ctrl_cost': ctrl_cost,
            'base_reward/contact_cost': 0.0,
            'base_reward/gait_total': tier3_total,
            'base_reward/total_reward': total_reward,
            
            # Environment metrics
            'env_metrics/forward_velocity': forward_velocity,
            'env_metrics/x_position': x_after,
            'env_metrics/y_position': y_after,
            'env_metrics/z_position': height_z,
            'env_metrics/steps_taken': self.steps_taken,
            'env_metrics/left_contact': float(left_contact),
            'env_metrics/right_contact': float(right_contact),
            'env_metrics/no_contact': float(no_contact),
            'env_metrics/both_contact': float(both_contact),
            'env_metrics/single_support': float(single_support),
            'env_metrics/left_foot_height': left_foot_z,
            'env_metrics/right_foot_height': right_foot_z,
            
            # Curriculum
            'curriculum/walking_progress': self.walking_progress,
            'curriculum/alpha_standing': 1.0 - self.walking_progress,
            'curriculum/alpha_walking': self.walking_progress,
            'curriculum/standing_rew': tier1_total,
            'curriculum/walking_rew': tier2_total + tier3_total,
            'curriculum/progressive_forward_weight': gait_scale,
            'curriculum/gait_penalty_scale': gait_scale,
            'curriculum/ultra_simple_mode': 1.0 if phase_name == "ultra_simple" else 0.0,
            
            # Tier 1: Survival (Gaussian)
            'standing_phase/balance_reward': height_reward,
            'standing_phase/height_reward': height_reward,
            'standing_phase/velocity_penalty': 0.0,  # No penalties!
            'standing_phase/torso_upright': upright_reward,
            
            # Tier 2: Velocity (Gaussian)
            'walking_phase/enhanced_forward_reward': velocity_reward,
            'walking_phase/scaled_gait_reward': tier3_total,
            'walking_phase/velocity_tracking': velocity_reward,
            'walking_phase/sustained_speed_bonus': 0.0,
            
            # Ultra simple
            'ultra_simple/balance_reward': height_reward if phase_name == "ultra_simple" else 0.0,
            'ultra_simple/upright_reward': upright_reward if phase_name == "ultra_simple" else 0.0,
            'ultra_simple/neutral_pose_penalty': 0.0,
            
            # Tier 3: Gait Quality (Gaussian) - repurposing old metric names
            'joint_constraints/total_penalty': 0.0,  # No penalties!
            'joint_constraints/shoulder1_penalty': 0.0,
            'joint_constraints/shoulder2_penalty': arm_reward,  # Now a reward!
            'joint_constraints/elbow_penalty': 0.0,
            'joint_constraints/ankle_y_penalty': pitch_reward,  # Torso pitch reward
            'joint_constraints/ankle_x_penalty': airborne_reward,  # Airborne reward
            'joint_constraints/abdomen_penalty': lateral_reward,  # Lateral reward
            'joint_constraints/progress_scale': gait_scale,
            
            # Gait rewards
            'gait_reward/alternation_reward': step_bonus,
            'gait_reward/step_frequency_reward': 0.0,
            'gait_reward/stride_length_reward': stride_reward,
            'gait_reward/static_standing_penalty': 0.0,
            'gait_reward/contact_pattern_rew': airborne_reward,
            'gait_reward/clearance_rew': 0.0,
            'gait_reward/com_smoothness_pen': 0.0,
            'gait_reward/orientation_pen': 0.0,
            'gait_reward/torso_rotation_pen': 0.0,
            'gait_reward/foot_slide_pen': 0.0,
            
            # Arm swing
            'arm_swing/movement_reward': arm_reward,
            'arm_swing/coordination_reward': 0.0,
            'arm_swing/total_reward': arm_reward,
        }
        
        observation = self._get_obs()
        truncated = False
        
        return observation, total_reward, terminated, truncated, all_metrics