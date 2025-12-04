
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.utils import EzPickle
import numpy as np
import os

DEFAULT_XML_PATH = os.path.join(os.path.dirname(__file__), 'assets', 'humanoid_180_75.xml')


def gaussian(x, mu, sigma):
    """Gaussian function peaked at mu with width sigma."""
    return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


class HumanoidWalkEnv(MujocoEnv, EzPickle):
    
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
        print(f"Initialized in '{training_phase}' phase")
        
        EzPickle.__init__(self, xml_file=xml_file, frame_skip=frame_skip, 
                         training_phase=training_phase, **kwargs)

        # ================================================================
        # REWARD PARAMETERS
        # ================================================================
        
        # CONSTANT: Survival rewards
        self.alive_bonus = 2.0              # Constant for staying alive
        self.target_height = 1.4
        
        # LINEAR + GAUSSIAN: Forward velocity (the key innovation)
        self.linear_velocity_weight = 1.5   # Push to move
        self.gaussian_velocity_weight = 8.0 # Pull to target
        self.target_velocity = 0.5          # Target: 0.5 m/s
        self.velocity_sigma = 0.15          # Tight targeting
        
        # DISCRETE: Step bonus
        self.step_bonus = 12.0              # Per alternating step
        
        # GAUSSIAN: Optimal gait values
        self.stride_target = 0.40           # 40cm optimal stride
        self.stride_sigma = 0.15
        self.stride_weight = 4.0
        
        self.airborne_target = 0.20         # 20% optimal airborne
        self.airborne_sigma = 0.12
        self.airborne_weight = 3.0
        
        # SOFT PENALTIES: Discourage extremes (capped)
        self.arm_penalty_weight = 2.0       # Arms too high
        self.arm_threshold = 0.3            # rad - above this = penalty
        
        self.pitch_penalty_weight = 3.0     # Leaning too much
        self.pitch_threshold = 0.2          # rad (~11°)
        
        self.lateral_penalty_weight = 2.0   # Drifting sideways
        self.lateral_threshold = 0.3        # meters
        
        self.max_penalty = 3.0              # Cap all penalties
        
        # Control cost
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
        qpos[2] = self.target_height
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

    def _soft_penalty(self, value, threshold, weight):
        """Soft penalty that activates beyond threshold, capped at max_penalty."""
        if abs(value) > threshold:
            excess = abs(value) - threshold
            penalty = -weight * (excess ** 2)
            return max(penalty, -self.max_penalty)
        return 0.0

    def step(self, action):
        """Execute one timestep with principled hybrid rewards."""
        
        x_before = self.data.qpos[0]
        y_before = self.data.qpos[1]
        
        self.do_simulation(action, self.frame_skip)
        
        x_after = self.data.qpos[0]
        y_after = self.data.qpos[1]
        height_z = self.data.qpos[2]
        
        # Velocity
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
        
        # Track airborne
        self.airborne_history.append(1.0 if no_contact else 0.0)
        if len(self.airborne_history) > self.airborne_history_maxlen:
            self.airborne_history.pop(0)
        airborne_ratio = np.mean(self.airborne_history) if self.airborne_history else 0.0
        
        terminated = not self.is_healthy
        
        # Torso orientation
        torso_quat = self.data.xquat[self.torso_id]
        torso_quat_w = torso_quat[0]
        
        # Calculate torso pitch (forward lean)
        torso_pitch = 2.0 * np.arcsin(np.clip(torso_quat[2], -1.0, 1.0))
        
        # Arm positions
        qpos = self.data.qpos
        shoulder2_right = qpos[self.shoulder2_right_idx]
        shoulder2_left = qpos[self.shoulder2_left_idx]
        
        # Lateral drift
        lateral_drift = y_after - self.episode_start_y
        
        # ================================================================
        # CONSTANT: Survival
        # ================================================================
        alive_reward = self.alive_bonus if not terminated else 0.0
        
        # ================================================================
        # LINEAR + GAUSSIAN: Forward velocity (key innovation)
        # ================================================================
        # Linear push: encourages any forward movement
        linear_push = self.linear_velocity_weight * max(0, forward_velocity)
        
        # Gaussian pull: attracts to target velocity
        gaussian_pull = self.gaussian_velocity_weight * gaussian(avg_velocity, self.target_velocity, self.velocity_sigma)
        
        velocity_reward = linear_push + gaussian_pull
        
        # ================================================================
        # DISCRETE: Step bonus
        # ================================================================
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
                    step_bonus = self.step_bonus
                    # GAUSSIAN: Stride length reward
                    stride_reward = self.stride_weight * gaussian(stride_length, self.stride_target, self.stride_sigma)
                self.last_step_x = current_x
                self.last_swing_foot = 'left'
            
            # Right foot touchdown after left was swinging
            elif right_touchdown and self.last_swing_foot == 'left':
                stride_length = current_x - self.last_step_x
                if stride_length > 0.02:
                    self.steps_taken += 1
                    step_bonus = self.step_bonus
                    stride_reward = self.stride_weight * gaussian(stride_length, self.stride_target, self.stride_sigma)
                self.last_step_x = current_x
                self.last_swing_foot = 'right'
            
            # Initialize swing foot
            elif left_touchdown and self.last_swing_foot is None:
                self.last_swing_foot = 'left'
                self.last_step_x = current_x
            elif right_touchdown and self.last_swing_foot is None:
                self.last_swing_foot = 'right'
                self.last_step_x = current_x
        
        # ================================================================
        # GAUSSIAN: Airborne ratio (optimal at 20%)
        # ================================================================
        airborne_reward = self.airborne_weight * gaussian(airborne_ratio, self.airborne_target, self.airborne_sigma)
        
        # ================================================================
        # SOFT PENALTIES: Discourage extremes
        # ================================================================
        
        # Arm penalty - raised arms
        arm_penalty = 0.0
        if shoulder2_right > self.arm_threshold:
            arm_penalty += self._soft_penalty(shoulder2_right, self.arm_threshold, self.arm_penalty_weight)
        if shoulder2_left > self.arm_threshold:
            arm_penalty += self._soft_penalty(shoulder2_left, self.arm_threshold, self.arm_penalty_weight)
        
        # Pitch penalty - leaning forward/backward
        pitch_penalty = self._soft_penalty(torso_pitch, self.pitch_threshold, self.pitch_penalty_weight)
        
        # Lateral penalty - drifting sideways
        lateral_penalty = self._soft_penalty(lateral_drift, self.lateral_threshold, self.lateral_penalty_weight)
        
        # Control cost
        ctrl_cost = -self.ctrl_cost_weight * np.sum(np.square(action))
        
        total_penalties = arm_penalty + pitch_penalty + lateral_penalty + ctrl_cost
        
        # ================================================================
        # TOTAL REWARD
        # ================================================================
        
        # Scale gait components by curriculum
        gait_scale = 0.3 + 0.7 * self.walking_progress
        
        if self.training_phase == "standing" and self.walking_progress < 0.3:
            total_reward = alive_reward + ctrl_cost
            phase_name = "ultra_simple"
        else:
            total_reward = (
                alive_reward +
                velocity_reward +
                step_bonus +
                stride_reward * gait_scale +
                airborne_reward * gait_scale +
                total_penalties
            )
            phase_name = "walking"
        
        # Update contact tracking
        self.last_left_contact = left_contact
        self.last_right_contact = right_contact
        
        # ================================================================
        # METRICS
        # ================================================================
        
        left_foot_z = self.data.site_xpos[self.left_foot_site_id][2]
        right_foot_z = self.data.site_xpos[self.right_foot_site_id][2]
        
        all_metrics = {
            # Base rewards
            'base_reward/healthy': alive_reward,
            'base_reward/ctrl_cost': ctrl_cost,
            'base_reward/contact_cost': 0.0,
            'base_reward/gait_total': step_bonus + stride_reward + airborne_reward,
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
            'curriculum/standing_rew': alive_reward,
            'curriculum/walking_rew': velocity_reward + step_bonus,
            'curriculum/progressive_forward_weight': gait_scale,
            'curriculum/gait_penalty_scale': gait_scale,
            'curriculum/ultra_simple_mode': 1.0 if phase_name == "ultra_simple" else 0.0,
            
            # Standing phase
            'standing_phase/balance_reward': alive_reward,
            'standing_phase/height_reward': 0.0,
            'standing_phase/velocity_penalty': 0.0,
            'standing_phase/torso_upright': torso_quat_w,
            
            # Walking phase - velocity breakdown
            'walking_phase/enhanced_forward_reward': velocity_reward,
            'walking_phase/scaled_gait_reward': step_bonus + stride_reward,
            'walking_phase/velocity_tracking': gaussian_pull,
            'walking_phase/sustained_speed_bonus': linear_push,
            
            # Ultra simple
            'ultra_simple/balance_reward': alive_reward if phase_name == "ultra_simple" else 0.0,
            'ultra_simple/upright_reward': 0.0,
            'ultra_simple/neutral_pose_penalty': 0.0,
            
            # Penalties (now actually penalties, not rewards)
            'joint_constraints/total_penalty': total_penalties,
            'joint_constraints/shoulder1_penalty': 0.0,
            'joint_constraints/shoulder2_penalty': arm_penalty,
            'joint_constraints/elbow_penalty': 0.0,
            'joint_constraints/ankle_y_penalty': pitch_penalty,
            'joint_constraints/ankle_x_penalty': 0.0,
            'joint_constraints/abdomen_penalty': lateral_penalty,
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
            'arm_swing/movement_reward': 0.0,
            'arm_swing/coordination_reward': 0.0,
            'arm_swing/total_reward': 0.0,
        }
        
        observation = self._get_obs()
        truncated = False
        
        return observation, total_reward, terminated, truncated, all_metrics