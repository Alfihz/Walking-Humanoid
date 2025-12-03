from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.utils import EzPickle
import numpy as np
import os

DEFAULT_XML_PATH = os.path.join(os.path.dirname(__file__), 'assets', 'humanoid_180_75.xml')


class HumanoidWalkEnv(MujocoEnv, EzPickle):
    
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 100,
    }

    def __init__(self, xml_file=DEFAULT_XML_PATH, frame_skip=5, training_phase="standing", **kwargs):
        # Initialize MujocoEnv
        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            **kwargs
        )

        # ================================================================
        # BODY/SENSOR IDs - Matched to humanoid_180_75.xml
        # ================================================================
        try:
            self.torso_id = self.model.body('torax').id
            self.pelvis_id = self.model.body('pelvis').id
            self.left_foot_site_id = self.model.site('foot_left_site').id
            self.right_foot_site_id = self.model.site('foot_right_site').id
            
            self.left_foot_geoms = ['foot1_left', 'foot2_left']
            self.right_foot_geoms = ['foot1_right', 'foot2_right']
            
        except KeyError as e:
            raise KeyError(f"Required body/site not found in XML: {e}") from e

        # Update observation space
        obs_sample = self._get_obs()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_sample.shape[0],), dtype=np.float32
        )

        # ================================================================
        # TRAINING PHASE & CURRICULUM
        # ================================================================
        self.training_phase = training_phase
        self.walking_progress = 0.0
        print(f"Initialized HumanoidWalkEnv in '{training_phase}' phase")
        
        EzPickle.__init__(self, xml_file=xml_file, frame_skip=frame_skip, 
                         training_phase=training_phase, **kwargs)

        # ================================================================
        # TIER 1: SURVIVAL REWARDS (HIGH PRIORITY)
        # ================================================================
        self.healthy_reward = 5.0           # INCREASED from 2.0
        self.balance_weight = 10.0          # Strong balance incentive
        self.upright_weight = 5.0           # Stay upright
        self.target_height = 1.4
        
        # ================================================================
        # TIER 2: FORWARD PROGRESS REWARDS
        # ================================================================
        self.forward_reward_weight = 3.0    # Base forward reward
        self.target_velocity = 0.5          # Target: 0.5 m/s
        self.velocity_bonus_weight = 2.0    # Bonus for reaching target velocity
        
        # ================================================================
        # TIER 3: STEPPING REWARDS (conditional on forward movement)
        # ================================================================
        self.step_reward = 15.0             # Reward per alternating step
        self.min_velocity_for_step_reward = 0.05  # Must be moving forward
        self.stride_bonus_weight = 5.0      # Bonus for good stride length
        self.target_stride = 0.15           # 15cm target stride
        
        # ================================================================
        # TIER 4: GAIT QUALITY (all penalties CAPPED)
        # ================================================================
        self.max_penalty_per_component = 5.0  # CAP all penalties!
        self.gait_quality_scale = 0.3         # Reduced from 1.0
        
        # Control cost (tiny)
        self.ctrl_cost_weight = 0.001       # Very small
        
        # Joint constraint weights (reduced)
        self.joint_constraint_weight = 0.5  # Reduced from 2.0
        
        # Joint indices
        self.abdomen_x_idx = 9
        self.abdomen_y_idx = 8
        self.abdomen_z_idx = 7
        self.shoulder1_right_idx = 24
        self.shoulder2_right_idx = 25
        self.elbow_right_idx = 26
        self.shoulder1_left_idx = 27
        self.shoulder2_left_idx = 28
        self.elbow_left_idx = 29
        self.ankle_y_right_idx = 14
        self.ankle_x_right_idx = 15
        self.ankle_y_left_idx = 20
        self.ankle_x_left_idx = 21

        # ================================================================
        # GAIT TRACKING VARIABLES
        # ================================================================
        self.last_left_contact = False
        self.last_right_contact = False
        self.last_swing_foot = None
        self.steps_taken = 0
        self.last_step_x = 0.0
        self.episode_start_x = 0.0
        self.episode_start_y = 0.0
        self.velocity_history = []
        self.velocity_history_maxlen = 50

    def set_training_phase(self, phase: str, progress: float = 0.0):
        """Set training phase and curriculum progress."""
        self.training_phase = phase
        self.walking_progress = np.clip(progress, 0.0, 1.0)

    @property
    def healthy_z_range(self):
        return (1.0, 2.0)

    @property
    def is_healthy(self):
        z = self.data.qpos[2]
        min_z, max_z = self.healthy_z_range
        healthy_height = min_z < z < max_z
        torso_quat_w = self.data.xquat[self.torso_id][0]
        torso_upright = torso_quat_w > 0.7
        return healthy_height and torso_upright

    @property
    def contact_forces(self):
        """Get contact forces from simulation."""
        raw_contact_forces = self.data.cfrc_ext
        return np.clip(raw_contact_forces, -1.0, 1.0)

    def _get_obs(self):
        """Get observation vector."""
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()
        com_inertia = self.data.xipos[self.torso_id].copy()
        com_velocity = self.data.cvel[self.torso_id].copy()
        actuator_forces = self.data.qfrc_actuator.flat.copy()
        return np.concatenate((position, velocity, com_inertia, com_velocity.flat, actuator_forces))

    def reset_model(self):
        """Reset to initial standing position."""
        noise_low = -0.01
        noise_high = 0.01

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )
        
        qpos[2] = self.target_height
        self.set_state(qpos, qvel)
        
        # Reset all tracking
        self.last_left_contact = False
        self.last_right_contact = False
        self.last_swing_foot = None
        self.steps_taken = 0
        self.last_step_x = self.data.qpos[0]
        self.episode_start_x = self.data.qpos[0]
        self.episode_start_y = self.data.qpos[1]
        self.velocity_history = []
        
        return self._get_obs()

    def _check_foot_contact(self, foot_geom_names):
        """Check if foot geoms are in contact with ground."""
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_name = self.model.geom(contact.geom1).name
            geom2_name = self.model.geom(contact.geom2).name
            if 'floor' in [geom1_name, geom2_name]:
                for foot_geom in foot_geom_names:
                    if foot_geom in [geom1_name, geom2_name]:
                        return True
        return False

    def _cap_penalty(self, penalty, max_val=None):
        """Cap penalty to prevent overwhelming learning signal."""
        if max_val is None:
            max_val = self.max_penalty_per_component
        return max(penalty, -max_val)

    def step(self, action):
        """Execute one timestep with HIERARCHICAL rewards."""
        
        # Store position before step
        x_before = self.data.qpos[0]
        y_before = self.data.qpos[1]
        
        # Simulate physics
        self.do_simulation(action, self.frame_skip)
        
        x_after = self.data.qpos[0]
        y_after = self.data.qpos[1]
        height_z = self.data.qpos[2]
        
        # Calculate velocities
        forward_velocity = (x_after - x_before) / self.dt
        lateral_velocity = (y_after - y_before) / self.dt
        
        # Track velocity
        self.velocity_history.append(forward_velocity)
        if len(self.velocity_history) > self.velocity_history_maxlen:
            self.velocity_history.pop(0)
        avg_velocity = np.mean(self.velocity_history[-10:]) if len(self.velocity_history) >= 10 else forward_velocity
        
        # Get contacts
        left_contact = self._check_foot_contact(self.left_foot_geoms)
        right_contact = self._check_foot_contact(self.right_foot_geoms)
        
        # Check termination
        terminated = not self.is_healthy
        
        # Get torso orientation
        torso_quat = self.data.xquat[self.torso_id]
        torso_quat_w = torso_quat[0]
        
        # ================================================================
        # TIER 1: SURVIVAL REWARDS (Always active, high priority)
        # ================================================================
        
        # Healthy reward - constant reward for staying alive
        healthy_reward = self.healthy_reward if not terminated else 0.0
        
        # Balance reward - smooth reward for maintaining height
        height_deviation = abs(height_z - self.target_height)
        balance_reward = self.balance_weight * np.exp(-3.0 * height_deviation)
        
        # Upright reward - reward for keeping torso vertical
        upright_reward = self.upright_weight * max(0, torso_quat_w - 0.7) / 0.3  # 0 at 0.7, 5 at 1.0
        
        tier1_reward = healthy_reward + balance_reward + upright_reward
        
        # ================================================================
        # TIER 2: FORWARD PROGRESS (Always active)
        # ================================================================
        
        # Progressive forward weight (scales with curriculum)
        progressive_weight = 0.5 + 0.5 * self.walking_progress  # 0.5 to 1.0
        forward_reward = self.forward_reward_weight * progressive_weight * max(0, forward_velocity)
        
        # Velocity bonus - extra reward for reaching target velocity
        velocity_bonus = 0.0
        if avg_velocity > 0.1:
            velocity_error = abs(avg_velocity - self.target_velocity)
            velocity_bonus = self.velocity_bonus_weight * np.exp(-3.0 * velocity_error)
        
        tier2_reward = forward_reward + velocity_bonus
        
        # ================================================================
        # TIER 3: STEPPING REWARDS (Only when moving forward!)
        # ================================================================
        
        step_reward = 0.0
        stride_bonus = 0.0
        
        # Only reward steps if actually moving forward
        if avg_velocity > self.min_velocity_for_step_reward:
            # Detect step transitions
            left_touchdown = left_contact and not self.last_left_contact
            right_touchdown = right_contact and not self.last_right_contact
            
            current_x = self.data.qpos[0]
            
            # Left foot touchdown after right was swinging
            if left_touchdown and self.last_swing_foot == 'right':
                stride_length = current_x - self.last_step_x
                if stride_length > 0.02:  # At least 2cm forward
                    self.steps_taken += 1
                    step_reward = self.step_reward
                    # Stride bonus for good stride length
                    stride_error = abs(stride_length - self.target_stride)
                    stride_bonus = self.stride_bonus_weight * np.exp(-5.0 * stride_error)
                self.last_step_x = current_x
                self.last_swing_foot = 'left'
            
            # Right foot touchdown after left was swinging
            elif right_touchdown and self.last_swing_foot == 'left':
                stride_length = current_x - self.last_step_x
                if stride_length > 0.02:
                    self.steps_taken += 1
                    step_reward = self.step_reward  # SYMMETRIC!
                    stride_error = abs(stride_length - self.target_stride)
                    stride_bonus = self.stride_bonus_weight * np.exp(-5.0 * stride_error)
                self.last_step_x = current_x
                self.last_swing_foot = 'right'
            
            # Initialize swing foot tracking
            elif left_touchdown and self.last_swing_foot is None:
                self.last_swing_foot = 'left'
                self.last_step_x = current_x
            elif right_touchdown and self.last_swing_foot is None:
                self.last_swing_foot = 'right'
                self.last_step_x = current_x
        
        tier3_reward = step_reward + stride_bonus
        
        # ================================================================
        # TIER 4: GAIT QUALITY (Capped penalties, conditional)
        # ================================================================
        
        # Scale quality penalties by curriculum progress
        quality_scale = self.gait_quality_scale * self.walking_progress
        
        # Control cost (always active but tiny)
        ctrl_cost = self._cap_penalty(-self.ctrl_cost_weight * np.sum(np.square(action)), 1.0)
        
        # Joint constraints (capped)
        joint_penalty = 0.0
        qpos = self.data.qpos
        
        # Only apply joint penalties if moving (don't punish standing)
        if avg_velocity > 0.05:
            # Shoulder penalties (keep arms natural)
            shoulder2_right = qpos[self.shoulder2_right_idx]
            shoulder2_left = qpos[self.shoulder2_left_idx]
            shoulder_penalty = -0.5 * (shoulder2_right ** 2 + shoulder2_left ** 2)
            joint_penalty += self._cap_penalty(shoulder_penalty * quality_scale, 2.0)
            
            # Abdomen penalties (no excessive bending)
            abdomen_x = qpos[self.abdomen_x_idx]
            abdomen_y = qpos[self.abdomen_y_idx]
            abdomen_z = qpos[self.abdomen_z_idx]
            abdomen_penalty = -0.3 * (abdomen_x ** 2 + abdomen_y ** 2 + abdomen_z ** 2)
            joint_penalty += self._cap_penalty(abdomen_penalty * quality_scale, 2.0)
        
        # Lateral drift penalty (capped)
        total_lateral_drift = abs(y_after - self.episode_start_y)
        lateral_penalty = 0.0
        if total_lateral_drift > 0.3:
            lateral_penalty = self._cap_penalty(-2.0 * (total_lateral_drift - 0.3), 3.0)
        
        # Contact pattern penalty - HEAVILY REDUCED AND CAPPED
        contact_penalty = 0.0
        no_contact = not left_contact and not right_contact
        both_contact = left_contact and right_contact
        
        # Only penalize bad contact patterns when trying to walk
        if self.walking_progress > 0.3 and avg_velocity > 0.1:
            if no_contact:
                # Brief airborne is ok, prolonged is bad
                contact_penalty = self._cap_penalty(-1.0, 2.0)  # Much smaller!
            elif both_contact:
                # Double support is ok during weight transfer
                contact_penalty = self._cap_penalty(-0.2, 1.0)  # Very small
        
        tier4_penalty = ctrl_cost + joint_penalty + lateral_penalty + contact_penalty
        
        # ================================================================
        # TOTAL REWARD CALCULATION
        # ================================================================
        
        # Phase-dependent reward mixing
        if self.training_phase == "standing" and self.walking_progress < 0.3:
            # Ultra-simple mode: just balance
            total_reward = tier1_reward + ctrl_cost
            phase_name = "ultra_simple"
        else:
            # Walking mode: all tiers
            total_reward = tier1_reward + tier2_reward + tier3_reward + tier4_penalty
            phase_name = "walking"
        
        # Update contact tracking
        self.last_left_contact = left_contact
        self.last_right_contact = right_contact
        
        # ================================================================
        # METRICS (same names as V22 for comparison)
        # ================================================================
        
        # Foot heights
        left_foot_z = self.data.site_xpos[self.left_foot_site_id][2]
        right_foot_z = self.data.site_xpos[self.right_foot_site_id][2]
        
        all_metrics = {
            # Base rewards
            'base_reward/healthy': healthy_reward,
            'base_reward/ctrl_cost': ctrl_cost,
            'base_reward/contact_cost': 0.0,  # Removed
            'base_reward/gait_total': tier3_reward + tier4_penalty,
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
            'env_metrics/single_support': float((left_contact and not right_contact) or (right_contact and not left_contact)),
            'env_metrics/left_foot_height': left_foot_z,
            'env_metrics/right_foot_height': right_foot_z,
            
            # Curriculum
            'curriculum/walking_progress': self.walking_progress,
            'curriculum/alpha_standing': 1.0 - self.walking_progress,
            'curriculum/alpha_walking': self.walking_progress,
            'curriculum/standing_rew': tier1_reward,
            'curriculum/walking_rew': tier2_reward + tier3_reward,
            'curriculum/progressive_forward_weight': self.forward_reward_weight * (0.5 + 0.5 * self.walking_progress),
            'curriculum/gait_penalty_scale': quality_scale,
            'curriculum/ultra_simple_mode': 1.0 if phase_name == "ultra_simple" else 0.0,
            
            # Standing phase
            'standing_phase/balance_reward': balance_reward,
            'standing_phase/height_reward': 0.0,
            'standing_phase/velocity_penalty': 0.0,
            'standing_phase/torso_upright': upright_reward,
            
            # Walking phase
            'walking_phase/enhanced_forward_reward': forward_reward,
            'walking_phase/scaled_gait_reward': tier3_reward,
            'walking_phase/velocity_tracking': velocity_bonus,
            'walking_phase/sustained_speed_bonus': 0.0,
            
            # Ultra simple
            'ultra_simple/balance_reward': balance_reward if phase_name == "ultra_simple" else 0.0,
            'ultra_simple/upright_reward': upright_reward if phase_name == "ultra_simple" else 0.0,
            'ultra_simple/neutral_pose_penalty': 0.0,
            
            # Joint constraints (simplified)
            'joint_constraints/total_penalty': joint_penalty,
            'joint_constraints/shoulder1_penalty': 0.0,
            'joint_constraints/shoulder2_penalty': 0.0,
            'joint_constraints/elbow_penalty': 0.0,
            'joint_constraints/ankle_y_penalty': 0.0,
            'joint_constraints/ankle_x_penalty': 0.0,
            'joint_constraints/abdomen_penalty': 0.0,
            'joint_constraints/progress_scale': quality_scale,
            
            # Gait rewards (simplified)
            'gait_reward/alternation_reward': step_reward,
            'gait_reward/step_frequency_reward': 0.0,
            'gait_reward/stride_length_reward': stride_bonus,
            'gait_reward/static_standing_penalty': 0.0,
            'gait_reward/contact_pattern_rew': contact_penalty,
            'gait_reward/clearance_rew': 0.0,
            'gait_reward/com_smoothness_pen': 0.0,
            'gait_reward/orientation_pen': 0.0,
            'gait_reward/torso_rotation_pen': 0.0,
            'gait_reward/foot_slide_pen': 0.0,
            
            # Arm swing (simplified)
            'arm_swing/movement_reward': 0.0,
            'arm_swing/coordination_reward': 0.0,
            'arm_swing/total_reward': 0.0,
        }
        
        observation = self._get_obs()
        truncated = False
        
        return observation, total_reward, terminated, truncated, all_metrics