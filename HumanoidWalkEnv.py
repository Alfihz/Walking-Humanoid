"""
HumanoidWalkEnv V21 - Clean Slate Implementation
================================================
A simplified, reward-focused approach to humanoid walking.

Key Principles:
1. REWARD what you want (forward movement, good gait)
2. MINIMIZE penalties (only essential safety constraints)
3. Use GAUSSIAN SHAPING (smooth gradients toward optimal values)
4. SYMMETRIC left/right treatment
5. ALL variables properly initialized

Training Phases:
- Phase 1 (progress < 0.3): Ultra-simple balance
- Phase 2 (0.3 <= progress < 1.0): Standing to walking transition
- Phase 3 (progress = 1.0): Full walking mode

Target Gait:
- Velocity: 0.5 m/s (natural walking speed)
- Stride: 25-35 cm (human-like steps)
- Style: Bipedal walking with alternating steps, not running

Author: Claude (Opus 4.5) for Alfi's thesis project
Date: December 2025
"""

from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.utils import EzPickle
import numpy as np
import os

DEFAULT_XML_PATH = os.path.join(os.path.dirname(__file__), 'assets', 'humanoid_180_75.xml')


class HumanoidWalkEnv(MujocoEnv, EzPickle):
    """
    Humanoid walking environment with clean reward structure.
    
    Reward Philosophy:
    - Primary: Forward velocity (Gaussian peaked at target)
    - Secondary: Step quality (alternation, stride length)
    - Minimal: Control cost only
    """
    
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
            self.torso_id = self.model.body('torax').id  # Note: 'torax' not 'torso'
            self.pelvis_id = self.model.body('pelvis').id
            self.left_foot_site_id = self.model.site('foot_left_site').id
            self.right_foot_site_id = self.model.site('foot_right_site').id
            
            # Foot geom names for contact detection
            self.left_foot_geoms = ['foot1_left', 'foot2_left']
            self.right_foot_geoms = ['foot1_right', 'foot2_right']
            
        except KeyError as e:
            raise KeyError(
                f"Required body/site not found in XML: {e}. "
                "Verify XML contains: bodies 'torax' and 'pelvis', "
                "sites 'foot_left_site' and 'foot_right_site'."
            ) from e

        # Update observation space with actual size
        obs_sample = self._get_obs()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_sample.shape[0],), dtype=np.float32
        )

        # ================================================================
        # TRAINING PHASE & CURRICULUM
        # ================================================================
        self.training_phase = training_phase
        self.walking_progress = 0.0  # 0.0 = standing, 1.0 = full walking
        print(f"[V21] Initialized HumanoidWalkEnv in '{training_phase}' phase")

        # ================================================================
        # PRIMARY REWARD WEIGHTS (What we want most)
        # ================================================================
        self.healthy_reward = 2.0           # Constant reward for staying alive
        self.forward_reward_weight = 5.0    # Base weight for forward movement
        
        # Velocity targeting (Gaussian reward)
        self.target_velocity = 0.5          # m/s - natural walking speed
        self.velocity_reward_weight = 3.0   # Weight for velocity shaping
        self.velocity_sigma = 0.15          # Width of Gaussian (how forgiving)
        
        # ================================================================
        # SECONDARY REWARD WEIGHTS (Gait quality)
        # ================================================================
        # Step alternation
        self.step_reward = 10.0             # Reward for completing a proper step
        self.step_bonus_with_progress = 5.0 # Extra bonus if moving forward
        
        # Stride length (Gaussian reward)
        self.target_stride = 0.30           # 30 cm - natural stride
        self.stride_reward_weight = 8.0     # Weight for good strides
        self.stride_sigma = 0.08            # Width of Gaussian
        self.min_stride_for_reward = 0.05   # 5 cm minimum to count as a step
        
        # Ground contact (walking vs running)
        self.ground_contact_reward = 0.5    # Small reward for having foot on ground
        self.max_airborne_ratio = 0.25      # Max 25% time airborne for walking
        self.airborne_penalty_weight = 2.0  # Penalty if exceeding airborne ratio
        
        # Upright posture
        self.upright_reward_weight = 1.0    # Reward for staying upright
        
        # ================================================================
        # MINIMAL PENALTIES (Safety only)
        # ================================================================
        self.ctrl_cost_weight = 0.001       # Tiny control cost (standard)
        
        # ================================================================
        # GAIT TRACKING VARIABLES
        # ================================================================
        self.last_left_contact = False
        self.last_right_contact = False
        self.last_swing_foot = None         # 'left' or 'right'
        self.steps_taken = 0
        self.last_step_x = 0.0              # X position at last step
        self.episode_start_x = 0.0
        self.episode_start_y = 0.0
        
        # Velocity tracking
        self.velocity_history = []
        self.velocity_history_maxlen = 50
        
        # Airborne tracking
        self.airborne_frames = []
        self.airborne_history_maxlen = 20
        
        # ================================================================
        # STANDING PHASE PARAMETERS
        # ================================================================
        self.target_height = 1.4            # Standing height in meters
        
        EzPickle.__init__(self, xml_file=xml_file, frame_skip=frame_skip, 
                         training_phase=training_phase, **kwargs)

    # ================================================================
    # CURRICULUM CONTROL
    # ================================================================
    def set_training_phase(self, phase: str, progress: float = 0.0):
        """Set training phase and curriculum progress."""
        self.training_phase = phase
        self.walking_progress = np.clip(progress, 0.0, 1.0)

    # ================================================================
    # HEALTH CHECK
    # ================================================================
    @property
    def healthy_z_range(self):
        return (1.0, 2.0)

    @property
    def is_healthy(self):
        z = self.data.qpos[2]
        min_z, max_z = self.healthy_z_range
        healthy_height = min_z < z < max_z
        
        # Check torso is relatively upright
        torso_quat_w = self.data.xquat[self.torso_id][0]
        torso_upright = torso_quat_w > 0.7
        
        return healthy_height and torso_upright

    # ================================================================
    # OBSERVATION
    # ================================================================
    def _get_obs(self):
        """Get observation vector."""
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()
        
        com_inertia = self.data.xipos[self.torso_id].copy()
        com_velocity = self.data.cvel[self.torso_id].copy()
        
        actuator_forces = self.data.qfrc_actuator.flat.copy()
        
        return np.concatenate((
            position,
            velocity,
            com_inertia,
            com_velocity.flat,
            actuator_forces,
        ))

    # ================================================================
    # RESET
    # ================================================================
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
        
        # Ensure standing height
        qpos[2] = self.target_height
        
        self.set_state(qpos, qvel)
        
        # Reset gait tracking
        self.last_left_contact = False
        self.last_right_contact = False
        self.last_swing_foot = None
        self.steps_taken = 0
        self.last_step_x = self.data.qpos[0]
        self.episode_start_x = self.data.qpos[0]
        self.episode_start_y = self.data.qpos[1]
        self.velocity_history = []
        self.airborne_frames = []
        
        return self._get_obs()

    # ================================================================
    # CONTACT DETECTION
    # ================================================================
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

    # ================================================================
    # GAUSSIAN REWARD HELPER
    # ================================================================
    def _gaussian_reward(self, value, target, sigma, weight):
        """
        Gaussian reward that peaks at target value.
        Returns: weight * exp(-((value - target)^2) / (2 * sigma^2))
        
        This provides smooth gradients toward the optimal value.
        """
        return weight * np.exp(-((value - target) ** 2) / (2 * sigma ** 2))

    # ================================================================
    # MAIN STEP FUNCTION
    # ================================================================
    def step(self, action):
        """Execute one timestep with clean reward structure."""

        # ============================================================
        # INITIALIZE ALL METRICS TO 0 (prevents logging errors)
        # ============================================================
        
        info = {
            # Reward components
            'rewards/total': 0.0,
            'rewards/healthy': 0.0,
            'rewards/ctrl_cost': 0.0,
            'rewards/forward_base': 0.0,
            'rewards/velocity_shaping': 0.0,
            'rewards/step': 0.0,
            'rewards/stride': 0.0,
            'rewards/ground_contact': 0.0,
            'rewards/airborne_penalty': 0.0,
            'rewards/upright': 0.0,
            'rewards/lateral_penalty': 0.0,
            'rewards/balance': 0.0,
            'rewards/standing_blend': 0.0,
            'rewards/walking_blend': 0.0,
            
            # Environment metrics
            'env/forward_velocity': 0.0,
            'env/x_position': 0.0,
            'env/y_position': 0.0,
            'env/z_position': 0.0,
            'env/steps_taken': 0,
            'env/left_contact': 0.0,
            'env/right_contact': 0.0,
            'env/both_contact': 0.0,
            'env/no_contact': 0.0,
            'env/left_foot_height': 0.0,
            'env/right_foot_height': 0.0,
            
            # Gait metrics
            'gait/stride_length': 0.0,
            'gait/airborne_ratio': 0.0,
            
            # Curriculum
            'curriculum/walking_progress': self.walking_progress,
            'phase': 'init',
        }
        
        # Store position before step
        x_before = self.data.qpos[0]
        y_before = self.data.qpos[1]
        
        # Simulate physics
        self.do_simulation(action, self.frame_skip)
        
        x_after = self.data.qpos[0]
        y_after = self.data.qpos[1]
        
        # Calculate velocities
        forward_velocity = (x_after - x_before) / self.dt
        lateral_velocity = (y_after - y_before) / self.dt
        
        # Update velocity history
        self.velocity_history.append(forward_velocity)
        if len(self.velocity_history) > self.velocity_history_maxlen:
            self.velocity_history.pop(0)
        
        # Get foot contacts
        left_contact = self._check_foot_contact(self.left_foot_geoms)
        right_contact = self._check_foot_contact(self.right_foot_geoms)
        
        # Update airborne tracking
        no_contact = not left_contact and not right_contact
        self.airborne_frames.append(1.0 if no_contact else 0.0)
        if len(self.airborne_frames) > self.airborne_history_maxlen:
            self.airborne_frames.pop(0)
        
        # Check termination
        terminated = not self.is_healthy
        
        # ============================================================
        # REWARD CALCULATION
        # ============================================================
        
        # --- Control Cost (minimal, always applied) ---
        ctrl_cost = self.ctrl_cost_weight * np.sum(np.square(action))
        info['rewards/ctrl_cost'] = -ctrl_cost
        
        # --- Healthy Reward ---
        healthy_reward = self.healthy_reward if not terminated else 0.0
        info['rewards/healthy'] = healthy_reward
        
        # ============================================================
        # PHASE-DEPENDENT REWARDS
        # ============================================================
        
        if self.walking_progress < 0.3:
            # ====================
            # PHASE 1: ULTRA-SIMPLE BALANCE
            # ====================
            # Just learn to stand upright
            
            height_z = self.data.qpos[2]
            height_error = abs(height_z - self.target_height)
            balance_reward = 5.0 * np.exp(-2.0 * height_error)
            
            torso_quat_w = self.data.xquat[self.torso_id][0]
            upright_reward = 5.0 if torso_quat_w > 0.9 else 2.0 * torso_quat_w
            
            total_reward = balance_reward + upright_reward + healthy_reward - ctrl_cost
            
            info['phase'] = 'ultra_simple'
            info['rewards/balance'] = balance_reward
            info['rewards/upright'] = upright_reward
            info['rewards/total'] = total_reward
            
        else:
            # ====================
            # PHASE 2 & 3: WALKING
            # ====================
            
            # Blend factor: how much to weight walking vs standing rewards
            walk_blend = min(1.0, (self.walking_progress - 0.3) / 0.7)
            
            # --- Forward Movement Reward ---
            # Base reward proportional to forward velocity
            forward_reward = self.forward_reward_weight * max(0, forward_velocity)
            info['rewards/forward_base'] = forward_reward
            
            # --- Velocity Shaping (Gaussian) ---
            # Reward peaks at target velocity, falls off smoothly
            if len(self.velocity_history) > 10:
                avg_velocity = np.mean(self.velocity_history[-10:])
                velocity_reward = self._gaussian_reward(
                    avg_velocity, 
                    self.target_velocity, 
                    self.velocity_sigma, 
                    self.velocity_reward_weight
                )
            else:
                velocity_reward = 0.0
            info['rewards/velocity_shaping'] = velocity_reward
            
            # --- Step Detection & Rewards ---
            step_reward = 0.0
            stride_reward = 0.0
            
            # Detect step transitions
            left_touchdown = left_contact and not self.last_left_contact
            right_touchdown = right_contact and not self.last_right_contact
            
            current_x = self.data.qpos[0]
            
            # Left foot touchdown after right was swinging
            if left_touchdown and self.last_swing_foot == 'right':
                self.steps_taken += 1
                stride_length = current_x - self.last_step_x
                
                if stride_length > self.min_stride_for_reward:
                    # Base step reward (SYMMETRIC)
                    step_reward = self.step_reward
                    
                    # Bonus for forward progress
                    step_reward += self.step_bonus_with_progress
                    
                    # Stride length shaping (Gaussian)
                    stride_reward = self._gaussian_reward(
                        stride_length,
                        self.target_stride,
                        self.stride_sigma,
                        self.stride_reward_weight
                    )
                    
                    info['gait/stride_length'] = stride_length
                
                self.last_step_x = current_x
                self.last_swing_foot = 'left'
            
            # Right foot touchdown after left was swinging
            elif right_touchdown and self.last_swing_foot == 'left':
                self.steps_taken += 1
                stride_length = current_x - self.last_step_x
                
                if stride_length > self.min_stride_for_reward:
                    # Base step reward (SYMMETRIC - same as left!)
                    step_reward = self.step_reward
                    
                    # Bonus for forward progress
                    step_reward += self.step_bonus_with_progress
                    
                    # Stride length shaping (Gaussian)
                    stride_reward = self._gaussian_reward(
                        stride_length,
                        self.target_stride,
                        self.stride_sigma,
                        self.stride_reward_weight
                    )
                    
                    info['gait/stride_length'] = stride_length
                
                self.last_step_x = current_x
                self.last_swing_foot = 'right'
            
            # Initialize swing foot tracking
            elif left_touchdown and self.last_swing_foot is None:
                self.last_swing_foot = 'left'
                self.last_step_x = current_x
            elif right_touchdown and self.last_swing_foot is None:
                self.last_swing_foot = 'right'
                self.last_step_x = current_x
            
            info['rewards/step'] = step_reward
            info['rewards/stride'] = stride_reward
            
            # --- Ground Contact Reward (Walking, not Running) ---
            ground_reward = 0.0
            airborne_penalty = 0.0
            
            if left_contact or right_contact:
                ground_reward = self.ground_contact_reward
            
            # Check airborne ratio
            if len(self.airborne_frames) >= 10:
                airborne_ratio = np.mean(self.airborne_frames)
                info['gait/airborne_ratio'] = airborne_ratio
                
                if airborne_ratio > self.max_airborne_ratio:
                    # Gentle penalty for running instead of walking
                    excess = airborne_ratio - self.max_airborne_ratio
                    airborne_penalty = self.airborne_penalty_weight * excess
            
            info['rewards/ground_contact'] = ground_reward
            info['rewards/airborne_penalty'] = -airborne_penalty
            
            # --- Upright Bonus ---
            torso_quat_w = self.data.xquat[self.torso_id][0]
            upright_reward = self.upright_reward_weight * max(0, torso_quat_w - 0.8) * 5
            info['rewards/upright'] = upright_reward
            
            # --- Lateral Drift Penalty (gentle) ---
            total_lateral_drift = abs(self.data.qpos[1] - self.episode_start_y)
            lateral_penalty = 0.0
            if total_lateral_drift > 0.5:
                lateral_penalty = 1.0 * (total_lateral_drift - 0.5)
            info['rewards/lateral_penalty'] = -lateral_penalty
            
            # --- Standing Baseline (blended out as walking increases) ---
            height_z = self.data.qpos[2]
            height_error = abs(height_z - self.target_height)
            standing_reward = 3.0 * np.exp(-2.0 * height_error)
            
            # --- Combine Walking Rewards ---
            walking_reward = (
                forward_reward
                + velocity_reward
                + step_reward
                + stride_reward
                + ground_reward
                - airborne_penalty
                + upright_reward
                - lateral_penalty
            )
            
            # Blend standing and walking rewards
            total_reward = (
                healthy_reward
                - ctrl_cost
                + (1 - walk_blend) * standing_reward
                + walk_blend * walking_reward
            )
            
            info['phase'] = 'walking'
            info['rewards/standing_blend'] = (1 - walk_blend) * standing_reward
            info['rewards/walking_blend'] = walk_blend * walking_reward
            info['rewards/total'] = total_reward
        
        # ============================================================
        # UPDATE TRACKING & RETURN
        # ============================================================
        self.last_left_contact = left_contact
        self.last_right_contact = right_contact
        
        # Common metrics
        info['env/forward_velocity'] = forward_velocity
        info['env/x_position'] = x_after
        info['env/y_position'] = y_after
        info['env/z_position'] = self.data.qpos[2]
        info['env/steps_taken'] = self.steps_taken
        info['env/left_contact'] = float(left_contact)
        info['env/right_contact'] = float(right_contact)
        info['env/both_contact'] = float(left_contact and right_contact)
        info['env/no_contact'] = float(no_contact)
        info['curriculum/walking_progress'] = self.walking_progress
        
        # Foot heights
        info['env/left_foot_height'] = self.data.site_xpos[self.left_foot_site_id][2]
        info['env/right_foot_height'] = self.data.site_xpos[self.right_foot_site_id][2]
        
        observation = self._get_obs()
        truncated = False
        
        return observation, total_reward, terminated, truncated, info