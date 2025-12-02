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
        self.walking_progress = 0.0  # 0.0 = pure standing, 1.0 = pure walking
        print(f"Initialized HumanoidWalkEnv in '{training_phase}' phase")
        
        EzPickle.__init__(self, xml_file=xml_file, frame_skip=frame_skip, training_phase=training_phase, **kwargs)

        # OPTIMIZED: Adjusted reward weights for quality over speed
        self.forward_reward_weight = 3.0  # Keep same
        self.ctrl_cost_weight = 0.01  # Keep same
        self.healthy_reward = 2.0  # Keep same
        self.contact_cost_weight = 5e-7
        
        # NEW: Anti-skating penalties
        self.arm_movement_penalty_weight = 0.5  # Penalize excessive arm flailing
        self.torso_rotation_penalty_weight = 0.3  # Penalize torso rotation for momentum
        self.foot_slide_penalty_weight = 1.5  # Penalize sliding feet instead of lifting

        # ENHANCED: Optimized gait reward weights for better step alternation
        self.gait_reward_weight = 0.3  # Keep same
        self.contact_pattern_weight = 2.0  # Keep same
        self.swing_clearance_weight = 2.5  # INCREASED from 1.2 - much stronger emphasis on lifting legs
        self.com_smoothness_weight = 0.6  # Keep same
        self.feet_air_time_penalty = 10.0  # Keep same
        self.orientation_weight = 0.2  # Keep same
        
        # NEW: Step frequency tracking for rhythmic walking
        self.step_frequency_target = 1.8  # Keep same
        self.step_frequency_weight = 1.5  # Keep same
        self.step_time_history = []  # Track time between steps
        
        # NEW: Minimum clearance requirements
        self.min_clearance_height = 0.08  # INCREASED from 0.03 to force real lifting (8cm minimum)
        
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
        self.path_deviation_history = []

        # Posture weights
        self.torso_stability_weight = 0.5
        self.head_stability_weight = 0.3

        # ============================================================
        # NEW: NATURAL JOINT CONSTRAINT WEIGHTS
        # ============================================================
        # These enforce human-like arm and ankle positions
        self.joint_constraint_weight = 2.0  # Master weight for all constraints

        # Individual constraint weights (multiplied by master weight)
        self.shoulder1_constraint_weight = 1.0   # Arm swing limits
        self.shoulder2_constraint_weight = 1.5   # Arm position (keep arms down)
        self.elbow_constraint_weight = 0.8       # Keep elbows straight
        self.ankle_y_constraint_weight = 1.2     # Forward ankle flexion
        self.ankle_x_constraint_weight = 1.0     # Lateral ankle stability

        # Coordinated arm swing weights
        self.arm_swing_reward_weight = 1.5  # Reward for natural arm swing
        self.arm_swing_coordination_weight = 2.0  # Reward for arm-leg coordination

        # Abdomen constraint weights
        self.abdomen_x_constraint_weight = 2.0  # Strong penalty for side bending
        self.abdomen_y_constraint_weight = 1.5  # Moderate penalty for forward/back
        self.abdomen_z_constraint_weight = 1.8  # Strong penalty for twisting

        # Enhanced stride rewards
        self.stride_length_reward_weight = 2.5  # Reward for good stride length
        self.single_support_reward_weight = 3.0  # Reward for clear single support phase
        self.step_completion_bonus = 5.0  # Bonus for completing full step cycle
        self.double_support_penalty_weight = 2.0  # Penalize prolonged double support

        # ============================================================
        # JOINT INDICES (based on XML structure)
        # ============================================================
        # qpos indices (first 7 are free joint: x, y, z, qw, qx, qy, qz)
        # Then body joints in order they appear in XML:
        # Abdomen joint indices (torso bending)
        self.abdomen_x_idx = 9   # Side bending
        self.abdomen_y_idx = 8   # Forward/backward bending  
        self.abdomen_z_idx = 7   # Twisting
        self.shoulder1_right_idx = 24  # shoulder1_right joint
        self.shoulder2_right_idx = 25  # shoulder2_right joint
        self.elbow_right_idx = 26      # elbow_right joint
        self.shoulder1_left_idx = 27   # shoulder1_left joint
        self.shoulder2_left_idx = 28   # shoulder2_left joint
        self.elbow_left_idx = 29       # elbow_left joint
        self.ankle_y_right_idx = 14    # ankle_y_right joint
        self.ankle_x_right_idx = 15    # ankle_x_right joint
        self.ankle_y_left_idx = 20     # ankle_y_left joint
        self.ankle_x_left_idx = 21     # ankle_x_left joint

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
        ankle_y_right = qpos[self.ankle_y_right_idx]
        ankle_y_left = qpos[self.ankle_y_left_idx]
        
        ankle_y_right_penalty = 0.0
        if ankle_y_right < -0.1:
            # Too much dorsiflexion
            ankle_y_right_penalty = -self.ankle_y_constraint_weight * ((ankle_y_right + 0.1) ** 2)
        elif ankle_y_right > 0.0:
            # Plantarflexion
            ankle_y_right_penalty = -self.ankle_y_constraint_weight * (ankle_y_right ** 2)
        
        ankle_y_left_penalty = 0.0
        if ankle_y_left < -0.1:
            ankle_y_left_penalty = -self.ankle_y_constraint_weight * ((ankle_y_left + 0.1) ** 2)
        elif ankle_y_left > 0.0:
            ankle_y_left_penalty = -self.ankle_y_constraint_weight * (ankle_y_left ** 2)
        
        ankle_y_penalty = ankle_y_right_penalty + ankle_y_left_penalty
        total_penalty += ankle_y_penalty
        
        info['joint_constraints/ankle_y_right'] = ankle_y_right
        info['joint_constraints/ankle_y_left'] = ankle_y_left
        info['joint_constraints/ankle_y_penalty'] = ankle_y_penalty
        
        # ============================================================
        # ANKLE_X CONSTRAINTS (No lateral twist: keep at 0)
        # ============================================================
        ankle_x_right = qpos[self.ankle_x_right_idx]
        ankle_x_left = qpos[self.ankle_x_left_idx]
        
        # Quadratic penalty for deviation from 0
        ankle_x_right_penalty = -self.ankle_x_constraint_weight * (ankle_x_right ** 2)
        ankle_x_left_penalty = -self.ankle_x_constraint_weight * (ankle_x_left ** 2)
        
        ankle_x_penalty = ankle_x_right_penalty + ankle_x_left_penalty
        total_penalty += ankle_x_penalty
        
        info['joint_constraints/ankle_x_right'] = ankle_x_right
        info['joint_constraints/ankle_x_left'] = ankle_x_left
        info['joint_constraints/ankle_x_penalty'] = ankle_x_penalty
        
        # ============================================================
        # TOTAL CONSTRAINT PENALTY (apply master weight)
        # ============================================================
        constraint_progress_scale = 0.2 + (0.8 * min(1.0, self.walking_progress * 2))
        total_penalty *= self.joint_constraint_weight * constraint_progress_scale
        info['joint_constraints/total_penalty'] = total_penalty
        info['joint_constraints/progress_scale'] = constraint_progress_scale
        
        return total_penalty, info

    def _calculate_arm_swing_rewards(self, left_contact, right_contact):
        """
        NEW: Reward natural arm swing coordinated with leg movement.
        
        Human walking pattern:
        - Right leg forward → Left arm should swing forward (positive shoulder1)
        - Left leg forward → Right arm should swing forward (positive shoulder1)
        
        Returns:
            reward: Positive reward for natural coordinated arm swing
            info: Metrics dictionary
        """
        info = {}
        reward = 0.0
        
        # Get arm positions
        shoulder1_right = self.data.qpos[self.shoulder1_right_idx]
        shoulder1_left = self.data.qpos[self.shoulder1_left_idx]
        
        # Get arm velocities
        shoulder1_right_vel = self.data.qvel[self.shoulder1_right_idx - 7]  # Adjust for free joint
        shoulder1_left_vel = self.data.qvel[self.shoulder1_left_idx - 7]
        
        # ============================================================
        # 1. REWARD FOR ARM MOVEMENT (any swing is good)
        # ============================================================
        # Encourage arms to move, not stay frozen
        arm_movement = abs(shoulder1_right_vel) + abs(shoulder1_left_vel)
        movement_reward = self.arm_swing_reward_weight * min(arm_movement, 2.0)  # Cap at 2.0
        reward += movement_reward
        info['arm_swing/movement_reward'] = movement_reward
        
        # ============================================================
        # 2. REWARD FOR COORDINATION (opposite arm-leg swing)
        # ============================================================
        # When right leg is in stance (contact), left arm should be forward
        # When left leg is in stance (contact), right arm should be forward
        
        coordination_reward = 0.0
        
        # Right leg stance phase - reward left arm forward
        if right_contact and not left_contact:
            # Left arm forward (positive shoulder1_left) is good
            if shoulder1_left > 0.2:  # At least 0.2 radians forward
                coordination_reward += 1.5 * shoulder1_left
            
            # Right arm should be back/neutral - penalize if too far backward
            if shoulder1_right < 0.0:  # Backward (negative)
                coordination_reward -= 1.0 * abs(shoulder1_right)  # Penalty for backward
            elif shoulder1_right < 0.3:  # Slightly back/neutral is OK
                coordination_reward += 0.3 * (0.3 - shoulder1_right)  # Small reward
        
        # Left leg stance phase - reward right arm forward
        elif left_contact and not right_contact:
            # Right arm forward (positive shoulder1_right) is good
            if shoulder1_right > 0.2:
                coordination_reward += 1.5 * shoulder1_right
            
            # Left arm should be back/neutral - penalize if too far backward
            if shoulder1_left < 0.0:  # Backward (negative)
                coordination_reward -= 1.0 * abs(shoulder1_left)  # Penalty for backward
            elif shoulder1_left < 0.3:  # Slightly back/neutral is OK
                coordination_reward += 0.3 * (0.3 - shoulder1_left)  # Small reward
        
        coordination_reward *= self.arm_swing_coordination_weight
        reward += coordination_reward
        info['arm_swing/coordination_reward'] = coordination_reward
        
        # ============================================================
        # 3. KEEP SHOULDER2 CONSTRAINT (arms at sides, not spread)
        # ============================================================
        # This one we still want - prevents arms from spreading wide
        
        info['arm_swing/shoulder1_right'] = shoulder1_right
        info['arm_swing/shoulder1_left'] = shoulder1_left
        info['arm_swing/total_reward'] = reward
        
        return reward, info

    def _calculate_gait_rewards(self):
        """ENHANCED: Advanced gait rewards with step frequency tracking"""
        gait_reward = 0.0
        info = {}
        
        # Get contact states
        left_contact = self._check_foot_contact_mujoco(self.left_foot_geoms)
        right_contact = self._check_foot_contact_mujoco(self.right_foot_geoms)
        
        info['env_metrics/left_contact'] = float(left_contact)
        info['env_metrics/right_contact'] = float(right_contact)

        # Detect step transitions
        left_touchdown = left_contact and not self.last_left_contact
        right_touchdown = right_contact and not self.last_right_contact
        
        # ENHANCED: Reward alternating steps with step frequency tracking
        alternation_reward = 0.0
        step_frequency_reward = 0.0
        stride_length_reward = 0.0

        # Hip progression
        current_hip_x = self.data.qpos[0]  # Root X position
        hip_progression = current_hip_x - self.last_hip_x
        self.last_hip_x = current_hip_x
        
        if left_touchdown and self.last_swing_foot == 'right':
            self.steps_taken += 1
            self.last_swing_foot = 'left'

            if hip_progression > 0.001:
                alternation_reward = 15.0

                # NEW: Stride length reward
                if hip_progression > 0.05:  # Good stride length (5cm+)
                    stride_length_reward = self.stride_length_reward_weight * min(hip_progression * 20, 5.0)
            
                # NEW: Step completion bonus
                alternation_reward += self.step_completion_bonus
            else:
                alternation_reward = -20.0  # INCREASED penalty from -5.0
            
            # Track step timing
            current_time = self.data.time
            if not hasattr(self, 'last_step_time'):
                self.last_step_time = current_time
            else:
                step_interval = current_time - self.last_step_time
                if not hasattr(self, 'step_time_history'):
                    self.step_time_history = []
                self.step_time_history.append(step_interval)
                if len(self.step_time_history) > 20:
                    self.step_time_history.pop(0)
                self.last_step_time = current_time
                
        elif right_touchdown and self.last_swing_foot == 'left':
            self.steps_taken += 1
            self.last_swing_foot = 'right'

            if hip_progression > 0.001:
                alternation_reward = 5.0
                # NEW: Stride length reward
                if hip_progression > 0.05:
                    stride_length_reward = self.stride_length_reward_weight * min(hip_progression * 20, 5.0)
                
                # NEW: Step completion bonus
                alternation_reward += self.step_completion_bonus
            else:
                alternation_reward = -8.0  # INCREASED penalty from -5.0
            
            # Track step timing
            current_time = self.data.time
            if not hasattr(self, 'last_step_time'):
                self.last_step_time = current_time
            else:
                step_interval = current_time - self.last_step_time
                if not hasattr(self, 'step_time_history'):
                    self.step_time_history = []
                self.step_time_history.append(step_interval)
                if len(self.step_time_history) > 20:
                    self.step_time_history.pop(0)
                self.last_step_time = current_time
                
        elif left_touchdown and self.last_swing_foot is None:
            self.last_swing_foot = 'left'
        elif right_touchdown and self.last_swing_foot is None:
            self.last_swing_foot = 'right'
        
        # NEW: Step frequency reward - encourage consistent rhythm
        if hasattr(self, 'step_time_history') and len(self.step_time_history) >= 3:
            avg_step_interval = np.mean(self.step_time_history[-5:])  # Average of last 5 steps
            actual_frequency = 1.0 / max(avg_step_interval, 0.1)  # Steps per second
            frequency_error = abs(actual_frequency - self.step_frequency_target)
            step_frequency_reward = self.step_frequency_weight * np.exp(-2.0 * frequency_error)
        
        gait_reward += alternation_reward + step_frequency_reward + stride_length_reward
        info['gait_reward/alternation_reward'] = alternation_reward
        info['gait_reward/step_frequency_reward'] = step_frequency_reward
        info['gait_reward/stride_length_reward'] = stride_length_reward

        if hasattr(self, 'steps_taken'):
            if not hasattr(self, 'last_steps_check'):
                self.last_steps_check = 0
                self.last_steps_time = 0.0
            
            # Check every 0.5 seconds
            if self.data.time - self.last_steps_time > 0.5:
                steps_delta = self.steps_taken - self.last_steps_check
                
                if steps_delta == 0:
                    # No steps in 0.5 seconds = BAD
                    static_penalty = -10.0
                    gait_reward += static_penalty
                    info['gait_reward/static_standing_penalty'] = static_penalty
                else:
                    info['gait_reward/static_standing_penalty'] = 0.0
                
                self.last_steps_check = self.steps_taken
                self.last_steps_time = self.data.time
            else:
                info['gait_reward/static_standing_penalty'] = 0.0

        no_feet_contact = not left_contact and not right_contact
        both_feet_contact = left_contact and right_contact
        single_support = (left_contact and not right_contact) or (right_contact and not left_contact)

        info['env_metrics/no_contact'] = float(no_feet_contact)
        info['env_metrics/both_contact'] = float(both_feet_contact)
        info['env_metrics/single_support'] = float(single_support)

        # ENHANCED: Stronger contact pattern rewards
        contact_pattern_reward = 0.0
        if no_feet_contact:
            # Heavy penalty for jumping/hopping
            if not hasattr(self, 'airborne_duration'):
                self.airborne_duration = 0
            self.airborne_duration += 1
            
            # Allow very brief airborne phases (< 3 timesteps), penalize longer
            if self.airborne_duration > 3:
                contact_pattern_reward = -self.feet_air_time_penalty * (self.airborne_duration - 3)
            else:
                contact_pattern_reward = -2.0  # Small penalty for being airborne
        elif single_support:
            # Strong reward for proper single support phase
            if len(self.velocity_history) > 10:
                avg_velocity = np.mean(self.velocity_history[-10:])

                # Check if steps are happening
                if hasattr(self, 'steps_taken') and hasattr(self, 'last_single_support_steps'):
                    steps_since_last = self.steps_taken - self.last_single_support_steps
                else:
                    steps_since_last = 0
                    self.last_single_support_steps = getattr(self, 'steps_taken', 0)

                # Reset counter periodically
                if not hasattr(self, 'single_support_counter'):
                    self.single_support_counter = 0
                self.single_support_counter += 1
                
                if self.single_support_counter > 50:  # Every 50 timesteps
                    self.last_single_support_steps = self.steps_taken
                    self.single_support_counter = 0

                if avg_velocity > 0.1 and steps_since_last > 0:
                    contact_pattern_reward = self.single_support_reward_weight
                    # Extra reward for good velocity during single support
                    if avg_velocity > 0.3:
                        contact_pattern_reward += 2.0
                else:
                    contact_pattern_reward = -5.0  # Penalty for slow movement
            else:
                contact_pattern_reward = 0.0
                
        elif both_feet_contact:
            # Penalize PROLONGED double support (it should be brief)

            # Reset airborne counter when feet touch ground
            if hasattr(self, 'airborne_duration'):
                self.airborne_duration = 0

            # Track how long we've been in double support
            if not hasattr(self, 'double_support_duration'):
                self.double_support_duration = 0
            self.double_support_duration += 1
            
            if self.double_support_duration > 5:  # More than 5 timesteps
                 # Cap penalty to avoid excessive punishment for brief double support
                excess_duration = min(self.double_support_duration - 5, 5)  # Cap at 5 timesteps excess
                contact_pattern_reward = -self.double_support_penalty_weight * excess_duration
            else:
                contact_pattern_reward = -0.5  # Small penalty for double support
        else:
            # Reset double support counter
            if hasattr(self, 'double_support_duration'):
                self.double_support_duration = 0
        
        gait_reward += self.contact_pattern_weight * contact_pattern_reward
        info['gait_reward/contact_pattern_rew'] = self.contact_pattern_weight * contact_pattern_reward

        # Check if feet are moving too far laterally during steps
        left_foot_y = self.data.site_xpos[self.left_foot_site_id][1]
        right_foot_y = self.data.site_xpos[self.right_foot_site_id][1]

        # Feet should stay roughly aligned (not too wide)
        feet_lateral_distance = abs(left_foot_y - right_foot_y)
        if feet_lateral_distance > 0.3:  # Too wide
            gait_reward -= 2.0
            info['gait_reward/wide_stance_penalty'] = -2.0
        elif feet_lateral_distance < 0.1:  # Too narrow (might trip)
            gait_reward -= 1.0
            info['gait_reward/narrow_stance_penalty'] = -1.0

        # ENHANCED: Swing foot clearance reward with higher minimum
        min_clearance_height = self.min_clearance_height  # Now 0.08m instead of 0.03m
        clearance_reward = 0.0
        # achieved_clearance = 0.0

        left_foot_z = self.data.site_xpos[self.left_foot_site_id][2]
        right_foot_z = self.data.site_xpos[self.right_foot_site_id][2]
        info['env_metrics/left_foot_height'] = left_foot_z
        info['env_metrics/right_foot_height'] = right_foot_z

        optimal_clearance = 0.10
        
        if left_contact and not right_contact:  # Right foot swinging
            if right_foot_z >= min_clearance_height:
                clearance_deviation = abs(right_foot_z - optimal_clearance)
                
                if clearance_deviation < 0.04:
                    clearance_reward = 2.0 - (clearance_deviation * 10)
                else:
                    clearance_reward = 0.5
            else:
                clearance_reward = -1.0  # Penalty for dragging foot

        elif right_contact and not left_contact:  # Left foot swinging
            if left_foot_z >= min_clearance_height:
                clearance_deviation = abs(left_foot_z - optimal_clearance)

                if clearance_deviation < 0.04:
                    clearance_reward = 2.0 - (clearance_deviation * 10)
                else:
                    clearance_reward = 0.5
            else:
                clearance_reward = -1.0  # Penalty for dragging foot
        
        gait_reward += self.swing_clearance_weight * clearance_reward
        info['gait_reward/clearance_rew'] = self.swing_clearance_weight * clearance_reward
        # info['gait_reward/clearance_achieved'] = achieved_clearance

        # CoM smoothness penalty
        com_vel = self.data.cvel[self.pelvis_id, :3]
        com_vel_penalty = np.square(com_vel[0]) + np.square(com_vel[2])  # Lateral (X) and vertical (Z)
        gait_reward -= self.com_smoothness_weight * com_vel_penalty
        info['gait_reward/com_smoothness_pen'] = -self.com_smoothness_weight * com_vel_penalty

        # Orientation penalty
        torso_orientation_quat = self.data.xquat[self.torso_id]
        orientation_penalty = np.sum(np.square(torso_orientation_quat[1:]))
        gait_reward -= self.orientation_weight * orientation_penalty
        info['gait_reward/orientation_pen'] = -self.orientation_weight * orientation_penalty
        
        # NEW: Anti-skating penalties
        
        # # 1. Arm movement penalty - discourage using arms for momentum
        # # Get shoulder joint velocities (assuming joints 0-6 are base, 7+ are body joints)
        # if self.data.qvel.shape[0] > 10:  # Make sure we have arm joints
        #     arm_velocities = self.data.qvel[21:27]  # First 6 body joints are typically arms
        #     arm_movement = np.sum(np.square(arm_velocities))
        #     arm_penalty = -self.arm_movement_penalty_weight * arm_movement
        #     gait_reward += arm_penalty
        #     info['gait_reward/arm_penalty'] = arm_penalty
        # else:
        #     info['gait_reward/arm_penalty'] = 0.0
        
        # 2. Torso rotation penalty - discourage using torso twist for momentum
        torso_angular_vel = self.data.cvel[self.torso_id, 3:]  # Angular velocity components
        torso_rotation = np.sum(np.square(torso_angular_vel))
        torso_rotation_penalty = -self.torso_rotation_penalty_weight * torso_rotation
        gait_reward += torso_rotation_penalty
        info['gait_reward/torso_rotation_pen'] = torso_rotation_penalty
        
        # 3. Foot slide penalty - penalize feet moving horizontally while in contact
        foot_slide_penalty = 0.0
        if left_contact:
            # Get left foot horizontal velocity - use correct body name
            try:
                left_foot_vel = self.data.cvel[self.model.body('foot_left').id, :2]  # X,Y velocity
                left_slide = np.sum(np.square(left_foot_vel))
                if left_slide > 0.01:  # Threshold for "sliding"
                    foot_slide_penalty -= left_slide
            except KeyError:
                # If 'foot_left' doesn't exist, skip this penalty
                pass
                
        if right_contact:
            try:
                right_foot_vel = self.data.cvel[self.model.body('foot_right').id, :2]
                right_slide = np.sum(np.square(right_foot_vel))
                if right_slide > 0.01:
                    foot_slide_penalty -= right_slide
            except KeyError:
                # If 'foot_right' doesn't exist, skip this penalty
                pass
        
        foot_slide_penalty *= self.foot_slide_penalty_weight
        gait_reward += foot_slide_penalty
        info['gait_reward/foot_slide_pen'] = foot_slide_penalty

        # Update tracking
        self.last_left_contact = left_contact
        self.last_right_contact = right_contact

        return gait_reward, info

    def step(self, action):
        """Execute one timestep with proper metric handling for ALL phases"""
        # Store position before step
        xy_position_before = self.data.qpos[:2].copy()
        
        # Simulate physics
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.data.qpos[:2].copy()

        # CRITICAL FIX: Consistent forward velocity definition
        x_velocity = (xy_position_after[0] - xy_position_before[0]) / self.dt
        forward_velocity = x_velocity  # Robot faces +X direction

        y_velocity = (xy_position_after[1] - xy_position_before[1]) / self.dt
        lateral_penalty = -self.lateral_penalty_weight * abs(y_velocity)  # Penalize lateral movement

        if not hasattr(self, 'episode_start_y'):
            self.episode_start_y = self.data.qpos[1]

        total_lateral_drift = abs(self.data.qpos[1] - self.episode_start_y)
        if total_lateral_drift > 0.5:  # Drifted more than 0.5m sideways
            lateral_penalty -= 10.0  # Heavy penalty
        
        # Track velocity history
        if not hasattr(self, 'velocity_history'):
            self.velocity_history = []
        self.velocity_history.append(forward_velocity)
        if len(self.velocity_history) > 50:
            self.velocity_history.pop(0)

        # Track heading consistency
        if not hasattr(self, 'path_deviation_history'):
            self.path_deviation_history = []

        # Calculate path straightness
        current_y = self.data.qpos[1]
        self.path_deviation_history.append(current_y)
        if len(self.path_deviation_history) > 50:
            self.path_deviation_history.pop(0)

        # Reward for straight path
        if len(self.path_deviation_history) > 20:
            path_variance = np.var(self.path_deviation_history[-20:])
            if path_variance < 0.01:  # Very straight
                straight_path_bonus = 5.0
            elif path_variance < 0.05:  # Reasonably straight
                straight_path_bonus = 2.0
            else:
                straight_path_bonus = 0.0
        else:
            straight_path_bonus = 0.0

        # Get pelvis lateral velocity
        pelvis_lateral_vel = self.data.cvel[self.pelvis_id, 1]  # Y-axis velocity
        hip_stability_penalty = -2.0 * np.square(pelvis_lateral_vel)

        # Get torso lateral tilt
        torso_quat = self.data.xquat[self.torso_id]
        # Convert to euler angles (approximate)
        torso_roll = 2.0 * np.arcsin(torso_quat[1])  # Roll (side tilt)
        lateral_tilt_penalty = -3.0 * np.square(torso_roll)
        torso_pitch = 2.0 * np.arcsin(np.clip(torso_quat[2], -1.0, 1.0))
        pitch_penalty = -5.0 * np.square(torso_pitch)

        # Get arm joint velocities (adjust indices based on your model)
        # Typically arms are joints 15-22 in humanoid models
        arm_joint_velocities = self.data.qvel[21:27]  # Adjust indices as needed
        arm_flailing_penalty = -self.arm_penalty_weight * np.sum(np.square(arm_joint_velocities))
        # Suggested weight: self.arm_penalty_weight = 0.1

        # Also penalize excessive arm positions
        arm_joint_positions = self.data.qpos[22:28]  # Adjust indices
        # Arms should stay relatively close to neutral position
        arm_position_penalty = -0.05 * np.sum(np.square(arm_joint_positions))

        # Torso upright bonus (you have this but can enhance it)
        torso_quat = self.data.xquat[self.torso_id]
        # w component close to 1 means upright
        upright_bonus = 8.0 * torso_quat[0]  # w component

        # ADD: Extra penalty for significant deviation
        if torso_quat[0] < 0.95:  # Not vertical enough (w < 0.95 means >18° tilt)
            upright_penalty = -5.0 * (0.95 - torso_quat[0]) ** 2
        else:
            upright_penalty = 0.0

        # Penalize torso angular velocity (reduce wobbling)
        torso_angular_vel = self.data.cvel[self.torso_id, 3:]  # Angular velocities
        torso_stability_penalty = -self.torso_stability_weight * np.sum(np.square(torso_angular_vel))

        # Head stability (if you have a head body)
        try:
            head_id = self.model.body('head').id
            head_angular_vel = self.data.cvel[head_id, 3:]
            head_stability_penalty = -self.head_stability_weight * np.sum(np.square(head_angular_vel))
        except:
            head_stability_penalty = 0.0
        
        ctrl_cost = self.ctrl_cost_weight * np.sum(np.square(action))
        contact_cost = self.contact_cost_weight * np.sum(np.square(self.contact_forces))

        # Gait rewards
        gait_reward, gait_info = self._calculate_gait_rewards()

        # NEW: Joint constraint penalties
        joint_constraint_penalty, joint_constraint_info = self._calculate_joint_constraint_penalties()

        # Arm swing rewards
        left_contact = self._check_foot_contact_mujoco(self.left_foot_geoms)
        right_contact = self._check_foot_contact_mujoco(self.right_foot_geoms)
        arm_swing_reward, arm_swing_info = self._calculate_arm_swing_rewards(left_contact, right_contact)

        # Termination and healthy reward
        terminated = not self.is_healthy
        healthy_reward = self.healthy_reward if not terminated else 0.0

        # Calculate standing-focused rewards
        height_z = self.data.qpos[2]
        target_height = 1.4
        height_deviation = abs(height_z - target_height)
        smooth_balance_reward = 10.0 * np.exp(-2.0 * height_deviation)

        torso_quat_w = self.data.xquat[self.torso_id][0]
        orientation_deviation = max(0, 0.9 - torso_quat_w)
        smooth_upright_reward = 5.0 * np.exp(-3.0 * orientation_deviation)
        
        balance_reward = smooth_balance_reward + smooth_upright_reward
        if not self.is_healthy:
            balance_reward = -20.0
        
        height_reward = 3.0 * np.exp(-height_deviation)
        velocity_penalty = -0.03 * np.sum(np.square(self.data.qvel[6:]))
        
        # Initialize ALL metrics with default values first
        all_metrics = {
            # Base rewards - ALWAYS present
            'base_reward/healthy': healthy_reward,
            'base_reward/ctrl_cost': -ctrl_cost,
            'base_reward/contact_cost': -contact_cost,
            'base_reward/gait_total': self.gait_reward_weight * gait_reward,
            'base_reward/total_reward': 0.0,  # Will be set based on phase
            
            # Environment metrics - ALWAYS present
            'env_metrics/forward_velocity': forward_velocity,
            'env_metrics/x_position': xy_position_after[0],
            'env_metrics/y_position': xy_position_after[1],
            'env_metrics/z_position': height_z,
            'env_metrics/steps_taken': getattr(self, 'steps_taken', 0),
            
            # Curriculum metrics - ALWAYS present
            'curriculum/walking_progress': self.walking_progress,
            'curriculum/alpha_standing': 0.0,
            'curriculum/alpha_walking': 0.0,
            'curriculum/standing_rew': 0.0,
            'curriculum/walking_rew': 0.0,
            'curriculum/progressive_forward_weight': 0.0,
            'curriculum/gait_penalty_scale': 0.0,
            'curriculum/ultra_simple_mode': 0.0,
            
            # Standing phase metrics - ALWAYS present
            'standing_phase/balance_reward': 0.0,
            'standing_phase/height_reward': 0.0,
            'standing_phase/velocity_penalty': 0.0,
            'standing_phase/torso_upright': 0.0,
            
            # Walking phase metrics - ALWAYS present
            'walking_phase/enhanced_forward_reward': 0.0,
            'walking_phase/scaled_gait_reward': 0.0,
            'walking_phase/velocity_tracking': 0.0,
            'walking_phase/sustained_speed_bonus': 0.0,
            
            # Ultra simple metrics - ALWAYS present
            'ultra_simple/balance_reward': 0.0,
            'ultra_simple/upright_reward': 0.0,
            'ultra_simple/neutral_pose_penalty': 0.0,

            # NEW: Add joint constraint metrics (after ultra_simple metrics)
            'joint_constraints/total_penalty': 0.0,
            'joint_constraints/shoulder1_penalty': 0.0,
            'joint_constraints/shoulder2_penalty': 0.0,
            'joint_constraints/elbow_penalty': 0.0,
            'joint_constraints/ankle_y_penalty': 0.0,
            'joint_constraints/ankle_x_penalty': 0.0,
            'joint_constraints/shoulder1_right': 0.0,
            'joint_constraints/shoulder1_left': 0.0,
            'joint_constraints/shoulder2_right': 0.0,
            'joint_constraints/shoulder2_left': 0.0,
            'joint_constraints/elbow_right': 0.0,
            'joint_constraints/elbow_left': 0.0,
            'joint_constraints/ankle_y_right': 0.0,
            'joint_constraints/ankle_y_left': 0.0,
            'joint_constraints/ankle_x_right': 0.0,
            'joint_constraints/ankle_x_left': 0.0,
            'joint_constraints/abdomen_penalty': 0.0,
        }
        
        # Merge with gait_info (this may override some values)
        all_metrics.update(gait_info)

        # NEW: Merge with joint constraint info
        all_metrics.update(joint_constraint_info)

        all_metrics.update(arm_swing_info)

        # Now calculate phase-specific rewards and update metrics
        if self.training_phase == "standing" and self.walking_progress < 0.3:
            # Ultra-simple mode
            simple_balance = 10.0 if self.is_healthy else -50.0
            simple_upright = 5.0 if self.data.xquat[self.torso_id][0] > 0.85 else -5.0
            
            joint_positions = self.data.qpos[7:]
            neutral_pose_penalty = -0.1 * np.sum(np.square(joint_positions))
            
            total_reward = simple_balance + simple_upright + neutral_pose_penalty - ctrl_cost
            
            # Update ultra-simple specific metrics
            all_metrics.update({
                'ultra_simple/balance_reward': simple_balance,
                'ultra_simple/upright_reward': simple_upright,
                'ultra_simple/neutral_pose_penalty': neutral_pose_penalty,
                'curriculum/ultra_simple_mode': 1.0,
                'curriculum/alpha_standing': 1.0,
                'curriculum/alpha_walking': 0.0,
                'curriculum/standing_rew': total_reward,
                'base_reward/total_reward': total_reward,
                # Also update standing metrics with current values
                'standing_phase/balance_reward': balance_reward,
                'standing_phase/height_reward': height_reward,
                'standing_phase/velocity_penalty': velocity_penalty,
                'standing_phase/torso_upright': smooth_upright_reward,
            })
            
        else:
            # Standard curriculum blending (standing -> walking transition)
            standing_reward = balance_reward + height_reward + velocity_penalty + smooth_upright_reward - ctrl_cost
            
            # Progressive forward reward
            progressive_forward_weight = self.walking_progress * self.forward_reward_weight
            enhanced_forward_reward = progressive_forward_weight * forward_velocity
            
            # ENHANCED: Velocity tracking reward with better scaling
            MIN_SPEED = 0.1
            if len(self.velocity_history) > 10:
                avg_velocity = np.mean(self.velocity_history[-10:])
                velocity_error = abs(avg_velocity - getattr(self, 'target_velocity', 0.5))
                

                if forward_velocity < MIN_SPEED:
                    speed_penalty = -5.0 * (MIN_SPEED - forward_velocity)
                else:
                    speed_penalty = 0.0

                # Stronger reward for matching target velocity
                velocity_reward = 2.0 * np.exp(-3.0 * velocity_error) + speed_penalty # Increased from 1.0 * exp(-2.0)
            else:
                velocity_reward = 0.0
            
            if all(v > 0.2 for v in self.velocity_history[-20:]):
                sustained_bonus = 10.0  # Bonus for sustained speed
            else:
                sustained_bonus = 0.0
            
            # Scale gait penalties during curriculum
            gait_penalty_scale = max(0.3, self.walking_progress)
            scaled_gait_reward = gait_reward * gait_penalty_scale

            step_rate_bonus = 0.0
            if hasattr(self, 'steps_taken') and len(self.velocity_history) > 100:
                time_elapsed = self.data.time
                if time_elapsed > 1.0:  # After 1 second
                    steps_per_second = self.steps_taken / time_elapsed
                    
                    # Target: at least 1 step per second
                    if steps_per_second > 0.8:
                        step_rate_bonus = 5.0
                    elif steps_per_second > 0.3:
                        step_rate_bonus = 0.0
                    else:
                        step_rate_bonus = -10.0  # Heavy penalty for not stepping
            
            walking_reward = (
                forward_reward
                + velocity_reward
                + sustained_bonus
                + straight_path_bonus
                + upright_penalty
                + head_stability_penalty
                + lateral_penalty
                + hip_stability_penalty
                + lateral_tilt_penalty
                + pitch_penalty
                + arm_flailing_penalty
                + arm_position_penalty
                + upright_bonus
                + torso_stability_penalty
                + joint_constraint_penalty
                + arm_swing_reward
                + step_rate_bonus
            )
            
            # Joint position regularization
            joint_positions = self.data.qpos[7:]
            neutral_pose_penalty = -0.05 * np.sum(np.square(joint_positions))
            standing_reward += neutral_pose_penalty
            walking_reward += neutral_pose_penalty
            
            distance_covered = self.data.qpos[0] - self.episode_start_x
            insufficient_movement_penalty = 0.0
            if len(self.step_time_history) > 20 and distance_covered < 0.1:
                insufficient_movement_penalty = -20.0  # Penalty for not moving sufficiently

            # Blend rewards based on curriculum progress
            alpha_standing = 1.0 - self.walking_progress
            alpha_walking = self.walking_progress
            total_reward = alpha_standing * standing_reward + alpha_walking * walking_reward
            total_reward += insufficient_movement_penalty
            
            # Update all phase-specific metrics
            all_metrics.update({
                'standing_phase/balance_reward': balance_reward,
                'standing_phase/height_reward': height_reward,
                'standing_phase/velocity_penalty': velocity_penalty,
                'standing_phase/torso_upright': smooth_upright_reward,
                'walking_phase/enhanced_forward_reward': enhanced_forward_reward,
                'walking_phase/scaled_gait_reward': scaled_gait_reward,
                'walking_phase/velocity_tracking': velocity_reward,
                'walking_phase/sustained_speed_bonus': sustained_bonus,
                'curriculum/standing_rew': standing_reward,
                'curriculum/walking_rew': walking_reward,
                'curriculum/alpha_standing': alpha_standing,
                'curriculum/alpha_walking': alpha_walking,
                'curriculum/progressive_forward_weight': progressive_forward_weight,
                'curriculum/gait_penalty_scale': gait_penalty_scale,
                'base_reward/total_reward': total_reward,
            })

        # Prepare return values
        observation = self._get_obs()
        truncated = False
        
        return observation, total_reward, terminated, truncated, info