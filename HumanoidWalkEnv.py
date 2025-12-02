"""
HumanoidWalkEnv V17 - FIXED VERSION
===================================
Original V17 reward structure with all bugs fixed.

Fixes Applied:
1. All missing variables initialized
2. Function names corrected (_check_foot_contact_mujoco -> _check_foot_contact)
3. Asymmetric step rewards fixed (both 10.0)
4. Returns correct info dict (all_metrics, not info)
5. Variable naming fixed (qpos -> self.data.qpos)
6. total_penalty properly initialized
7. forward_reward properly defined

Original V17 Features Preserved:
- Curriculum-scaled forward reward (walking_progress * weight * velocity)
- Speed governors and penalties
- Gait quality rewards with step alternation
- Joint constraint system
- Anti-skating penalties
"""

from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.utils import EzPickle
import numpy as np
import os

DEFAULT_XML_PATH = os.path.join(os.path.dirname(__file__), 'assets', 'humanoid_180_75.xml')


class HumanoidWalkEnv(MujocoEnv, EzPickle):
    """
    V17 Fixed - Humanoid walking environment with curriculum learning.
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
        print(f"[V17-Fixed] Initialized HumanoidWalkEnv in '{training_phase}' phase")
        
        EzPickle.__init__(self, xml_file=xml_file, frame_skip=frame_skip, 
                         training_phase=training_phase, **kwargs)

        # ================================================================
        # CORE REWARD WEIGHTS (V17 Original)
        # ================================================================
        self.forward_reward_weight = 3.0
        self.ctrl_cost_weight = 0.01
        self.healthy_reward = 2.0
        self.contact_cost_weight = 5e-7
        
        # ================================================================
        # FIX: Initialize ALL missing variables
        # ================================================================
        self.target_height = 1.4
        self.target_velocity = 0.5
        self.velocity_history_maxlen = 50
        self.airborne_history_maxlen = 20
        self.lateral_penalty_weight = 0.5
        self.arm_penalty_weight = 0.1
        
        # Anti-skating penalties
        self.arm_movement_penalty_weight = 0.5
        self.torso_rotation_penalty_weight = 0.3
        self.foot_slide_penalty_weight = 1.5

        # Gait reward weights
        self.gait_reward_weight = 0.3
        self.contact_pattern_weight = 2.0
        self.swing_clearance_weight = 2.5
        self.com_smoothness_weight = 0.6
        self.feet_air_time_penalty = 10.0
        self.orientation_weight = 0.2
        
        # Step frequency
        self.step_frequency_target = 1.8
        self.step_frequency_weight = 1.5
        self.step_time_history = []
        
        # Clearance
        self.min_clearance_height = 0.08

        # Posture weights
        self.torso_stability_weight = 0.5
        self.head_stability_weight = 0.3

        # Joint constraint weights
        self.joint_constraint_weight = 2.0
        self.shoulder1_constraint_weight = 1.0
        self.shoulder2_constraint_weight = 1.5
        self.elbow_constraint_weight = 0.8
        self.ankle_y_constraint_weight = 1.2
        self.ankle_x_constraint_weight = 1.0
        self.arm_swing_reward_weight = 1.5
        self.arm_swing_coordination_weight = 2.0
        self.abdomen_x_constraint_weight = 2.0
        self.abdomen_y_constraint_weight = 1.5
        self.abdomen_z_constraint_weight = 1.8

        # Enhanced stride rewards
        self.stride_length_reward_weight = 2.5
        self.single_support_reward_weight = 3.0
        self.step_completion_bonus = 5.0
        self.double_support_penalty_weight = 2.0

        # ================================================================
        # JOINT INDICES
        # ================================================================
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
        self.path_deviation_history = []
        self.velocity_history = []
        self.airborne_frames = []
        self.last_hip_x = 0.0  # FIX: Initialize this
        self.airborne_duration = 0
        self.double_support_duration = 0
        self.last_step_time = 0.0
        self.last_steps_check = 0
        self.last_steps_time = 0.0
        self.single_support_counter = 0
        self.last_single_support_steps = 0

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
        contact_forces = np.clip(raw_contact_forces, -1.0, 1.0)
        return contact_forces

    def _get_obs(self):
        """Get observation vector."""
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()
        com_inertia = self.data.xipos[self.torso_id].copy()
        com_velocity = self.data.cvel[self.torso_id].copy()
        actuator_forces = self.data.qfrc_actuator.flat.copy()
        
        return np.concatenate((
            position, velocity, com_inertia,
            com_velocity.flat, actuator_forces,
        ))

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
        self.airborne_frames = []
        self.path_deviation_history = []
        self.last_hip_x = self.data.qpos[0]
        self.step_time_history = []
        self.airborne_duration = 0
        self.double_support_duration = 0
        self.last_step_time = 0.0
        self.last_steps_check = 0
        self.last_steps_time = 0.0
        self.single_support_counter = 0
        self.last_single_support_steps = 0
        
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

    def _calculate_joint_constraint_penalties(self):
        """Calculate penalties for unnatural joint positions."""
        total_penalty = 0.0
        info = {}
        
        qpos = self.data.qpos  # FIX: Use self.data.qpos
        
        # Shoulder1 constraints (arm swing limits)
        shoulder1_right = qpos[self.shoulder1_right_idx]
        shoulder1_left = qpos[self.shoulder1_left_idx]
        
        # Allow natural arm swing range: -0.5 to 0.8 radians
        shoulder1_penalty = 0.0
        if shoulder1_right < -0.5:
            shoulder1_penalty += self.shoulder1_constraint_weight * ((shoulder1_right + 0.5) ** 2)
        elif shoulder1_right > 0.8:
            shoulder1_penalty += self.shoulder1_constraint_weight * ((shoulder1_right - 0.8) ** 2)
            
        if shoulder1_left < -0.5:
            shoulder1_penalty += self.shoulder1_constraint_weight * ((shoulder1_left + 0.5) ** 2)
        elif shoulder1_left > 0.8:
            shoulder1_penalty += self.shoulder1_constraint_weight * ((shoulder1_left - 0.8) ** 2)
        
        total_penalty -= shoulder1_penalty
        info['joint_constraints/shoulder1_penalty'] = -shoulder1_penalty
        info['joint_constraints/shoulder1_right'] = shoulder1_right
        info['joint_constraints/shoulder1_left'] = shoulder1_left

        # Shoulder2 constraints (keep arms at sides)
        shoulder2_right = qpos[self.shoulder2_right_idx]
        shoulder2_left = qpos[self.shoulder2_left_idx]
        
        shoulder2_penalty = self.shoulder2_constraint_weight * (shoulder2_right ** 2 + shoulder2_left ** 2)
        total_penalty -= shoulder2_penalty
        
        info['joint_constraints/shoulder2_penalty'] = -shoulder2_penalty
        info['joint_constraints/shoulder2_right'] = shoulder2_right
        info['joint_constraints/shoulder2_left'] = shoulder2_left

        # Elbow constraints
        elbow_right = qpos[self.elbow_right_idx]
        elbow_left = qpos[self.elbow_left_idx]
        
        elbow_penalty = 0.0
        if elbow_right < 0:
            elbow_penalty += self.elbow_constraint_weight * (elbow_right ** 2)
        if elbow_left < 0:
            elbow_penalty += self.elbow_constraint_weight * (elbow_left ** 2)
        
        total_penalty -= elbow_penalty
        info['joint_constraints/elbow_penalty'] = -elbow_penalty
        info['joint_constraints/elbow_right'] = elbow_right
        info['joint_constraints/elbow_left'] = elbow_left

        # Ankle Y constraints
        ankle_y_right = qpos[self.ankle_y_right_idx]
        ankle_y_left = qpos[self.ankle_y_left_idx]
        
        ankle_y_penalty = 0.0
        if ankle_y_right < -0.1:
            ankle_y_penalty += self.ankle_y_constraint_weight * ((ankle_y_right + 0.1) ** 2)
        elif ankle_y_right > 0.0:
            ankle_y_penalty += self.ankle_y_constraint_weight * (ankle_y_right ** 2)
        
        if ankle_y_left < -0.1:
            ankle_y_penalty += self.ankle_y_constraint_weight * ((ankle_y_left + 0.1) ** 2)
        elif ankle_y_left > 0.0:
            ankle_y_penalty += self.ankle_y_constraint_weight * (ankle_y_left ** 2)
        
        total_penalty -= ankle_y_penalty
        info['joint_constraints/ankle_y_penalty'] = -ankle_y_penalty
        info['joint_constraints/ankle_y_right'] = ankle_y_right
        info['joint_constraints/ankle_y_left'] = ankle_y_left

        # Ankle X constraints
        ankle_x_right = qpos[self.ankle_x_right_idx]
        ankle_x_left = qpos[self.ankle_x_left_idx]
        
        ankle_x_penalty = self.ankle_x_constraint_weight * (ankle_x_right ** 2 + ankle_x_left ** 2)
        total_penalty -= ankle_x_penalty
        
        info['joint_constraints/ankle_x_penalty'] = -ankle_x_penalty
        info['joint_constraints/ankle_x_right'] = ankle_x_right
        info['joint_constraints/ankle_x_left'] = ankle_x_left

        # Abdomen constraints
        abdomen_x = qpos[self.abdomen_x_idx]
        abdomen_y = qpos[self.abdomen_y_idx]
        abdomen_z = qpos[self.abdomen_z_idx]
        
        abdomen_penalty = (
            self.abdomen_x_constraint_weight * (abdomen_x ** 2) +
            self.abdomen_y_constraint_weight * (abdomen_y ** 2) +
            self.abdomen_z_constraint_weight * (abdomen_z ** 2)
        )
        total_penalty -= abdomen_penalty
        info['joint_constraints/abdomen_penalty'] = -abdomen_penalty

        # Apply curriculum scaling
        constraint_progress_scale = 0.2 + (0.8 * min(1.0, self.walking_progress * 2))
        total_penalty *= self.joint_constraint_weight * constraint_progress_scale
        
        info['joint_constraints/total_penalty'] = total_penalty
        info['joint_constraints/progress_scale'] = constraint_progress_scale
        
        return total_penalty, info

    def _calculate_arm_swing_rewards(self, left_contact, right_contact):
        """Reward natural arm swing coordinated with leg movement."""
        info = {}
        reward = 0.0
        
        shoulder1_right = self.data.qpos[self.shoulder1_right_idx]
        shoulder1_left = self.data.qpos[self.shoulder1_left_idx]
        
        shoulder1_right_vel = self.data.qvel[self.shoulder1_right_idx - 7]
        shoulder1_left_vel = self.data.qvel[self.shoulder1_left_idx - 7]
        
        # Arm movement reward
        arm_movement = abs(shoulder1_right_vel) + abs(shoulder1_left_vel)
        movement_reward = self.arm_swing_reward_weight * min(arm_movement, 2.0)
        reward += movement_reward
        info['arm_swing/movement_reward'] = movement_reward
        
        # Coordination reward
        coordination_reward = 0.0
        
        if right_contact and not left_contact:
            if shoulder1_left > 0.2:
                coordination_reward += 1.5 * shoulder1_left
            if shoulder1_right < 0.0:
                coordination_reward -= 1.0 * abs(shoulder1_right)
            elif shoulder1_right < 0.3:
                coordination_reward += 0.3 * (0.3 - shoulder1_right)
        
        elif left_contact and not right_contact:
            if shoulder1_right > 0.2:
                coordination_reward += 1.5 * shoulder1_right
            if shoulder1_left < 0.0:
                coordination_reward -= 1.0 * abs(shoulder1_left)
            elif shoulder1_left < 0.3:
                coordination_reward += 0.3 * (0.3 - shoulder1_left)
        
        coordination_reward *= self.arm_swing_coordination_weight
        reward += coordination_reward
        info['arm_swing/coordination_reward'] = coordination_reward
        
        info['arm_swing/shoulder1_right'] = shoulder1_right
        info['arm_swing/shoulder1_left'] = shoulder1_left
        info['arm_swing/total_reward'] = reward
        
        return reward, info

    def _calculate_gait_rewards(self):
        """Calculate gait quality rewards."""
        gait_reward = 0.0
        info = {}
        
        left_contact = self._check_foot_contact(self.left_foot_geoms)
        right_contact = self._check_foot_contact(self.right_foot_geoms)
        
        info['env_metrics/left_contact'] = float(left_contact)
        info['env_metrics/right_contact'] = float(right_contact)

        # Step detection
        left_touchdown = left_contact and not self.last_left_contact
        right_touchdown = right_contact and not self.last_right_contact
        
        alternation_reward = 0.0
        step_frequency_reward = 0.0
        stride_length_reward = 0.0

        current_hip_x = self.data.qpos[0]
        hip_progression = current_hip_x - self.last_hip_x
        self.last_hip_x = current_hip_x
        
        # FIX: Symmetric step rewards (both 10.0, not 15.0 vs 5.0)
        if left_touchdown and self.last_swing_foot == 'right':
            self.steps_taken += 1
            self.last_swing_foot = 'left'

            if hip_progression > 0.001:
                alternation_reward = 10.0  # FIX: Was 15.0
                if hip_progression > 0.05:
                    stride_length_reward = self.stride_length_reward_weight * min(hip_progression * 20, 5.0)
                alternation_reward += self.step_completion_bonus
            else:
                alternation_reward = -15.0
            
            self._update_step_timing()
                
        elif right_touchdown and self.last_swing_foot == 'left':
            self.steps_taken += 1
            self.last_swing_foot = 'right'

            if hip_progression > 0.001:
                alternation_reward = 10.0  # FIX: Was 5.0
                if hip_progression > 0.05:
                    stride_length_reward = self.stride_length_reward_weight * min(hip_progression * 20, 5.0)
                alternation_reward += self.step_completion_bonus
            else:
                alternation_reward = -15.0
            
            self._update_step_timing()
                
        elif left_touchdown and self.last_swing_foot is None:
            self.last_swing_foot = 'left'
        elif right_touchdown and self.last_swing_foot is None:
            self.last_swing_foot = 'right'
        
        # Step frequency reward
        if len(self.step_time_history) >= 3:
            avg_step_interval = np.mean(self.step_time_history[-5:])
            actual_frequency = 1.0 / max(avg_step_interval, 0.1)
            frequency_error = abs(actual_frequency - self.step_frequency_target)
            step_frequency_reward = self.step_frequency_weight * np.exp(-2.0 * frequency_error)
        
        gait_reward += alternation_reward + step_frequency_reward + stride_length_reward
        info['gait_reward/alternation_reward'] = alternation_reward
        info['gait_reward/step_frequency_reward'] = step_frequency_reward
        info['gait_reward/stride_length_reward'] = stride_length_reward

        # Static standing penalty
        if self.data.time - self.last_steps_time > 0.5:
            steps_delta = self.steps_taken - self.last_steps_check
            if steps_delta == 0:
                gait_reward -= 10.0
                info['gait_reward/static_standing_penalty'] = -10.0
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

        # Contact pattern rewards
        contact_pattern_reward = 0.0
        if no_feet_contact:
            self.airborne_duration += 1
            if self.airborne_duration > 3:
                contact_pattern_reward = -self.feet_air_time_penalty * (self.airborne_duration - 3)
            else:
                contact_pattern_reward = -2.0
        elif single_support:
            self.airborne_duration = 0
            self.double_support_duration = 0
            if len(self.velocity_history) > 10:
                avg_velocity = np.mean(self.velocity_history[-10:])
                self.single_support_counter += 1
                if self.single_support_counter > 50:
                    self.last_single_support_steps = self.steps_taken
                    self.single_support_counter = 0
                steps_since_last = self.steps_taken - self.last_single_support_steps
                if avg_velocity > 0.1 and steps_since_last > 0:
                    contact_pattern_reward = self.single_support_reward_weight
                    if avg_velocity > 0.3:
                        contact_pattern_reward += 2.0
                else:
                    contact_pattern_reward = -5.0
        elif both_feet_contact:
            self.airborne_duration = 0
            self.double_support_duration += 1
            if self.double_support_duration > 5:
                excess_duration = min(self.double_support_duration - 5, 5)
                contact_pattern_reward = -self.double_support_penalty_weight * excess_duration
            else:
                contact_pattern_reward = -0.5
        else:
            self.double_support_duration = 0
        
        gait_reward += self.contact_pattern_weight * contact_pattern_reward
        info['gait_reward/contact_pattern_rew'] = self.contact_pattern_weight * contact_pattern_reward

        # Stance width
        left_foot_y = self.data.site_xpos[self.left_foot_site_id][1]
        right_foot_y = self.data.site_xpos[self.right_foot_site_id][1]
        feet_lateral_distance = abs(left_foot_y - right_foot_y)
        
        if feet_lateral_distance > 0.3:
            gait_reward -= 2.0
            info['gait_reward/wide_stance_penalty'] = -2.0
        elif feet_lateral_distance < 0.1:
            gait_reward -= 1.0
            info['gait_reward/narrow_stance_penalty'] = -1.0
        else:
            info['gait_reward/wide_stance_penalty'] = 0.0
            info['gait_reward/narrow_stance_penalty'] = 0.0

        # Swing clearance
        left_foot_z = self.data.site_xpos[self.left_foot_site_id][2]
        right_foot_z = self.data.site_xpos[self.right_foot_site_id][2]
        info['env_metrics/left_foot_height'] = left_foot_z
        info['env_metrics/right_foot_height'] = right_foot_z

        optimal_clearance = 0.10
        clearance_reward = 0.0
        
        if left_contact and not right_contact:
            if right_foot_z >= self.min_clearance_height:
                clearance_deviation = abs(right_foot_z - optimal_clearance)
                clearance_reward = 2.0 - (clearance_deviation * 10) if clearance_deviation < 0.04 else 0.5
            else:
                clearance_reward = -1.0
        elif right_contact and not left_contact:
            if left_foot_z >= self.min_clearance_height:
                clearance_deviation = abs(left_foot_z - optimal_clearance)
                clearance_reward = 2.0 - (clearance_deviation * 10) if clearance_deviation < 0.04 else 0.5
            else:
                clearance_reward = -1.0
        
        gait_reward += self.swing_clearance_weight * clearance_reward
        info['gait_reward/clearance_rew'] = self.swing_clearance_weight * clearance_reward

        # CoM smoothness
        com_vel = self.data.cvel[self.pelvis_id, :3]
        com_vel_penalty = np.square(com_vel[0]) + np.square(com_vel[2])
        gait_reward -= self.com_smoothness_weight * com_vel_penalty
        info['gait_reward/com_smoothness_pen'] = -self.com_smoothness_weight * com_vel_penalty

        # Orientation
        torso_orientation_quat = self.data.xquat[self.torso_id]
        orientation_penalty = np.sum(np.square(torso_orientation_quat[1:]))
        gait_reward -= self.orientation_weight * orientation_penalty
        info['gait_reward/orientation_pen'] = -self.orientation_weight * orientation_penalty
        
        # Torso rotation
        torso_angular_vel = self.data.cvel[self.torso_id, 3:]
        torso_rotation = np.sum(np.square(torso_angular_vel))
        torso_rotation_penalty = -self.torso_rotation_penalty_weight * torso_rotation
        gait_reward += torso_rotation_penalty
        info['gait_reward/torso_rotation_pen'] = torso_rotation_penalty
        
        # Foot slide
        foot_slide_penalty = 0.0
        if left_contact:
            try:
                left_foot_vel = self.data.cvel[self.model.body('foot_left').id, :2]
                left_slide = np.sum(np.square(left_foot_vel))
                if left_slide > 0.01:
                    foot_slide_penalty -= left_slide
            except KeyError:
                pass
                
        if right_contact:
            try:
                right_foot_vel = self.data.cvel[self.model.body('foot_right').id, :2]
                right_slide = np.sum(np.square(right_foot_vel))
                if right_slide > 0.01:
                    foot_slide_penalty -= right_slide
            except KeyError:
                pass
        
        foot_slide_penalty *= self.foot_slide_penalty_weight
        gait_reward += foot_slide_penalty
        info['gait_reward/foot_slide_pen'] = foot_slide_penalty

        # Update tracking
        self.last_left_contact = left_contact
        self.last_right_contact = right_contact

        return gait_reward, info

    def _update_step_timing(self):
        """Update step timing history."""
        current_time = self.data.time
        if self.last_step_time > 0:
            step_interval = current_time - self.last_step_time
            self.step_time_history.append(step_interval)
            if len(self.step_time_history) > 20:
                self.step_time_history.pop(0)
        self.last_step_time = current_time

    def step(self, action):
        """Execute one timestep with proper metric handling for ALL phases."""
        
        # Store position before step
        xy_position_before = self.data.qpos[:2].copy()
        
        # Simulate physics
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.data.qpos[:2].copy()

        # Velocity calculation
        x_velocity = (xy_position_after[0] - xy_position_before[0]) / self.dt
        forward_velocity = x_velocity
        y_velocity = (xy_position_after[1] - xy_position_before[1]) / self.dt
        
        # Lateral penalty
        lateral_penalty = -self.lateral_penalty_weight * abs(y_velocity)
        total_lateral_drift = abs(self.data.qpos[1] - self.episode_start_y)
        if total_lateral_drift > 0.5:
            lateral_penalty -= 10.0
        
        # Track velocity
        self.velocity_history.append(forward_velocity)
        if len(self.velocity_history) > self.velocity_history_maxlen:
            self.velocity_history.pop(0)

        # Path deviation
        current_y = self.data.qpos[1]
        self.path_deviation_history.append(current_y)
        if len(self.path_deviation_history) > 50:
            self.path_deviation_history.pop(0)

        # Straight path bonus
        straight_path_bonus = 0.0
        if len(self.path_deviation_history) > 20:
            path_variance = np.var(self.path_deviation_history[-20:])
            if path_variance < 0.01:
                straight_path_bonus = 5.0
            elif path_variance < 0.05:
                straight_path_bonus = 2.0

        # Pelvis stability
        pelvis_lateral_vel = self.data.cvel[self.pelvis_id, 1]
        hip_stability_penalty = -2.0 * np.square(pelvis_lateral_vel)

        # Torso orientation
        torso_quat = self.data.xquat[self.torso_id]
        torso_roll = 2.0 * np.arcsin(np.clip(torso_quat[1], -1.0, 1.0))
        lateral_tilt_penalty = -3.0 * np.square(torso_roll)
        torso_pitch = 2.0 * np.arcsin(np.clip(torso_quat[2], -1.0, 1.0))
        pitch_penalty = -5.0 * np.square(torso_pitch)

        # Arm penalties
        arm_joint_velocities = self.data.qvel[21:27]
        arm_flailing_penalty = -self.arm_penalty_weight * np.sum(np.square(arm_joint_velocities))
        arm_joint_positions = self.data.qpos[22:28]
        arm_position_penalty = -0.05 * np.sum(np.square(arm_joint_positions))

        # Upright bonus
        upright_bonus = 8.0 * torso_quat[0]
        upright_penalty = -5.0 * (0.95 - torso_quat[0]) ** 2 if torso_quat[0] < 0.95 else 0.0

        # Torso stability
        torso_angular_vel = self.data.cvel[self.torso_id, 3:]
        torso_stability_penalty = -self.torso_stability_weight * np.sum(np.square(torso_angular_vel))

        # Head stability
        try:
            head_id = self.model.body('head').id
            head_angular_vel = self.data.cvel[head_id, 3:]
            head_stability_penalty = -self.head_stability_weight * np.sum(np.square(head_angular_vel))
        except:
            head_stability_penalty = 0.0
        
        # Costs
        ctrl_cost = self.ctrl_cost_weight * np.sum(np.square(action))
        contact_cost = self.contact_cost_weight * np.sum(np.square(self.contact_forces))

        # Calculate reward components
        gait_reward, gait_info = self._calculate_gait_rewards()
        joint_constraint_penalty, joint_constraint_info = self._calculate_joint_constraint_penalties()
        
        left_contact = self._check_foot_contact(self.left_foot_geoms)
        right_contact = self._check_foot_contact(self.right_foot_geoms)
        arm_swing_reward, arm_swing_info = self._calculate_arm_swing_rewards(left_contact, right_contact)

        # Termination
        terminated = not self.is_healthy
        healthy_reward = self.healthy_reward if not terminated else 0.0

        # Standing rewards
        height_z = self.data.qpos[2]
        height_deviation = abs(height_z - self.target_height)
        smooth_balance_reward = 10.0 * np.exp(-2.0 * height_deviation)
        torso_quat_w = self.data.xquat[self.torso_id][0]
        orientation_deviation = max(0, 0.9 - torso_quat_w)
        smooth_upright_reward = 5.0 * np.exp(-3.0 * orientation_deviation)
        balance_reward = smooth_balance_reward + smooth_upright_reward
        if not self.is_healthy:
            balance_reward = -20.0
        
        height_reward = 3.0 * np.exp(-height_deviation)
        velocity_penalty = -0.03 * np.sum(np.square(self.data.qvel[6:]))
        
        # Initialize ALL metrics
        all_metrics = {
            'base_reward/healthy': healthy_reward,
            'base_reward/ctrl_cost': -ctrl_cost,
            'base_reward/contact_cost': -contact_cost,
            'base_reward/gait_total': self.gait_reward_weight * gait_reward,
            'base_reward/total_reward': 0.0,
            
            'env_metrics/forward_velocity': forward_velocity,
            'env_metrics/x_position': xy_position_after[0],
            'env_metrics/y_position': xy_position_after[1],
            'env_metrics/z_position': height_z,
            'env_metrics/steps_taken': self.steps_taken,
            
            'curriculum/walking_progress': self.walking_progress,
            'curriculum/alpha_standing': 0.0,
            'curriculum/alpha_walking': 0.0,
            'curriculum/standing_rew': 0.0,
            'curriculum/walking_rew': 0.0,
            'curriculum/progressive_forward_weight': 0.0,
            'curriculum/gait_penalty_scale': 0.0,
            'curriculum/ultra_simple_mode': 0.0,
            
            'standing_phase/balance_reward': 0.0,
            'standing_phase/height_reward': 0.0,
            'standing_phase/velocity_penalty': 0.0,
            'standing_phase/torso_upright': 0.0,
            
            'walking_phase/enhanced_forward_reward': 0.0,
            'walking_phase/scaled_gait_reward': 0.0,
            'walking_phase/velocity_tracking': 0.0,
            'walking_phase/sustained_speed_bonus': 0.0,
            
            'ultra_simple/balance_reward': 0.0,
            'ultra_simple/upright_reward': 0.0,
            'ultra_simple/neutral_pose_penalty': 0.0,

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
            'joint_constraints/progress_scale': 0.0,
        }
        
        # Merge calculated metrics
        all_metrics.update(gait_info)
        all_metrics.update(joint_constraint_info)
        all_metrics.update(arm_swing_info)

        # Phase-specific reward calculation
        if self.training_phase == "standing" and self.walking_progress < 0.3:
            # Ultra-simple mode
            simple_balance = 10.0 if self.is_healthy else -50.0
            simple_upright = 5.0 if torso_quat_w > 0.85 else -5.0
            
            joint_positions = self.data.qpos[7:]
            neutral_pose_penalty = -0.1 * np.sum(np.square(joint_positions))
            
            total_reward = simple_balance + simple_upright + neutral_pose_penalty - ctrl_cost
            
            all_metrics.update({
                'ultra_simple/balance_reward': simple_balance,
                'ultra_simple/upright_reward': simple_upright,
                'ultra_simple/neutral_pose_penalty': neutral_pose_penalty,
                'curriculum/ultra_simple_mode': 1.0,
                'curriculum/alpha_standing': 1.0,
                'curriculum/alpha_walking': 0.0,
                'curriculum/standing_rew': total_reward,
                'base_reward/total_reward': total_reward,
                'standing_phase/balance_reward': balance_reward,
                'standing_phase/height_reward': height_reward,
                'standing_phase/velocity_penalty': velocity_penalty,
                'standing_phase/torso_upright': smooth_upright_reward,
            })
            
        else:
            # Walking mode with curriculum blending
            standing_reward = balance_reward + height_reward + velocity_penalty + smooth_upright_reward - ctrl_cost
            
            # Progressive forward reward (V17's key feature - curriculum scaled!)
            progressive_forward_weight = self.walking_progress * self.forward_reward_weight
            enhanced_forward_reward = progressive_forward_weight * forward_velocity
            
            # Velocity tracking
            velocity_reward = 0.0
            speed_penalty = 0.0
            MIN_SPEED = 0.1
            
            if len(self.velocity_history) > 10:
                avg_velocity = np.mean(self.velocity_history[-10:])
                velocity_error = abs(avg_velocity - self.target_velocity)
                
                if forward_velocity < MIN_SPEED:
                    speed_penalty = -5.0 * (MIN_SPEED - forward_velocity)
                
                velocity_reward = 2.0 * np.exp(-3.0 * velocity_error) + speed_penalty
            
            # Sustained speed bonus
            sustained_bonus = 0.0
            if len(self.velocity_history) >= 20:
                if all(v > 0.2 for v in self.velocity_history[-20:]):
                    sustained_bonus = 10.0
            
            # Scale gait penalties
            gait_penalty_scale = max(0.3, self.walking_progress)
            scaled_gait_reward = gait_reward * gait_penalty_scale

            # Step rate bonus
            step_rate_bonus = 0.0
            if len(self.velocity_history) > 100:
                time_elapsed = self.data.time
                if time_elapsed > 1.0:
                    steps_per_second = self.steps_taken / time_elapsed
                    if steps_per_second > 0.8:
                        step_rate_bonus = 5.0
                    elif steps_per_second > 0.3:
                        step_rate_bonus = 0.0
                    else:
                        step_rate_bonus = -10.0
            
            # FIX: Use enhanced_forward_reward (was undefined forward_reward)
            walking_reward = (
                enhanced_forward_reward
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
            
            # Joint regularization
            joint_positions = self.data.qpos[7:]
            neutral_pose_penalty = -0.05 * np.sum(np.square(joint_positions))
            standing_reward += neutral_pose_penalty
            walking_reward += neutral_pose_penalty
            
            # Movement penalty
            distance_covered = self.data.qpos[0] - self.episode_start_x
            insufficient_movement_penalty = 0.0
            if len(self.step_time_history) > 20 and distance_covered < 0.1:
                insufficient_movement_penalty = -20.0

            # Blend rewards
            alpha_standing = 1.0 - self.walking_progress
            alpha_walking = self.walking_progress
            total_reward = alpha_standing * standing_reward + alpha_walking * walking_reward
            total_reward += insufficient_movement_penalty
            
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

        observation = self._get_obs()
        truncated = False
        
        # FIX: Return all_metrics, not info!
        return observation, total_reward, terminated, truncated, all_metrics