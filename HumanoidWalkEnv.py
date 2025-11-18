"""
Custom Humanoid Walking Environment for MuJoCo - FIXED VERSION
Implements curriculum learning from standing to natural bipedal walking
Critical fixes applied for coordinate consistency and gait rewards
"""

from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.utils import EzPickle
import numpy as np
import os

DEFAULT_XML_PATH = os.path.join(os.path.dirname(__file__), 'assets', 'humanoid_180_75.xml')


class HumanoidWalkEnv(MujocoEnv, EzPickle):
    """
    Humanoid walking environment with curriculum learning support.
    
    Training progresses through three phases:
    1. Ultra-simple balance (progress < 0.3): Basic standing with minimal rewards
    2. Standing phase (progress 0.3-1.0 in 'standing' mode): Balance with smooth transitions
    3. Walking phase ('walking' mode): Full locomotion with gait rewards
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

        # Get body and sensor IDs from model
        try:
            self.torso_id = self.model.body('torax').id
            self.pelvis_id = self.model.body('pelvis').id
            self.left_foot_site_id = self.model.site('foot_left_site').id
            self.right_foot_site_id = self.model.site('foot_right_site').id
            self.left_touch_sensor_id = self.model.sensor('foot_left_touch').id
            self.right_touch_sensor_id = self.model.sensor('foot_right_touch').id
            
            # Pre-calculate sensor addresses for efficiency
            self.left_touch_sensor_adr = self.model.sensor_adr[self.left_touch_sensor_id]
            self.right_touch_sensor_adr = self.model.sensor_adr[self.right_touch_sensor_id]
            
            # Store foot geom names for contact detection
            # Try different possible naming conventions from your XML
            self.left_foot_geoms = ['foot_left', 'foot1_left', 'foot2_left', 'lfoot', 'left_foot']
            self.right_foot_geoms = ['foot_right', 'foot1_right', 'foot2_right', 'rfoot', 'right_foot']
            
        except KeyError as e:
            raise KeyError(
                f"Required body/site/sensor not found in XML: {e}. "
                "Verify XML contains: bodies 'torax' and 'pelvis', "
                "sites 'foot_left_site' and 'foot_right_site', "
                "sensors 'foot_left_touch' and 'foot_right_touch'."
            ) from e

        # Update observation space with actual size
        obs_sample = self._get_obs()
        actual_obs_size = obs_sample.shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(actual_obs_size,), dtype=np.float32
        )

        # Initialize training phase
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
        
        # NEW: Gait tracking variables
        self.last_left_contact = False
        self.last_right_contact = False
        self.last_hip_x = self.data.qpos[0]  # Initialize hip X position
        self.last_swing_foot = None
        self.steps_taken = 0
        self.velocity_history = []
        self.target_velocity = 0.5  # m/s
        self.episode_start_x = 0.0

        # Lateral movement control
        self.lateral_penalty_weight = 5.0
        self.arm_penalty_weight = 0.1

        # Path tracking
        self.episode_start_y = 0.0
        self.path_deviation_history = []

        # Posture weights
        self.torso_stability_weight = 0.5
        self.head_stability_weight = 0.3

    def set_training_phase(self, phase: str, progress: float = 0.0):
        """
        Dynamically change training phase during training.
        
        Args:
            phase: Either 'standing' or 'walking'
            progress: Curriculum progress from 0.0 to 1.0
        """
        self.training_phase = phase
        self.walking_progress = min(1.0, max(0.0, progress))

    @property
    def healthy_z_range(self):
        return 1.0, 2.0

    @property
    def is_healthy(self):
        # Check height is within healthy range
        z = self.data.qpos[2]
        min_z, max_z = self.healthy_z_range
        healthy_height = min_z < z < max_z
        
        # Check torso is relatively upright (quaternion w-component)
        torso_upright = self.data.xquat[self.torso_id][0] > 0.7
        
        return healthy_height and torso_upright

    @property
    def contact_forces(self):
        """Contact forces from floor"""
        raw_contact_forces = self.data.cfrc_ext
        contact_forces = np.clip(raw_contact_forces, -1, 1)
        return contact_forces

    def _get_obs(self):
        """Get observation vector"""
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()
        
        com_inertia = self.data.xipos[self.torso_id].copy()
        com_velocity = self.data.cvel[self.torso_id].copy()
        
        actuator_forces = self.data.qfrc_actuator.flat.copy()
        contact_forces = self.contact_forces.flat.copy()
        
        return np.concatenate((
            position,
            velocity,
            com_inertia,
            com_velocity.flat,
            actuator_forces,
            contact_forces
        ))

    def reset_model(self):
        """Reset to initial standing position"""
        noise_low = -0.01
        noise_high = 0.01

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )
        
        # Ensure standing height
        qpos[2] = 1.4
        
        self.set_state(qpos, qvel)
        
        # Reset gait tracking
        self.last_left_contact = False
        self.last_right_contact = False
        self.steps_taken = 0
        self.last_swing_foot = None
        self.velocity_history = []

        self.episode_start_y = self.data.qpos[1]
        self.episode_start_x = self.data.qpos[0]
        
        return self._get_obs()

    def _check_foot_contact_mujoco(self, foot_geom_names):
        """Check if any of the foot geoms are in contact with ground"""
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_name = self.model.geom(contact.geom1).name
            geom2_name = self.model.geom(contact.geom2).name
            
            if 'floor' in [geom1_name, geom2_name]:
                for foot_geom in foot_geom_names:
                    if foot_geom in [geom1_name, geom2_name]:
                        return True
        return False

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

        # Hip progression
        current_hip_x = self.data.qpos[0]  # Root X position
        hip_progression = current_hip_x - self.last_hip_x
        self.last_hip_x = current_hip_x
        
        if left_touchdown and self.last_swing_foot == 'right':
            self.steps_taken += 1
            self.last_swing_foot = 'left'

            if hip_progression > 0.001:
                alternation_reward = 3.0
            else:
                alternation_reward = -5.0
            
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
                alternation_reward = 3.0
            else:
                alternation_reward = -5.0
            
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
        
        gait_reward += alternation_reward + step_frequency_reward
        info['gait_reward/alternation_reward'] = alternation_reward
        info['gait_reward/step_frequency_reward'] = step_frequency_reward

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
            contact_pattern_reward = -self.feet_air_time_penalty * 2.5
        elif single_support:
            # Strong reward for proper single support phase
            if len(self.velocity_history) > 10:
                avg_velocity = np.mean(self.velocity_history[-10:])
                if avg_velocity > 0.05:
                    contact_pattern_reward = 2.0  # Increased from 1.5
                else:
                    contact_pattern_reward = -1.0  # Penalty for slow movement
                
        elif both_feet_contact:
            # Small penalty for double support (should be brief)
            contact_pattern_reward = -0.3  # Slightly increased from -0.5
        
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
        achieved_clearance = 0.0

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
        info['gait_reward/clearance_achieved'] = achieved_clearance

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
        
        # 1. Arm movement penalty - discourage using arms for momentum
        # Get shoulder joint velocities (assuming joints 0-6 are base, 7+ are body joints)
        if self.data.qvel.shape[0] > 10:  # Make sure we have arm joints
            arm_velocities = self.data.qvel[21:27]  # First 6 body joints are typically arms
            arm_movement = np.sum(np.square(arm_velocities))
            arm_penalty = -self.arm_movement_penalty_weight * arm_movement
            gait_reward += arm_penalty
            info['gait_reward/arm_penalty'] = arm_penalty
        else:
            info['gait_reward/arm_penalty'] = 0.0
        
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
        }
        
        # Merge with gait_info (this may override some values)
        all_metrics.update(gait_info)

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
            
            walking_reward = (
                enhanced_forward_reward
                + healthy_reward
                - ctrl_cost
                - contact_cost
                + self.gait_reward_weight * scaled_gait_reward
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
            )
            
            # Joint position regularization
            joint_positions = self.data.qpos[7:]
            neutral_pose_penalty = -0.05 * np.sum(np.square(joint_positions))
            standing_reward += neutral_pose_penalty
            walking_reward += neutral_pose_penalty
            
            # Blend rewards based on curriculum progress
            alpha_standing = 1.0 - self.walking_progress
            alpha_walking = self.walking_progress
            total_reward = alpha_standing * standing_reward + alpha_walking * walking_reward

            distance_covered = self.data.qpos[0] - self.episode_start_x
            if len(self.step_time_history) > 20 and distance_covered < 0.1:
                total_reward = -20.0  # Penalty for not moving sufficiently
            
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
        reward = total_reward
        truncated = False
        info = all_metrics

        return observation, reward, terminated, truncated, info