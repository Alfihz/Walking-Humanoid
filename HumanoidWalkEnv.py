"""
HumanoidWalkEnv Gen2-09 - HIP Y ANTI-PHASE PENALTY
====================================================
Based on Gen2-08 with one targeted fix:

FIX (Gen2-09):
- Removed GRF symmetry penalty (Gen2-08) — did not affect behaviour.
- Added hip_y anti-phase penalty:
    * During normal walking, hip_y_right + hip_y_left ≈ 0 (one forward,
      one back). When both legs swing in the same direction (passive/sync),
      their sum deviates from zero.
    * Formula: -weight * (hip_y_right + hip_y_left)^2, capped at -15
    * Weight: 5.0
    * Gated on forward_velocity > 0.1
    * hip_y convention: negative = leg forward, positive = leg backward
    * New metrics: hip_y_antiphase_pen, hip_y_right, hip_y_left

FIX (Gen2-07 — preserved):
- Push-off reward: weight 6.0, capped at 2.0 rad/s, single-support gated

FIX (Gen2-06 — preserved):
- Hip Z deadband constraint: weight 4.0, deadband ±0.10 rad (±6°)

FIX (Gen2-05 — preserved):
- Positional lag penalty: LAG_TOLERANCE=45, LAG_PENALTY=-8.0, cap -20

FIX (Gen2-03 — preserved):
- Ankle X weight 3.0, deadband ±0.10 rad (±6°)

FIX (Gen2-02 — preserved):
- Ankle X and Hip Z applied AFTER curriculum scaling (always full strength)

Gen2-01 DESIGN PRINCIPLES (preserved):
- Minimal reward components (4-5 core only)
- Positive rewards drive behaviour; penalties are secondary and always CAPPED
- All left/right rewards are perfectly SYMMETRIC
- Forward progress (+X axis) is the PRIMARY mandatory goal
- Joint constraints are soft guidance, not hard punishment
- Curriculum: ultra-simple balance → standing → walking

BAKED-IN LESSONS FROM V17-V26:
- Forward = +X axis
- Replay buffer must be capped at 1M (prevents 3M slowdown)
- Asymmetric L/R rewards → persistent lean: keep BOTH identical
- Unbounded penalties → training collapse: ALL penalties capped
- contact_pattern penalty unbounded caused avg -131: hard cap at -15
- Balance reward floor must not be too negative (was -20, now -5)
- camera_name="track" must exist in XML (not just "back"/"side")

METRICS: 82 metrics (79 from Gen2-07 + 3 new: hip_y_antiphase_pen,
         hip_y_right, hip_y_left)
"""

from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.utils import EzPickle
import numpy as np
import os

DEFAULT_XML_PATH = os.path.join(os.path.dirname(__file__), 'assets', 'humanoid_180_75.xml')


class HumanoidWalkEnv(MujocoEnv, EzPickle):
    """
    Humanoid walking environment — V27 clean foundation.

    Curriculum phases:
      1. ultra-simple  (training_phase='standing', walking_progress < 0.3)
         Balance-only: stay upright, no movement required
      2. standing       (training_phase='standing', walking_progress 0.3-1.0)
         Smooth blend from standing to walking rewards
      3. walking        (training_phase='walking')
         Full locomotion with velocity targeting and gait rewards
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 100,
    }

    # ------------------------------------------------------------------ #
    #  INIT                                                                #
    # ------------------------------------------------------------------ #
    def __init__(self, xml_file=DEFAULT_XML_PATH, frame_skip=5,
                 training_phase="standing", **kwargs):

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
            ),
            **kwargs,
        )

        # --- Body / sensor IDs -------------------------------------------
        try:
            self.torso_id              = self.model.body('torax').id
            self.pelvis_id             = self.model.body('pelvis').id
            self.left_foot_site_id     = self.model.site('foot_left_site').id
            self.right_foot_site_id    = self.model.site('foot_right_site').id
            self.left_touch_sensor_id  = self.model.sensor('foot_left_touch').id
            self.right_touch_sensor_id = self.model.sensor('foot_right_touch').id
            self.left_touch_sensor_adr  = self.model.sensor_adr[self.left_touch_sensor_id]
            self.right_touch_sensor_adr = self.model.sensor_adr[self.right_touch_sensor_id]
            self.left_foot_geoms  = ['foot_left',  'foot1_left',  'foot2_left',  'lfoot',  'left_foot']
            self.right_foot_geoms = ['foot_right', 'foot1_right', 'foot2_right', 'rfoot', 'right_foot']
        except KeyError as e:
            raise KeyError(
                f"Required body/site/sensor not found in XML: {e}. "
                "Verify XML contains: bodies 'torax' and 'pelvis', "
                "sites 'foot_left_site' and 'foot_right_site', "
                "sensors 'foot_left_touch' and 'foot_right_touch'."
            ) from e

        # --- Observation space (auto-sized) --------------------------------
        obs_sample = self._get_obs()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_sample.shape[0],), dtype=np.float32
        )

        # --- Curriculum state ---------------------------------------------
        self.training_phase   = training_phase
        self.walking_progress = 0.0     # 0.0 = pure standing, 1.0 = pure walking
        print(f"Initialized HumanoidWalkEnv Gen2-09 in '{training_phase}' phase")

        EzPickle.__init__(self, xml_file=xml_file, frame_skip=frame_skip,
                          training_phase=training_phase, **kwargs)

        # ================================================================ #
        #  REWARD WEIGHTS — edit here to tune                              #
        # ================================================================ #

        # --- Core (primary) rewards --------------------------------------
        self.healthy_reward          = 2.0    # Alive bonus per step
        self.ctrl_cost_weight        = 0.01   # Action magnitude penalty
        self.contact_cost_weight     = 5e-7   # Contact force penalty

        # Velocity reward (Linear-push + Gaussian-pull hybrid)
        self.target_velocity         = 0.5    # m/s
        self.velocity_sigma          = 0.15   # Gaussian width around target
        self.linear_velocity_weight  = 1.5    # Pushes robot to move forward
        self.gaussian_velocity_weight= 8.0    # Pulls robot to exactly 0.5 m/s
        self.forward_reward_weight   = 10.0   # Scales enhanced_forward_reward

        # Upright reward
        self.upright_weight          = 8.0    # Torso quaternion w-component reward

        # Step alternation — MUST be identical L and R (lesson from V19)
        self.step_reward             = 12.0   # Reward for a good alternating step
        self.step_penalty            = -10.0  # Penalty for a backward/tiny step
        self.step_completion_bonus   = 5.0    # Bonus for completing a stride with hip progression

        # --- Secondary (gait quality) ------------------------------------
        self.gait_reward_weight          = 0.3
        self.contact_pattern_weight      = 2.0
        self.swing_clearance_weight      = 2.5
        self.com_smoothness_weight       = 0.6
        self.feet_air_time_penalty       = 5.0
        self.orientation_weight          = 0.2
        self.stride_length_reward_weight = 2.5
        self.single_support_reward_weight= 1.0
        self.double_support_penalty_weight=1.0
        self.step_frequency_target       = 1.8
        self.step_frequency_weight       = 1.5

        # --- Posture / stability -----------------------------------------
        self.lateral_penalty_weight      = 5.0
        self.torso_stability_weight      = 0.5
        self.head_stability_weight       = 0.3
        self.torso_rotation_penalty_weight=0.3
        self.foot_slide_penalty_weight   = 1.5
        self.arm_penalty_weight          = 0.1

        # --- Joint constraints -------------------------------------------
        self.joint_constraint_weight     = 2.0
        self.shoulder1_constraint_weight = 1.0
        self.shoulder2_constraint_weight = 1.5
        self.elbow_constraint_weight     = 0.8
        self.ankle_y_constraint_weight   = 1.2
        # Gen2-03: weight lowered 6.0→3.0, deadband widened ±3°→±6°
        self.ankle_x_constraint_weight   = 3.0
        # Deadband: feet are allowed ±0.10 rad (≈±6°) of lateral tilt.
        self.ankle_x_deadband            = 0.10
        self.abdomen_x_constraint_weight = 2.0
        self.abdomen_y_constraint_weight = 1.5
        self.abdomen_z_constraint_weight = 1.8
        # Gen2-06: hip_z controls foot direction (not ankle_x as previously assumed).
        # Deadband: ±0.10 rad (≈±6°) of natural rotation allowed during swing.
        # Applied AFTER curriculum scaling — always at full strength from step 1.
        self.hip_z_constraint_weight     = 4.0
        self.hip_z_deadband              = 0.10  # ±6°

        # --- Arm swing ---------------------------------------------------
        self.arm_swing_reward_weight        = 1.5
        self.arm_swing_coordination_weight  = 2.0

        # --- Forward requirement -----------------------------------------
        # Much lighter than V17's -100 — just a nudge, not a cliff
        self.forward_requirement_penalty    = -20.0  # Applied when avg_vel < 0.1

        # --- Positional lag penalty (Gen2-05) ----------------------------
        # If one foot stays behind the other for too long while moving,
        # apply a per-step penalty to force alternation.
        self.LAG_TOLERANCE  = 45    # steps before penalty kicks in
        self.LAG_PENALTY    = -8.0  # per step, capped at -20

        # --- Push-off reward (Gen2-07) -----------------------------------
        # Rewards ankle plantarflexion at toe-off during single support.
        self.push_off_weight = 6.0

        # --- Hip Y anti-phase penalty (Gen2-09) --------------------------
        # hip_y_right + hip_y_left should be near zero during walking
        # (one leg forward = other leg back). Penalises when both legs
        # swing in the same direction (passive/sync behaviour).
        # Formula: -weight * (hip_y_right + hip_y_left)^2, capped at -15.
        # Gated on forward_velocity > 0.1.
        self.hip_y_antiphase_weight = 5.0

        # --- Clearance ---------------------------------------------------
        self.min_clearance_height = 0.08  # 8 cm minimum swing clearance

        # ================================================================ #
        #  JOINT INDICES (qpos, first 7 = freejoint x,y,z,qw,qx,qy,qz)   #
        # ================================================================ #
        self.abdomen_z_idx       = 7
        self.abdomen_y_idx       = 8
        self.abdomen_x_idx       = 9
        self.hip_z_right_idx     = 11   # Gen2-06: hip rotation controls foot direction
        self.hip_y_right_idx     = 12   # Gen2-09: forward/backward leg swing
        self.hip_z_left_idx      = 17
        self.hip_y_left_idx      = 18   # Gen2-09: forward/backward leg swing
        self.ankle_y_right_idx   = 14
        self.ankle_x_right_idx   = 15
        self.ankle_y_left_idx    = 20
        self.ankle_x_left_idx    = 21
        self.shoulder1_right_idx = 24
        self.shoulder2_right_idx = 25
        self.elbow_right_idx     = 26
        self.shoulder1_left_idx  = 27
        self.shoulder2_left_idx  = 28
        self.elbow_left_idx      = 29

        # --- Gait tracking state -----------------------------------------
        self.last_left_contact  = False
        self.last_right_contact = False
        self.last_hip_x         = self.data.qpos[0]
        self.last_swing_foot    = None
        self.steps_taken        = 0
        self.velocity_history   = []
        self.step_time_history  = []
        self.episode_start_x    = 0.0
        self.episode_start_y    = 0.0
        self.path_deviation_history = []
        self.left_foot_lag_steps  = 0   # Gen2-05: positional lag counter
        self.right_foot_lag_steps = 0

    # ------------------------------------------------------------------ #
    #  CURRICULUM CONTROL                                                 #
    # ------------------------------------------------------------------ #
    def set_training_phase(self, phase: str, progress: float = 0.0):
        """Called by training script to advance curriculum."""
        self.training_phase   = phase
        self.walking_progress = float(np.clip(progress, 0.0, 1.0))

    # ------------------------------------------------------------------ #
    #  PROPERTIES                                                         #
    # ------------------------------------------------------------------ #
    @property
    def healthy_z_range(self):
        return (1.0, 2.0)

    @property
    def is_healthy(self):
        z            = self.data.qpos[2]
        min_z, max_z = self.healthy_z_range
        height_ok    = min_z < z < max_z
        upright_ok   = self.data.xquat[self.torso_id][0] > 0.7
        return height_ok and upright_ok

    @property
    def contact_forces(self):
        return np.clip(self.data.cfrc_ext, -1, 1)

    # ------------------------------------------------------------------ #
    #  OBSERVATION                                                        #
    # ------------------------------------------------------------------ #
    def _get_obs(self):
        return np.concatenate((
            self.data.qpos.flat.copy(),
            self.data.qvel.flat.copy(),
            self.data.xipos[self.torso_id].copy(),
            self.data.cvel[self.torso_id].flat.copy(),
            self.data.qfrc_actuator.flat.copy(),
            self.contact_forces.flat.copy(),
        ))

    # ------------------------------------------------------------------ #
    #  RESET                                                              #
    # ------------------------------------------------------------------ #
    def reset_model(self):
        noise = 0.01
        qpos  = self.init_qpos + self.np_random.uniform(-noise, noise, self.model.nq)
        qvel  = self.init_qvel + self.np_random.uniform(-noise, noise, self.model.nv)
        qpos[2] = 1.4   # Start standing
        self.set_state(qpos, qvel)

        # Reset all gait tracking
        self.last_left_contact      = False
        self.last_right_contact     = False
        self.last_hip_x             = self.data.qpos[0]
        self.last_swing_foot        = None
        self.steps_taken            = 0
        self.velocity_history       = []
        self.step_time_history      = []
        self.path_deviation_history = []
        self.episode_start_x        = self.data.qpos[0]
        self.episode_start_y        = self.data.qpos[1]
        self.left_foot_lag_steps    = 0   # Gen2-05: positional lag counters
        self.right_foot_lag_steps   = 0

        # Reset any timestep-local counters
        for attr in ('airborne_duration', 'double_support_duration',
                     'single_support_counter', 'last_single_support_steps',
                     'last_steps_check', 'last_steps_time', 'last_step_time'):
            if hasattr(self, attr):
                delattr(self, attr)

        return self._get_obs()

    # ------------------------------------------------------------------ #
    #  CONTACT HELPER                                                     #
    # ------------------------------------------------------------------ #
    def _check_foot_contact_mujoco(self, foot_geom_names):
        for i in range(self.data.ncon):
            c   = self.data.contact[i]
            g1  = self.model.geom(c.geom1).name
            g2  = self.model.geom(c.geom2).name
            if 'floor' in (g1, g2):
                for fg in foot_geom_names:
                    if fg in (g1, g2):
                        return True
        return False

    # ------------------------------------------------------------------ #
    #  JOINT CONSTRAINT PENALTIES  (secondary; fully capped)             #
    # ------------------------------------------------------------------ #
    def _calculate_joint_constraint_penalties(self):
        info  = {}
        total = 0.0
        qpos  = self.data.qpos

        # Abdomen
        abd_x = qpos[self.abdomen_x_idx]
        abd_y = qpos[self.abdomen_y_idx]
        abd_z = qpos[self.abdomen_z_idx]

        abd_y_pen = 0.0
        if abd_y < -0.1:
            abd_y_pen = -self.abdomen_y_constraint_weight * (abd_y + 0.1) ** 2
        elif abd_y > 0.2:
            abd_y_pen = -self.abdomen_y_constraint_weight * (abd_y - 0.2) ** 2
        abd_x_pen = -self.abdomen_x_constraint_weight * abd_x ** 2
        abd_z_pen = -self.abdomen_z_constraint_weight * abd_z ** 2
        abd_pen   = abd_x_pen + abd_y_pen + abd_z_pen
        total    += abd_pen

        info['joint_constraints/abdomen_x']       = float(abd_x)
        info['joint_constraints/abdomen_y']       = float(abd_y)
        info['joint_constraints/abdomen_z']       = float(abd_z)
        info['joint_constraints/abdomen_penalty'] = float(abd_pen)

        # Shoulder1 (arm swing — allow 0..1 range naturally)
        s1r = qpos[self.shoulder1_right_idx]
        s1l = qpos[self.shoulder1_left_idx]
        s1r_pen = 0.0 if 0.0 <= s1r <= 1.0 else -self.shoulder1_constraint_weight * (s1r if s1r < 0 else (s1r - 1.0)) ** 2
        s1l_pen = 0.0 if 0.0 <= s1l <= 1.0 else -self.shoulder1_constraint_weight * (s1l if s1l < 0 else (s1l - 1.0)) ** 2
        s1_pen  = s1r_pen + s1l_pen
        total  += s1_pen

        info['joint_constraints/shoulder1_right']   = float(s1r)
        info['joint_constraints/shoulder1_left']    = float(s1l)
        info['joint_constraints/shoulder1_penalty'] = float(s1_pen)

        # Shoulder2 (keep arms at sides: -1.5 to -1.25)
        s2r = qpos[self.shoulder2_right_idx]
        s2l = qpos[self.shoulder2_left_idx]
        s2r_pen = -self.shoulder2_constraint_weight * (s2r + 1.50) ** 2 if s2r < -1.50 else \
                  -self.shoulder2_constraint_weight * (s2r + 1.25) ** 2 if s2r > -1.25 else 0.0
        s2l_pen = -self.shoulder2_constraint_weight * (s2l + 1.50) ** 2 if s2l < -1.50 else \
                  -self.shoulder2_constraint_weight * (s2l + 1.25) ** 2 if s2l > -1.25 else 0.0
        s2_pen  = s2r_pen + s2l_pen
        total  += s2_pen

        info['joint_constraints/shoulder2_right']   = float(s2r)
        info['joint_constraints/shoulder2_left']    = float(s2l)
        info['joint_constraints/shoulder2_penalty'] = float(s2_pen)

        # Elbow (keep near 0)
        er = qpos[self.elbow_right_idx]
        el = qpos[self.elbow_left_idx]
        e_pen = -self.elbow_constraint_weight * (er ** 2 + el ** 2)
        total += e_pen

        info['joint_constraints/elbow_right']   = float(er)
        info['joint_constraints/elbow_left']    = float(el)
        info['joint_constraints/elbow_penalty'] = float(e_pen)

        # Ankle Y (limited dorsiflexion; allow -0.1..0)
        ayr = qpos[self.ankle_y_right_idx]
        ayl = qpos[self.ankle_y_left_idx]
        ayr_pen = -self.ankle_y_constraint_weight * (ayr + 0.1) ** 2 if ayr < -0.1 else \
                  -self.ankle_y_constraint_weight * ayr ** 2 if ayr > 0.0 else 0.0
        ayl_pen = -self.ankle_y_constraint_weight * (ayl + 0.1) ** 2 if ayl < -0.1 else \
                  -self.ankle_y_constraint_weight * ayl ** 2 if ayl > 0.0 else 0.0
        ay_pen  = ayr_pen + ayl_pen
        total  += ay_pen

        info['joint_constraints/ankle_y_right']   = float(ayr)
        info['joint_constraints/ankle_y_left']    = float(ayl)
        info['joint_constraints/ankle_y_penalty'] = float(ay_pen)

        # Scale by master weight and curriculum progress
        # NOTE: ankle_x is intentionally excluded from this scaling —
        # foot orientation must be enforced from step 1, not just in walking phase.
        scale  = 0.2 + 0.8 * min(1.0, self.walking_progress * 2)
        total *= self.joint_constraint_weight * scale

        # ── ANKLE X (lateral foot twist) ─────────────────────────────────
        # Applied AFTER curriculum scaling so it is always at full strength.
        # Deadband: ±ankle_x_deadband rad is free (natural small tilt allowed).
        # Outside deadband: strong quadratic penalty on the excess angle only.
        axr = qpos[self.ankle_x_right_idx]
        axl = qpos[self.ankle_x_left_idx]
        db  = self.ankle_x_deadband  # ±0.05 rad ≈ ±3°

        def _ax_pen(angle):
            excess = abs(angle) - db
            if excess > 0:
                return -self.ankle_x_constraint_weight * (excess ** 2)
            return 0.0

        axr_pen = _ax_pen(axr)
        axl_pen = _ax_pen(axl)
        ax_pen  = axr_pen + axl_pen
        total  += ax_pen   # added at full weight, no curriculum discount

        info['joint_constraints/ankle_x_right']   = float(axr)
        info['joint_constraints/ankle_x_left']    = float(axl)
        info['joint_constraints/ankle_x_penalty'] = float(ax_pen)

        # ── HIP Z (leg rotation = foot direction) ─────────────────────────
        # Gen2-06: hip_z controls which direction the foot points.
        # Applied AFTER curriculum scaling — always at full strength.
        # Deadband: ±hip_z_deadband rad of natural rotation is free.
        hzr = qpos[self.hip_z_right_idx]
        hzl = qpos[self.hip_z_left_idx]
        db_hz = self.hip_z_deadband

        def _hz_pen(angle):
            excess = abs(angle) - db_hz
            if excess > 0:
                return -self.hip_z_constraint_weight * (excess ** 2)
            return 0.0

        hzr_pen = _hz_pen(hzr)
        hzl_pen = _hz_pen(hzl)
        hz_pen  = hzr_pen + hzl_pen
        total  += hz_pen

        info['joint_constraints/hip_z_right']   = float(hzr)
        info['joint_constraints/hip_z_left']    = float(hzl)
        info['joint_constraints/hip_z_penalty'] = float(hz_pen)

        info['joint_constraints/total_penalty']  = float(total)
        info['joint_constraints/progress_scale'] = float(scale)

        return total, info

    # ------------------------------------------------------------------ #
    #  ARM SWING REWARDS                                                  #
    # ------------------------------------------------------------------ #
    def _calculate_arm_swing_rewards(self, left_contact, right_contact):
        info   = {}
        reward = 0.0

        s1r     = self.data.qpos[self.shoulder1_right_idx]
        s1l     = self.data.qpos[self.shoulder1_left_idx]
        s1r_vel = self.data.qvel[self.shoulder1_right_idx - 7]
        s1l_vel = self.data.qvel[self.shoulder1_left_idx  - 7]

        # Reward any arm movement
        arm_movement   = abs(s1r_vel) + abs(s1l_vel)
        movement_rew   = self.arm_swing_reward_weight * min(arm_movement, 2.0)
        reward        += movement_rew
        info['arm_swing/movement_reward'] = float(movement_rew)

        # Coordination: opposite arm swings with stance leg
        coord_rew = 0.0
        if right_contact and not left_contact:     # Right stance → left arm forward
            if s1l > 0.2:
                coord_rew += 1.5 * s1l
            if s1r < 0.0:
                coord_rew -= 1.0 * abs(s1r)
            elif s1r < 0.3:
                coord_rew += 0.3 * (0.3 - s1r)
        elif left_contact and not right_contact:   # Left stance → right arm forward
            if s1r > 0.2:
                coord_rew += 1.5 * s1r
            if s1l < 0.0:
                coord_rew -= 1.0 * abs(s1l)
            elif s1l < 0.3:
                coord_rew += 0.3 * (0.3 - s1l)

        coord_rew *= self.arm_swing_coordination_weight
        reward    += coord_rew

        info['arm_swing/coordination_reward'] = float(coord_rew)
        info['arm_swing/shoulder1_right']     = float(s1r)
        info['arm_swing/shoulder1_left']      = float(s1l)
        info['arm_swing/total_reward']        = float(reward)

        return reward, info

    # ------------------------------------------------------------------ #
    #  GAIT REWARDS                                                       #
    # ------------------------------------------------------------------ #
    def _calculate_gait_rewards(self, forward_velocity):
        gait_reward = 0.0
        info        = {}

        left_contact  = self._check_foot_contact_mujoco(self.left_foot_geoms)
        right_contact = self._check_foot_contact_mujoco(self.right_foot_geoms)

        info['env_metrics/left_contact']  = float(left_contact)
        info['env_metrics/right_contact'] = float(right_contact)

        left_touchdown  = left_contact  and not self.last_left_contact
        right_touchdown = right_contact and not self.last_right_contact

        # ---- Step alternation (PERFECTLY SYMMETRIC L = R) ----
        alternation_reward  = 0.0
        step_frequency_rew  = 0.0
        stride_length_rew   = 0.0

        current_hip_x  = self.data.qpos[0]
        hip_progression = current_hip_x - self.last_hip_x
        self.last_hip_x = current_hip_x

        def _record_step():
            """Track timing for frequency reward."""
            t = self.data.time
            if hasattr(self, 'last_step_time'):
                self.step_time_history.append(t - self.last_step_time)
                if len(self.step_time_history) > 20:
                    self.step_time_history.pop(0)
            self.last_step_time = t

        if left_touchdown and self.last_swing_foot == 'right':
            self.steps_taken  += 1
            self.last_swing_foot = 'left'
            if hip_progression > 0.001:
                alternation_reward = self.step_reward
                if hip_progression > 0.05:
                    stride_length_rew = self.stride_length_reward_weight * min(hip_progression * 20, 5.0)
                alternation_reward += self.step_completion_bonus
            else:
                alternation_reward = self.step_penalty    # Capped; identical to right
            _record_step()

        elif right_touchdown and self.last_swing_foot == 'left':
            self.steps_taken  += 1
            self.last_swing_foot = 'right'
            if hip_progression > 0.001:
                alternation_reward = self.step_reward     # IDENTICAL to left step
                if hip_progression > 0.05:
                    stride_length_rew = self.stride_length_reward_weight * min(hip_progression * 20, 5.0)
                alternation_reward += self.step_completion_bonus
            else:
                alternation_reward = self.step_penalty    # IDENTICAL to left step
            _record_step()

        elif left_touchdown  and self.last_swing_foot is None:
            self.last_swing_foot = 'left'
        elif right_touchdown and self.last_swing_foot is None:
            self.last_swing_foot = 'right'

        # Step frequency reward
        if len(self.step_time_history) >= 3:
            avg_interval    = np.mean(self.step_time_history[-5:])
            actual_freq     = 1.0 / max(avg_interval, 0.1)
            freq_error      = abs(actual_freq - self.step_frequency_target)
            step_frequency_rew = self.step_frequency_weight * np.exp(-2.0 * freq_error)

        gait_reward += alternation_reward + step_frequency_rew + stride_length_rew
        info['gait_reward/alternation_reward']    = float(alternation_reward)
        info['gait_reward/step_frequency_reward'] = float(step_frequency_rew)
        info['gait_reward/stride_length_reward']  = float(stride_length_rew)

        # Static standing detection (penalty if no steps in 0.5s)
        if not hasattr(self, 'last_steps_check'):
            self.last_steps_check = 0
            self.last_steps_time  = 0.0

        static_pen = 0.0
        if self.data.time - self.last_steps_time > 0.5:
            if self.steps_taken == self.last_steps_check:
                static_pen = -10.0
            self.last_steps_check = self.steps_taken
            self.last_steps_time  = self.data.time
        gait_reward += static_pen
        info['gait_reward/static_standing_penalty'] = float(static_pen)

        # Contact states
        no_contact     = not left_contact  and not right_contact
        both_contact   = left_contact  and right_contact
        single_support = left_contact  != right_contact   # XOR

        info['env_metrics/no_contact']     = float(no_contact)
        info['env_metrics/both_contact']   = float(both_contact)
        info['env_metrics/single_support'] = float(single_support)

        # ---- Contact pattern reward (CAPPED) ----
        contact_pattern_rew = 0.0

        if no_contact:
            if not hasattr(self, 'airborne_duration'):
                self.airborne_duration = 0
            self.airborne_duration += 1
            if self.airborne_duration > 3:
                raw = -self.feet_air_time_penalty * (self.airborne_duration - 3)
                contact_pattern_rew = max(raw, -15.0)   # HARD CAP
            else:
                contact_pattern_rew = -2.0

        elif single_support:
            if hasattr(self, 'airborne_duration'):
                self.airborne_duration = 0
            if len(self.velocity_history) > 10:
                avg_vel = np.mean(self.velocity_history[-10:])
                if avg_vel > 0.2:
                    contact_pattern_rew = self.single_support_reward_weight
                    if avg_vel > 0.4:
                        contact_pattern_rew += 2.0
                else:
                    contact_pattern_rew = -5.0
            else:
                contact_pattern_rew = 0.0

        elif both_contact:
            if hasattr(self, 'airborne_duration'):
                self.airborne_duration = 0
            if not hasattr(self, 'double_support_duration'):
                self.double_support_duration = 0
            self.double_support_duration += 1
            if self.double_support_duration > 5:
                excess = min(self.double_support_duration - 5, 5)
                raw    = -self.double_support_penalty_weight * excess
                contact_pattern_rew = max(raw, -10.0)   # HARD CAP
            else:
                contact_pattern_rew = -0.5
        else:
            if hasattr(self, 'double_support_duration'):
                self.double_support_duration = 0

        gait_reward += self.contact_pattern_weight * contact_pattern_rew
        info['gait_reward/contact_pattern_rew'] = float(self.contact_pattern_weight * contact_pattern_rew)

        # Foot positions
        lf_y = self.data.site_xpos[self.left_foot_site_id][1]
        rf_y = self.data.site_xpos[self.right_foot_site_id][1]
        lf_z = self.data.site_xpos[self.left_foot_site_id][2]
        rf_z = self.data.site_xpos[self.right_foot_site_id][2]

        info['env_metrics/left_foot_height']  = float(lf_z)
        info['env_metrics/right_foot_height'] = float(rf_z)

        feet_lateral = abs(lf_y - rf_y)
        if feet_lateral > 0.3:
            gait_reward -= 2.0
            info['gait_reward/wide_stance_penalty']   = -2.0
            info['gait_reward/narrow_stance_penalty'] = 0.0
        elif feet_lateral < 0.1:
            gait_reward -= 1.0
            info['gait_reward/wide_stance_penalty']   = 0.0
            info['gait_reward/narrow_stance_penalty'] = -1.0
        else:
            info['gait_reward/wide_stance_penalty']   = 0.0
            info['gait_reward/narrow_stance_penalty'] = 0.0

        # ---- Swing clearance (only when moving forward) ----
        clearance_rew = 0.0
        if forward_velocity > 0.1:
            opt = 0.10   # Optimal clearance 10 cm
            if left_contact and not right_contact:  # Right foot swinging
                try:
                    rfv = self.data.cvel[self.model.body('foot_right').id, 0]
                    fwd = rfv > -0.1
                except Exception:
                    fwd = True
                if fwd and rf_z >= self.min_clearance_height:
                    dev = abs(rf_z - opt)
                    clearance_rew = 2.0 - dev * 10 if dev < 0.04 else 0.5
                elif rf_z < self.min_clearance_height and fwd:
                    clearance_rew = -1.0

            elif right_contact and not left_contact:  # Left foot swinging
                try:
                    lfv = self.data.cvel[self.model.body('foot_left').id, 0]
                    fwd = lfv > -0.1
                except Exception:
                    fwd = True
                if fwd and lf_z >= self.min_clearance_height:
                    dev = abs(lf_z - opt)
                    clearance_rew = 2.0 - dev * 10 if dev < 0.04 else 0.5
                elif lf_z < self.min_clearance_height and fwd:
                    clearance_rew = -1.0

        gait_reward += self.swing_clearance_weight * clearance_rew
        info['gait_reward/clearance_rew'] = float(self.swing_clearance_weight * clearance_rew)

        # CoM smoothness
        com_vel     = self.data.cvel[self.pelvis_id, :3]
        com_pen     = float(com_vel[0] ** 2 + com_vel[2] ** 2)
        gait_reward -= self.com_smoothness_weight * com_pen
        info['gait_reward/com_smoothness_pen'] = float(-self.com_smoothness_weight * com_pen)

        # Orientation penalty
        tq          = self.data.xquat[self.torso_id]
        orient_pen  = float(np.sum(tq[1:] ** 2))
        gait_reward -= self.orientation_weight * orient_pen
        info['gait_reward/orientation_pen'] = float(-self.orientation_weight * orient_pen)

        # Torso rotation penalty
        tav         = self.data.cvel[self.torso_id, 3:]
        tor_rot_pen = float(-self.torso_rotation_penalty_weight * np.sum(tav ** 2))
        gait_reward += tor_rot_pen
        info['gait_reward/torso_rotation_pen'] = float(tor_rot_pen)

        # Foot slide penalty
        fslide = 0.0
        if left_contact:
            try:
                lfv = self.data.cvel[self.model.body('foot_left').id, :2]
                ls  = float(np.sum(lfv ** 2))
                if ls > 0.01:
                    fslide -= ls
            except Exception:
                pass
        if right_contact:
            try:
                rfv = self.data.cvel[self.model.body('foot_right').id, :2]
                rs  = float(np.sum(rfv ** 2))
                if rs > 0.01:
                    fslide -= rs
            except Exception:
                pass
        fslide *= self.foot_slide_penalty_weight
        gait_reward += fslide
        info['gait_reward/foot_slide_pen'] = float(fslide)

        # ---- Positional lag penalty (Gen2-05) ----
        # If one foot stays positionally behind the other while the robot
        # is moving forward, increment its lag counter. After LAG_TOLERANCE
        # steps the penalty fires. Resets when the lagging foot catches up
        # or the robot stops moving.
        lag_penalty = 0.0
        lf_x = self.data.site_xpos[self.left_foot_site_id][0]
        rf_x = self.data.site_xpos[self.right_foot_site_id][0]
        if forward_velocity > 0.1:
            if lf_x < rf_x:   # left foot is behind
                self.left_foot_lag_steps  += 1
                self.right_foot_lag_steps  = 0
            elif rf_x < lf_x:  # right foot is behind
                self.right_foot_lag_steps += 1
                self.left_foot_lag_steps   = 0
            else:
                self.left_foot_lag_steps  = 0
                self.right_foot_lag_steps = 0

            lag_steps = max(self.left_foot_lag_steps, self.right_foot_lag_steps)
            if lag_steps > self.LAG_TOLERANCE:
                lag_penalty = max(self.LAG_PENALTY * (lag_steps - self.LAG_TOLERANCE), -20.0)
        else:
            self.left_foot_lag_steps  = 0
            self.right_foot_lag_steps = 0

        gait_reward += lag_penalty
        info['gait_reward/positional_lag_penalty'] = float(lag_penalty)

        # ---- Push-off reward (Gen2-07) ----
        # Rewards ankle plantarflexion (ankle_y going negative = toes pushing down)
        # at toe-off. Guard conditions prevent hop/jump exploitation:
        #   - Only fires during single support (opposite foot must be planted)
        #   - Only fires when forward_velocity > 0.1 m/s
        push_off_rew = 0.0
        if forward_velocity > 0.1:
            # Right leg push-off: left foot planted, right foot leaving
            if left_contact and not right_contact:
                ayr_vel = self.data.qvel[self.ankle_y_right_idx - 7]
                if ayr_vel < 0:
                    push_off_rew += self.push_off_weight * min(abs(ayr_vel), 2.0)
            # Left leg push-off: right foot planted, left foot leaving
            elif right_contact and not left_contact:
                ayl_vel = self.data.qvel[self.ankle_y_left_idx - 7]
                if ayl_vel < 0:
                    push_off_rew += self.push_off_weight * min(abs(ayl_vel), 2.0)

        gait_reward += push_off_rew
        info['gait_reward/push_off_reward'] = float(push_off_rew)

        # ---- Hip Y anti-phase penalty (Gen2-09) ----
        # During normal walking, hip_y_right and hip_y_left should be
        # in anti-phase: one leg forward (negative) while the other is
        # back (positive). Their sum should be near zero.
        # When both swing in the same direction (passive/sync), sum >> 0.
        # Formula: -weight * (hip_y_right + hip_y_left)^2, capped at -15.
        # Gated on forward_velocity > 0.1 to avoid disturbing standing balance.
        hip_y_antiphase_pen = 0.0
        if forward_velocity > 0.1:
            hy_r = self.data.qpos[self.hip_y_right_idx]
            hy_l = self.data.qpos[self.hip_y_left_idx]
            sync  = hy_r + hy_l   # near 0 = anti-phase, far from 0 = in-sync
            raw_pen = -self.hip_y_antiphase_weight * (sync ** 2)
            hip_y_antiphase_pen = max(raw_pen, -15.0)

        gait_reward += hip_y_antiphase_pen
        info['gait_reward/hip_y_antiphase_pen'] = float(hip_y_antiphase_pen)
        info['gait_reward/hip_y_right']         = float(self.data.qpos[self.hip_y_right_idx])
        info['gait_reward/hip_y_left']          = float(self.data.qpos[self.hip_y_left_idx])

        # Update last contacts
        self.last_left_contact  = left_contact
        self.last_right_contact = right_contact

        return gait_reward, info

    # ------------------------------------------------------------------ #
    #  STEP                                                               #
    # ------------------------------------------------------------------ #
    def step(self, action):
        """Single environment step with full metric emission."""

        # --- Physics ---
        xy_before = self.data.qpos[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_after  = self.data.qpos[:2].copy()

        forward_velocity = (xy_after[0] - xy_before[0]) / self.dt   # +X is forward
        y_velocity       = (xy_after[1] - xy_before[1]) / self.dt

        # Lateral drift penalty
        lateral_penalty = -self.lateral_penalty_weight * abs(y_velocity)
        total_lateral   = abs(self.data.qpos[1] - self.episode_start_y)
        if total_lateral > 0.5:
            lateral_penalty -= 10.0

        # Velocity history
        self.velocity_history.append(forward_velocity)
        if len(self.velocity_history) > 50:
            self.velocity_history.pop(0)

        # Path deviation tracking
        self.path_deviation_history.append(float(self.data.qpos[1]))
        if len(self.path_deviation_history) > 50:
            self.path_deviation_history.pop(0)

        # ---------------------------------------------------------------- #
        #  SECONDARY REWARD COMPONENTS                                     #
        # ---------------------------------------------------------------- #

        ctrl_cost    = self.ctrl_cost_weight    * float(np.sum(action ** 2))
        contact_cost = self.contact_cost_weight * float(np.sum(self.contact_forces ** 2))

        # --- Straight-path bonus ---
        straight_path_bonus = 0.0
        if len(self.path_deviation_history) > 20:
            pv = np.var(self.path_deviation_history[-20:])
            straight_path_bonus = 5.0 if pv < 0.01 else 2.0 if pv < 0.05 else 0.0

        # --- Torso geometry ---
        torso_quat       = self.data.xquat[self.torso_id]
        torso_quat_w     = float(torso_quat[0])
        torso_roll       = 2.0 * np.arcsin(float(np.clip(torso_quat[1], -1.0, 1.0)))
        torso_pitch      = 2.0 * np.arcsin(float(np.clip(torso_quat[2], -1.0, 1.0)))
        lateral_tilt_pen = -3.0 * torso_roll  ** 2
        pitch_penalty    = -5.0 * torso_pitch ** 2

        # --- Upright bonus / penalty ---
        upright_bonus   = self.upright_weight * torso_quat_w
        upright_penalty = -5.0 * (0.95 - torso_quat_w) ** 2 if torso_quat_w < 0.95 else 0.0

        # --- Torso / head stability ---
        tav              = self.data.cvel[self.torso_id, 3:]
        torso_stab_pen   = float(-self.torso_stability_weight * np.sum(tav ** 2))
        try:
            hid          = self.model.body('head').id
            hav          = self.data.cvel[hid, 3:]
            head_stab_pen = float(-self.head_stability_weight * np.sum(hav ** 2))
        except Exception:
            head_stab_pen = 0.0

        # --- Hip stability ---
        pelvis_lat_vel  = float(self.data.cvel[self.pelvis_id, 1])
        hip_stab_pen    = -2.0 * pelvis_lat_vel ** 2

        # --- Arm positions ---
        arm_pos         = self.data.qpos[22:28]
        arm_vel         = self.data.qvel[21:27]
        arm_flail_pen   = float(-self.arm_penalty_weight * np.sum(arm_vel ** 2))
        arm_pos_pen     = float(-0.05  * np.sum(arm_pos ** 2))

        # --- Gait rewards ---
        gait_reward, gait_info = self._calculate_gait_rewards(forward_velocity)

        # --- Joint constraints ---
        jc_pen, jc_info = self._calculate_joint_constraint_penalties()

        # --- Arm swing ---
        lc = self._check_foot_contact_mujoco(self.left_foot_geoms)
        rc = self._check_foot_contact_mujoco(self.right_foot_geoms)
        arm_swing_rew, arm_swing_info = self._calculate_arm_swing_rewards(lc, rc)

        # --- Termination and alive reward ---
        terminated    = not self.is_healthy
        healthy_rew   = self.healthy_reward if not terminated else 0.0

        # ---------------------------------------------------------------- #
        #  STANDING-PHASE REWARD COMPONENTS                               #
        # ---------------------------------------------------------------- #
        height_z        = float(self.data.qpos[2])
        target_height   = 1.4
        height_dev      = abs(height_z - target_height)

        smooth_balance  = 10.0 * np.exp(-2.0 * height_dev)
        orient_dev      = max(0.0, 0.9 - torso_quat_w)
        smooth_upright  = 5.0  * np.exp(-3.0 * orient_dev)
        balance_reward  = smooth_balance + smooth_upright
        if not self.is_healthy:
            balance_reward = -5.0          # V27: reduced from -10

        height_reward   = 3.0 * np.exp(-height_dev)
        velocity_penalty= -0.03 * float(np.sum(self.data.qvel[6:] ** 2))

        # ---------------------------------------------------------------- #
        #  PRIMARY VELOCITY REWARD (Linear + Gaussian)                    #
        # ---------------------------------------------------------------- #
        velocity_reward = 0.0
        if len(self.velocity_history) > 10:
            avg_vel    = float(np.mean(self.velocity_history[-10:]))
            linear_push = self.linear_velocity_weight  * max(0.0, avg_vel)
            gaussian_pull = self.gaussian_velocity_weight * np.exp(
                -((avg_vel - self.target_velocity) ** 2) / (2 * self.velocity_sigma ** 2)
            )
            speed_pen   = -5.0 * (0.1 - forward_velocity) if forward_velocity < 0.1 else 0.0
            velocity_reward = float(linear_push + gaussian_pull + speed_pen)

        # Sustained speed bonus
        sustained_bonus = 10.0 if (
            len(self.velocity_history) >= 20 and
            all(v > 0.2 for v in self.velocity_history[-20:])
        ) else 0.0

        # Forward requirement penalty (light; capped)
        avg_fwd_vel = float(np.mean(self.velocity_history[-20:])) if len(self.velocity_history) > 20 else 0.0
        fwd_req_pen = self.forward_requirement_penalty if avg_fwd_vel < 0.1 else 0.0

        # Step rate bonus
        step_rate_bonus = 0.0
        if self.data.time > 1.0:
            sps = self.steps_taken / self.data.time
            if sps > 0.8:
                step_rate_bonus = 5.0
            elif sps < 0.3:
                step_rate_bonus = -10.0

        # Neutral-pose regularisation
        neutral_pose_pen = -0.05 * float(np.sum(self.data.qpos[7:] ** 2))

        # ---------------------------------------------------------------- #
        #  BUILD METRIC DICT (all 72 keys — always emitted every step)    #
        # ---------------------------------------------------------------- #
        all_metrics = {
            # Base rewards
            'base_reward/healthy':       float(healthy_rew),
            'base_reward/ctrl_cost':     float(-ctrl_cost),
            'base_reward/contact_cost':  float(-contact_cost),
            'base_reward/gait_total':    float(self.gait_reward_weight * gait_reward),
            'base_reward/total_reward':  0.0,   # filled below

            # Environment metrics
            'env_metrics/forward_velocity': float(forward_velocity),
            'env_metrics/x_position':       float(xy_after[0]),
            'env_metrics/y_position':       float(xy_after[1]),
            'env_metrics/z_position':       float(height_z),
            'env_metrics/steps_taken':      float(self.steps_taken),

            # Curriculum
            'curriculum/walking_progress':         float(self.walking_progress),
            'curriculum/alpha_standing':            0.0,
            'curriculum/alpha_walking':             0.0,
            'curriculum/standing_rew':              0.0,
            'curriculum/walking_rew':               0.0,
            'curriculum/progressive_forward_weight':0.0,
            'curriculum/gait_penalty_scale':        0.0,
            'curriculum/ultra_simple_mode':         0.0,

            # Standing phase
            'standing_phase/balance_reward':  float(balance_reward),
            'standing_phase/height_reward':   float(height_reward),
            'standing_phase/velocity_penalty':float(velocity_penalty),
            'standing_phase/torso_upright':   float(smooth_upright),

            # Walking phase
            'walking_phase/enhanced_forward_reward': 0.0,
            'walking_phase/scaled_gait_reward':      0.0,
            'walking_phase/velocity_tracking':       0.0,
            'walking_phase/sustained_speed_bonus':   0.0,

            # Ultra-simple
            'ultra_simple/balance_reward':     0.0,
            'ultra_simple/upright_reward':     0.0,
            'ultra_simple/neutral_pose_penalty':0.0,

            # Joint constraints (initialised; overwritten below)
            'joint_constraints/total_penalty':  0.0,
            'joint_constraints/progress_scale': 0.0,
            'joint_constraints/shoulder1_penalty':0.0,
            'joint_constraints/shoulder2_penalty':0.0,
            'joint_constraints/elbow_penalty':   0.0,
            'joint_constraints/ankle_y_penalty': 0.0,
            'joint_constraints/ankle_x_penalty': 0.0,
            'joint_constraints/hip_z_penalty':   0.0,
            'joint_constraints/shoulder1_right': 0.0,
            'joint_constraints/shoulder1_left':  0.0,
            'joint_constraints/shoulder2_right': 0.0,
            'joint_constraints/shoulder2_left':  0.0,
            'joint_constraints/elbow_right':     0.0,
            'joint_constraints/elbow_left':      0.0,
            'joint_constraints/ankle_y_right':   0.0,
            'joint_constraints/ankle_y_left':    0.0,
            'joint_constraints/ankle_x_right':   0.0,
            'joint_constraints/ankle_x_left':    0.0,
            'joint_constraints/hip_z_right':     0.0,
            'joint_constraints/hip_z_left':      0.0,
            'joint_constraints/abdomen_penalty': 0.0,
            'joint_constraints/abdomen_x':       0.0,
            'joint_constraints/abdomen_y':       0.0,
            'joint_constraints/abdomen_z':       0.0,

            # Gait rewards (initialised; overwritten by gait_info below)
            'gait_reward/alternation_reward':      0.0,
            'gait_reward/step_frequency_reward':   0.0,
            'gait_reward/stride_length_reward':    0.0,
            'gait_reward/static_standing_penalty': 0.0,
            'gait_reward/contact_pattern_rew':     0.0,
            'gait_reward/wide_stance_penalty':     0.0,
            'gait_reward/narrow_stance_penalty':   0.0,
            'gait_reward/clearance_rew':           0.0,
            'gait_reward/com_smoothness_pen':      0.0,
            'gait_reward/orientation_pen':         0.0,
            'gait_reward/torso_rotation_pen':      0.0,
            'gait_reward/foot_slide_pen':          0.0,
            'gait_reward/positional_lag_penalty':  0.0,
            'gait_reward/push_off_reward':         0.0,
            'gait_reward/hip_y_antiphase_pen':     0.0,   # Gen2-09
            'gait_reward/hip_y_right':             0.0,   # Gen2-09
            'gait_reward/hip_y_left':              0.0,   # Gen2-09

            # Env metrics (foot heights and contact states emitted by gait_info; defaults as safety net)
            'env_metrics/left_foot_height':  0.0,
            'env_metrics/right_foot_height': 0.0,
            'env_metrics/left_contact':      0.0,
            'env_metrics/right_contact':     0.0,
            'env_metrics/no_contact':        0.0,
            'env_metrics/both_contact':      0.0,
            'env_metrics/single_support':    0.0,

            # Arm swing (initialised; overwritten by arm_swing_info below)
            'arm_swing/movement_reward':    0.0,
            'arm_swing/coordination_reward':0.0,
            'arm_swing/total_reward':       0.0,
            'arm_swing/shoulder1_right':    0.0,
            'arm_swing/shoulder1_left':     0.0,
        }

        # Merge sub-dicts
        all_metrics.update(gait_info)
        all_metrics.update(jc_info)
        all_metrics.update(arm_swing_info)

        # ---------------------------------------------------------------- #
        #  PHASE-SPECIFIC REWARD CALCULATION                              #
        # ---------------------------------------------------------------- #
        if self.training_phase == "standing" and self.walking_progress < 0.3:
            # ── Ultra-simple balance phase ─────────────────────────────
            simple_balance = 10.0 if self.is_healthy else -50.0
            simple_upright = 5.0  if torso_quat_w > 0.85 else -5.0
            neutral_pen    = -0.1 * float(np.sum(self.data.qpos[7:] ** 2))
            total_reward   = simple_balance + simple_upright + neutral_pen - ctrl_cost

            all_metrics.update({
                'ultra_simple/balance_reward':      float(simple_balance),
                'ultra_simple/upright_reward':      float(simple_upright),
                'ultra_simple/neutral_pose_penalty':float(neutral_pen),
                'curriculum/ultra_simple_mode':     1.0,
                'curriculum/alpha_standing':        1.0,
                'curriculum/alpha_walking':         0.0,
                'curriculum/standing_rew':          float(total_reward),
                'base_reward/total_reward':         float(total_reward),
                'standing_phase/balance_reward':    float(balance_reward),
                'standing_phase/height_reward':     float(height_reward),
                'standing_phase/velocity_penalty':  float(velocity_penalty),
                'standing_phase/torso_upright':     float(smooth_upright),
            })

        else:
            # ── Standard curriculum blend (standing → walking) ─────────
            standing_reward = (balance_reward + height_reward +
                               velocity_penalty + smooth_upright - ctrl_cost)
            standing_reward += neutral_pose_pen

            prog_fwd_weight   = self.walking_progress * self.forward_reward_weight
            enhanced_fwd_rew  = prog_fwd_weight * forward_velocity
            gait_pen_scale    = max(0.3, self.walking_progress)
            scaled_gait_rew   = gait_reward * gait_pen_scale

            walking_reward = (
                enhanced_fwd_rew
                + healthy_rew
                - ctrl_cost
                - contact_cost
                + self.gait_reward_weight  * scaled_gait_rew
                + velocity_reward
                + sustained_bonus
                + straight_path_bonus
                + upright_penalty
                + head_stab_pen
                + lateral_penalty
                + hip_stab_pen
                + lateral_tilt_pen
                + pitch_penalty
                + arm_flail_pen
                + arm_pos_pen
                + upright_bonus
                + torso_stab_pen
                + jc_pen
                + arm_swing_rew
                + step_rate_bonus
                + fwd_req_pen
                + neutral_pose_pen
            )

            alpha_s      = 1.0 - self.walking_progress
            alpha_w      = self.walking_progress
            total_reward = alpha_s * standing_reward + alpha_w * walking_reward

            # Insufficient-movement penalty
            dist_covered = self.data.qpos[0] - self.episode_start_x
            if len(self.step_time_history) > 20 and dist_covered < 0.1:
                total_reward += -20.0

            all_metrics.update({
                'standing_phase/balance_reward':             float(balance_reward),
                'standing_phase/height_reward':              float(height_reward),
                'standing_phase/velocity_penalty':           float(velocity_penalty),
                'standing_phase/torso_upright':              float(smooth_upright),
                'walking_phase/enhanced_forward_reward':     float(enhanced_fwd_rew),
                'walking_phase/scaled_gait_reward':          float(scaled_gait_rew),
                'walking_phase/velocity_tracking':           float(velocity_reward),
                'walking_phase/sustained_speed_bonus':       float(sustained_bonus),
                'curriculum/standing_rew':                   float(standing_reward),
                'curriculum/walking_rew':                    float(walking_reward),
                'curriculum/alpha_standing':                 float(alpha_s),
                'curriculum/alpha_walking':                  float(alpha_w),
                'curriculum/progressive_forward_weight':     float(prog_fwd_weight),
                'curriculum/gait_penalty_scale':             float(gait_pen_scale),
                'base_reward/total_reward':                  float(total_reward),
            })

        return self._get_obs(), float(total_reward), terminated, False, all_metrics