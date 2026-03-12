#!/usr/bin/env python3
"""
TQC Training Script - OPTIMIZED FOR RTX 5070 Ti
Increased parallel environments and adjusted hyperparameters for powerful GPU
"""

import torch
import os
import datetime
import numpy as np
import shutil
from sb3_contrib import TQC
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from HumanoidWalkEnv import HumanoidWalkEnv


# ============================================================================
# Configuration - OPTIMIZED FOR RTX 5070 Ti
# ============================================================================
XML_FILENAME = "assets/humanoid_180_75.xml"
SAVE_FREQ = 1_000_000  
TOTAL_TIMESTEPS = 20_000_000

# Curriculum phases
STANDING_PHASE_TIMESTEPS = int(0.15 * TOTAL_TIMESTEPS)  # 20% standing 
CURRICULUM_END_TIMESTEPS = int(0.45 * TOTAL_TIMESTEPS)  # 60% total

# Model loading
LOAD_MODEL_PATH = None

# Auto-increment run directories
AUTO_INCREMENT_DIRS = True

# ============================================================================
# INFO_KEYWORDS - MUST MATCH Gen2-10 METRIC NAMES EXACTLY
# Total: 82 metrics across 9 groups
# ============================================================================
INFO_KEYWORDS = (
    # ── Base rewards (5) ─────────────────────────────────────────────────
    'base_reward/healthy',
    'base_reward/ctrl_cost',
    'base_reward/contact_cost',
    'base_reward/gait_total',
    'base_reward/total_reward',

    # ── Environment metrics (12) ──────────────────────────────────────────
    'env_metrics/forward_velocity',
    'env_metrics/x_position',
    'env_metrics/y_position',
    'env_metrics/z_position',
    'env_metrics/steps_taken',
    'env_metrics/left_contact',
    'env_metrics/right_contact',
    'env_metrics/no_contact',
    'env_metrics/both_contact',
    'env_metrics/single_support',
    'env_metrics/left_foot_height',
    'env_metrics/right_foot_height',

    # ── Curriculum metrics (8) ────────────────────────────────────────────
    'curriculum/walking_progress',
    'curriculum/alpha_standing',
    'curriculum/alpha_walking',
    'curriculum/standing_rew',
    'curriculum/walking_rew',
    'curriculum/progressive_forward_weight',
    'curriculum/gait_penalty_scale',
    'curriculum/ultra_simple_mode',

    # ── Standing phase metrics (4) ────────────────────────────────────────
    'standing_phase/balance_reward',
    'standing_phase/height_reward',
    'standing_phase/velocity_penalty',
    'standing_phase/torso_upright',

    # ── Walking phase metrics (4) ─────────────────────────────────────────
    'walking_phase/enhanced_forward_reward',
    'walking_phase/scaled_gait_reward',
    'walking_phase/velocity_tracking',
    'walking_phase/sustained_speed_bonus',

    # ── Ultra-simple phase metrics (3) ────────────────────────────────────
    'ultra_simple/balance_reward',
    'ultra_simple/upright_reward',
    'ultra_simple/neutral_pose_penalty',

    # ── Gait rewards (17) ─────────────────────────────────────────────────
    'gait_reward/alternation_reward',
    'gait_reward/step_frequency_reward',
    'gait_reward/stride_length_reward',
    'gait_reward/static_standing_penalty',
    'gait_reward/contact_pattern_rew',
    'gait_reward/wide_stance_penalty',
    'gait_reward/narrow_stance_penalty',
    'gait_reward/clearance_rew',
    'gait_reward/com_smoothness_pen',
    'gait_reward/orientation_pen',
    'gait_reward/torso_rotation_pen',
    'gait_reward/foot_slide_pen',
    'gait_reward/positional_lag_penalty',   # Gen2-05
    'gait_reward/push_off_reward',          # Gen2-07
    'gait_reward/hip_y_excursion_pen',      # Gen2-10
    'gait_reward/hip_y_excursion_right',    # Gen2-10
    'gait_reward/hip_y_excursion_left',     # Gen2-10

    # ── Joint constraints (24) ────────────────────────────────────────────
    'joint_constraints/total_penalty',
    'joint_constraints/progress_scale',
    'joint_constraints/abdomen_penalty',
    'joint_constraints/abdomen_x',
    'joint_constraints/abdomen_y',
    'joint_constraints/abdomen_z',
    'joint_constraints/shoulder1_penalty',
    'joint_constraints/shoulder1_right',
    'joint_constraints/shoulder1_left',
    'joint_constraints/shoulder2_penalty',
    'joint_constraints/shoulder2_right',
    'joint_constraints/shoulder2_left',
    'joint_constraints/elbow_penalty',
    'joint_constraints/elbow_right',
    'joint_constraints/elbow_left',
    'joint_constraints/ankle_y_penalty',
    'joint_constraints/ankle_y_right',
    'joint_constraints/ankle_y_left',
    'joint_constraints/ankle_x_penalty',
    'joint_constraints/ankle_x_right',
    'joint_constraints/ankle_x_left',
    'joint_constraints/hip_z_penalty',      # Gen2-06
    'joint_constraints/hip_z_right',        # Gen2-06
    'joint_constraints/hip_z_left',         # Gen2-06

    # ── Arm swing (5) ─────────────────────────────────────────────────────
    'arm_swing/movement_reward',
    'arm_swing/coordination_reward',
    'arm_swing/total_reward',
    'arm_swing/shoulder1_right',
    'arm_swing/shoulder1_left',
)


# ============================================================================
# Helper Functions
# ============================================================================
def create_next_run_dir(base_dir="./models/"):
    """Create a new numbered directory"""
    import re
    
    os.makedirs(base_dir, exist_ok=True)
    
    pattern = re.compile(r"^tqc_(\d+)$")
    existing_dirs = [d for d in os.listdir(base_dir) 
                     if os.path.isdir(os.path.join(base_dir, d))]
    
    max_num = 0
    for dirname in existing_dirs:
        match = pattern.match(dirname)
        if match:
            num = int(match.group(1))
            max_num = max(max_num, num)
    
    new_dir_name = f"tqc_{max_num + 1:02d}"
    new_dir_path = os.path.join(base_dir, new_dir_name)
    os.makedirs(new_dir_path)
    
    print(f"Created new run directory: {new_dir_path}")
    return new_dir_path


def make_env(rank: int, xml_path: str, tensorboard_dir: str, seed: int = 0, training_phase: str = "standing"):
    """Create a single monitored environment"""
    def _init():
        env = HumanoidWalkEnv(xml_file=xml_path, training_phase=training_phase)
        env.reset(seed=seed + rank)
        
        monitor_dir = os.path.join(tensorboard_dir, "monitors")
        os.makedirs(monitor_dir, exist_ok=True)
        monitor_file = os.path.join(monitor_dir, f"env_{rank}")
        
        env = Monitor(env, filename=monitor_file, info_keywords=INFO_KEYWORDS)
        return env
    return _init


# ============================================================================
# Callbacks
# ============================================================================
class CurriculumProgressCallback(BaseCallback):
    """Manages curriculum progression through training phases"""
    def __init__(self, start_timestep: int, end_timestep: int, verbose: int = 0):
        super().__init__(verbose)
        assert end_timestep > start_timestep
        self.start_timestep = start_timestep
        self.end_timestep = end_timestep
        self._phase = 0
        self._last_progress = -1

    def _set_phase_all_envs(self, phase: str, progress: float = 0.0):
        """Set training phase across all parallel environments"""
        base_env = self.model.env.venv if hasattr(self.model.env, "venv") else self.model.env

        if hasattr(base_env, "remotes"):
            for remote in base_env.remotes:
                remote.send(('env_method', ('set_training_phase', [phase], {'progress': progress})))
                remote.recv()
        elif hasattr(base_env, "envs"):
            for env in base_env.envs:
                target = env.env if hasattr(env, 'env') else env
                if hasattr(target, "set_training_phase"):
                    target.set_training_phase(phase, progress=progress)

    def _on_step(self) -> bool:
        t = self.num_timesteps

        # Phase 1: Pure standing (progress 0.0)
        if t < self.start_timestep:
            if self._phase != 1:
                self._phase = 1
                if self.verbose:
                    print(f"\n[Curriculum] Phase 1: Ultra-simple balance (0-20%)")
                self._set_phase_all_envs("standing", progress=0.0)
            self.logger.record("curriculum/phase", 1)
            self.logger.record("curriculum/progress", 0.0)
            return True

        # Phase 2: Transition (progress 0.3 -> 1.0)
        if self.start_timestep <= t < self.end_timestep:
            if self._phase != 2:
                self._phase = 2
                if self.verbose:
                    print(f"\n[Curriculum] Phase 2: Transition to walking (20-60%)")
            
            # Map timestep progress to walking_progress 0.3 -> 1.0
            timestep_progress = (t - self.start_timestep) / (self.end_timestep - self.start_timestep)
            walking_progress = 0.3 + (0.7 * timestep_progress)
            walking_progress = min(1.0, max(0.3, walking_progress))
            
            # Only update if progress changed significantly
            if abs(walking_progress - self._last_progress) > 0.01:
                self._set_phase_all_envs("walking", progress=walking_progress)
                self._last_progress = walking_progress
                if self.verbose and int(timestep_progress * 100) % 10 == 0:
                    print(f"[Curriculum] Walking progress: {walking_progress:.2f}")
            
            self.logger.record("curriculum/phase", 2)
            self.logger.record("curriculum/progress", float(walking_progress))
            return True

        # Phase 3: Pure walking (progress 1.0)
        if self._phase != 3:
            self._phase = 3
            if self.verbose:
                print(f"\n[Curriculum] Phase 3: Full walking mode (60-100%)")
            self._set_phase_all_envs("walking", progress=1.0)
        
        self.logger.record("curriculum/phase", 3)
        self.logger.record("curriculum/progress", 1.0)
        return True


class CustomMetricsCallback(BaseCallback):
    """Enhanced metrics logging"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.best_reward = -float('inf')

    def _on_step(self) -> bool:
        if len(self.locals.get("infos", [])) > 0:
            for info in self.locals["infos"]:
                if info:
                    for key, value in info.items():
                        if isinstance(value, (int, float, np.integer, np.floating)):
                            self.logger.record(f"{key}", value)
        return True


# ============================================================================
# Main Training
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("TQC HUMANOID WALKER")
    print("=" * 70)
    
    # GPU check
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🎮 GPU Detected: {gpu_name}")
        
        # Adjust environments based on GPU
        if "5070" in gpu_name or "4090" in gpu_name or "4080" in gpu_name:
            NUM_ENVS = 24
        elif "4070" in gpu_name or "3090" in gpu_name or "3080" in gpu_name:
            NUM_ENVS = 12
        else:
            NUM_ENVS = 8
        print(f"📊 Using {NUM_ENVS} parallel environments")
    else:
        print("⚠️ No GPU detected, using CPU")
        NUM_ENVS = 8

    # Verify XML file
    SCRIPT_DIR = os.path.dirname(__file__)
    XML_PATH = os.path.join(SCRIPT_DIR, XML_FILENAME)
    if not os.path.exists(XML_PATH):
        print(f"❌ Error: XML file not found at {XML_PATH}")
        exit(1)
    print(f"✅ Using XML file: {XML_PATH}")

    # Setup directories
    if AUTO_INCREMENT_DIRS:
        MODEL_CHECKPOINT_DIR = create_next_run_dir()
        run_num = os.path.basename(MODEL_CHECKPOINT_DIR).split('_')[1]
        TENSORBOARD_LOG_DIR = f"./tensorboard/tqc_{run_num}"
        os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)
    else:
        MODEL_CHECKPOINT_DIR = "./models/tqc_01"
        TENSORBOARD_LOG_DIR = "./tensorboard/tqc_01"
        os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)
        os.makedirs(MODEL_CHECKPOINT_DIR, exist_ok=True)
    
    print(f"💾 Model checkpoints: {MODEL_CHECKPOINT_DIR}")
    print(f"📈 TensorBoard logs: {TENSORBOARD_LOG_DIR}")

    # Environment setup
    print(f"\n🏗️  Creating {NUM_ENVS} parallel environments...")
    
    if NUM_ENVS > 1:
        vec_env = SubprocVecEnv(
            [make_env(i, XML_PATH, TENSORBOARD_LOG_DIR, training_phase="standing") 
             for i in range(NUM_ENVS)], 
            start_method="spawn"
        )
    else:
        vec_env = DummyVecEnv([make_env(0, XML_PATH, TENSORBOARD_LOG_DIR, training_phase="standing")])

    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.)
    print("✅ Environments created with observation normalization")

    # Device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Add exploration noise
    n_actions = vec_env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions)
    )

    # Model configuration - Optimized for high-end GPU
    print("\n🤖 Creating TQC model with optimized hyperparameters...")
    
    if LOAD_MODEL_PATH:
        model = TQC.load(
            LOAD_MODEL_PATH,
            env=vec_env,
            device=device,
            learning_rate=3e-5,
        )
        print(f"📂 Model loaded from {LOAD_MODEL_PATH}")
    else:
        model = TQC(
            "MlpPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=TENSORBOARD_LOG_DIR,
            device=device,
            buffer_size=1_000_000,
            learning_rate=3e-4,
            batch_size=1024,
            learning_starts=25000,
            gamma=0.98,
            tau=0.02,
            ent_coef='auto',
            gradient_steps=1,
            train_freq=(1, "step"),
            action_noise=action_noise,
            policy_kwargs=dict(
                net_arch=[512, 400, 300],
                activation_fn=torch.nn.ReLU,
                n_critics=2,
            )
        )

    print("✅ Model created successfully")
    print(f"📊 Observation Space: {vec_env.observation_space}")
    print(f"🎯 Action Space: {vec_env.action_space}")

    # Callbacks
    curriculum_cb = CurriculumProgressCallback(
        start_timestep=STANDING_PHASE_TIMESTEPS,
        end_timestep=CURRICULUM_END_TIMESTEPS,
        verbose=1
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(SAVE_FREQ // NUM_ENVS, 1),
        save_path=MODEL_CHECKPOINT_DIR,
        name_prefix="tqc_humanoid_walker",
        save_replay_buffer=False,
        save_vecnormalize=True,
        verbose=1
    )

    custom_callback = CustomMetricsCallback(verbose=1)

    from stable_baselines3.common.callbacks import CallbackList
    callback_list = CallbackList([custom_callback, curriculum_cb, checkpoint_callback])

    # Training
    print(f"\n{'='*70}")
    print("🚀 TRAINING START - RTX 5070 Ti OPTIMIZED")
    print(f"{'='*70}")
    start_time = datetime.datetime.now()
    print(f"⏰ Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Total Timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"🏋️  Training Phases:")
    print(f"   Phase 1 (Balance):    0 - {STANDING_PHASE_TIMESTEPS:,} steps (20%)")
    print(f"   Phase 2 (Transition): {STANDING_PHASE_TIMESTEPS:,} - {CURRICULUM_END_TIMESTEPS:,} steps")
    print(f"   Phase 3 (Walking):    {CURRICULUM_END_TIMESTEPS:,} - {TOTAL_TIMESTEPS:,} steps")
    print(f"💾 Checkpoints every: {SAVE_FREQ:,} steps")
    print(f"🖥️  Parallel Environments: {NUM_ENVS}")
    print(f"📈 TensorBoard: tensorboard --logdir {TENSORBOARD_LOG_DIR}")
    print(f"{'='*70}\n")

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            log_interval=50,
            progress_bar=True,
            callback=callback_list,
            reset_num_timesteps=(LOAD_MODEL_PATH is None)
        )
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ ERROR DURING TRAINING: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Save final model
        print(f"\n{'='*70}")
        print("💾 Saving final model...")
        
        final_model_path = os.path.join(
            MODEL_CHECKPOINT_DIR, 
            f"tqc_humanoid_walker_final_{model.num_timesteps}_steps"
        )
        model.save(final_model_path)
        
        if isinstance(vec_env, VecNormalize):
            vec_env.save(os.path.join(MODEL_CHECKPOINT_DIR, "tqc_vecnormalize_final.pkl"))
        
        end_time = datetime.datetime.now()
        print(f"✅ Training Complete!")
        print(f"⏰ End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  Duration: {end_time - start_time}")
        print(f"💾 Model saved: {final_model_path}")
        print(f"{'='*70}\n")
        
        # Automatically zip the monitors folder
        monitors_dir = os.path.join(TENSORBOARD_LOG_DIR, 'monitors')
        if os.path.exists(monitors_dir):
            # Create zip filename with run info
            run_name = os.path.basename(MODEL_CHECKPOINT_DIR)
            zip_filename = f"{run_name}_monitors"
            zip_path = os.path.join(TENSORBOARD_LOG_DIR, zip_filename)
            
            print(f"📦 Zipping monitors folder...")
            try:
                shutil.make_archive(zip_path, 'zip', monitors_dir)
                print(f"✅ Monitors zipped: {zip_path}.zip")
            except Exception as e:
                print(f"⚠️  Warning: Failed to zip monitors folder: {e}")
        
        vec_env.close()
        print("👋 Done!")