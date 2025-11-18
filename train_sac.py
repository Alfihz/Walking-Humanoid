import torch
from stable_baselines3 import SAC # Import SAC
from sb3_contrib import TQC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
# Import VecNormalize if you plan to use it (often good for MuJoCo)
# from stable_baselines3.common.vec_env import VecNormalize
import datetime
import os
from HumanoidWalkEnv import HumanoidWalkEnv # Ensure this is correctly importable
# from GeminiHumanoidWalkEnv import HumanoidWalkEnv
import re

def create_next_numbered_folder(prefix, base_path="."):
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    existing_folders = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

    max_number = 0
    for folder in existing_folders:
        match = pattern.match(folder)
        if match:
            number = int(match.group(1))
            max_number = max(max_number, number)

    new_folder_name = f"{prefix}_{max_number + 1}"
    new_folder_path = os.path.join(base_path, new_folder_name)
    os.makedirs(new_folder_path)
    return new_folder_path

# Function to create a single monitored environment
def make_env(rank: int, seed: int = 0):
    """
    Utility function for multiprocessed env.
    :param rank: index of the subprocess
    :param seed: the initial seed for RNG
    """
    def _init():
        # Create the base environment
        env = HumanoidWalkEnv(xml_file=XML_PATH)
        # IMPORTANT: Set seed if running multiple processes for reproducibility
        # env.seed(seed + rank) # Deprecated in Gymnasium, use reset(seed=...)
        env.reset(seed=seed + rank) # Use reset with seed

        # Create the monitor directory for this specific process
        monitor_path = os.path.join(TENSORBOARD_LOG_DIR, str(rank))
        # os.makedirs(monitor_path, exist_ok=True)

        # Wrap the environment with Monitor
        env = Monitor(env, filename=monitor_path, info_keywords=info_keywords_to_log)
        return env
    # set_global_seeds(seed) # Deprecated
    return _init

# --- Configuration ---
rl_model_name = "sac" # Changed model name
XML_FILENAME = "assets/humanoid_180_75.xml" # <<-- Your corrected XML file
SAVE_FREQ = 500_000 # Save every 2 million timesteps
TOTAL_TIMESTEPS = 20_000_000 # Total training duration

# Define base directories
TENSORBOARD_LOG_DIR = f"./tensorboard/{rl_model_name}_humanoid_logs/"
# Directory specifically for periodic checkpoints
MODEL_CHECKPOINT_DIR = f"./models/sac_humanoid_checkpoints/SAC_!"#create_next_numbered_folder("SAC", f"./models/{rl_model_name}_humanoid_checkpoints/") #f"./models/{rl_model_name}_humanoid_checkpoints/SAC_2" 

LOAD_MODEL_PATH = f"./models/sac_humanoid_checkpoints/load_models_for_finetune/sac_humanoid_walker_2500000_steps"

info_keywords_to_log = (
    'base_reward/forward', 'base_reward/healthy', 'base_reward/ctrl_cost',
    'base_reward/gait_total', 'gait_reward/contact_pattern_rew', 'gait_reward/clearance_rew',
    'gait_reward/com_smoothness_pen', 'gait_reward/clearance_achieved',
    'gait_reward/left_contact', 'gait_reward/right_contact'
)

# Create directories if they don't exist
os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)
os.makedirs(MODEL_CHECKPOINT_DIR, exist_ok=True)

if __name__ == "__main__":
    # Check if XML file exists
    SCRIPT_DIR = os.path.dirname(__file__)
    XML_PATH = os.path.join(SCRIPT_DIR, XML_FILENAME)
    if not os.path.exists(XML_PATH):
        print(f"Error: XML file not found at {XML_PATH}")
        exit()
    else:
        print(f"Using XML file: {XML_PATH}")

    # Environment setup
    num_envs = 6 # SAC is off-policy, can still benefit from multiple envs for faster data collection
    # Create the environment, passing the XML path via lambda
    if num_envs > 1:
        vec_env = SubprocVecEnv([make_env(i) for i in range(num_envs)], start_method="spawn")
    else:
        vec_env = DummyVecEnv([make_env(0)])
    # vec_env = make_vec_env(lambda: HumanoidWalkEnv(xml_file=XML_PATH), n_envs=num_envs, monitor_dir=TENSORBOARD_LOG_DIR + "/SAC_2", monitor_kwargs={'info_keywords': info_keywords_to_log})

    # Optional: Wrap with VecNormalize
    # Normalizes observations and optionally rewards. Often improves performance on MuJoCo.
    # Make sure to save/load the normalization stats along with the model!
    # vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)
    # print("Using VecNormalize.")

    # --- Device Selection ---
    if torch.cuda.is_available():
        device = "cuda"
        print(f"CUDA available! Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("CUDA not available, using CPU.")
    print(f"Using device: {device}")


    # --- Instantiate the SAC Model ---
    print(f"Instantiating {rl_model_name.upper()} model...")

    # SAC Hyperparameters - Defaults are a starting point, but tuning IS often required for complex MuJoCo tasks.
    # Common parameters to potentially tune:
    # learning_rate: Often 3e-4 or slightly lower (e.g., 1e-4) works well. Default is 3e-4.
    # buffer_size: Size of the replay buffer. Needs to be large for off-policy (e.g., 1,000,000). Default is 1M.
    # learning_starts: How many steps to collect before learning starts (e.g., 10000). Default is 100.
    # batch_size: Samples per gradient update (e.g., 256). Default is 256.
    # tau: Soft update coefficient (e.g., 0.005). Default is 0.005.
    # gamma: Discount factor (e.g., 0.99). Default is 0.99.
    # ent_coef: Entropy coefficient ('auto' lets it be learned, often good). Default is 'auto'.
    # gradient_steps: How many gradient updates per environment step. Default is 1.

    # Refer to Stable Baselines3 documentation and RL Zoo (https://github.com/DLR-RM/rl-baselines3-zoo)
    # for commonly used hyperparameters for MuJoCo environments like Humanoid.

    # Using mostly defaults to start, but with a slightly larger learning_starts
    model = SAC(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=TENSORBOARD_LOG_DIR,
        device=device,
        buffer_size=1_000_000,        # Default: 1M steps
        learning_rate=3e-4,           # Default: 3e-4 (Can try 1e-4 if learning is unstable)
        batch_size=256,               # Default: 256
        learning_starts=10000,        # Default: 100. Increased to collect more diverse initial data.
        gamma=0.99,                   # Default: 0.99
        tau=0.005,                    # Default: 0.005
        ent_coef='auto',              # Default: 'auto' (Recommended)
        gradient_steps=64,           # Default: 1
        train_freq=(1, "step"),     # Default: (1, "step") - Train every env step
        # seed=None,                  # Set for reproducibility if needed
        # policy_kwargs=None          # e.g., dict(net_arch=[256, 256])
    )

    # model = SAC.load(
    #     LOAD_MODEL_PATH,
    #     env=vec_env, # Pass the environment
    #     device=device, # Specify device (optional, SB3 often detects)
    #     # You can override some parameters here if needed for fine-tuning,
    #     # but for just continuing training, often not necessary.
    #     learning_rate = 5e-5, # Keep original LR for now unless fine-tuning rewards
    #     # learning_starts=0 # Set learning_starts to 0 since buffer is presumably full enough
    # )
    # Reset learning_starts if loading a full model with buffer
    # model.learning_starts = max(0, model.learning_starts - model.num_timesteps) # Adjust based on how SB3 handles this, often just set to 0
    # model.learning_starts = 0 # Simpler: Assume buffer is ready
    print("Model loaded successfully.")
    print(f"Continuing training from {model.num_timesteps} timesteps (internal counter reset by load, training continues).")


    print("Model and Environment Setup Complete.")
    print(f"Observation Space: {vec_env.observation_space}")
    print(f"Action Space: {vec_env.action_space}")


    # --- Setup Checkpoint Callback ---
    # Saves the model (and VecNormalize stats if used) periodically
    checkpoint_callback = CheckpointCallback(
    save_freq = max(SAVE_FREQ // num_envs, 1), #save_freq=SAVE_FREQ,
    save_path = MODEL_CHECKPOINT_DIR,
    name_prefix = f"{rl_model_name}_humanoid_walker",
    save_replay_buffer = True,
    save_vecnormalize = False,
    verbose=1
    )
    print(f"Model checkpoints will be saved every {SAVE_FREQ:,} total timesteps to: {MODEL_CHECKPOINT_DIR}")


    # --- Training ---
    print(f"\n--- Training Start ---")
    start_time_dt = datetime.datetime.now()
    start_time_str = start_time_dt.strftime("%Y-%m-%d %H:%M:%S")
    print(f"Start Time: {start_time_str}")
    print(f"Total Timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"Logging to TensorBoard: {TENSORBOARD_LOG_DIR}")
    print("Run 'tensorboard --logdir ./tensorboard/' in another terminal to monitor.")
    print("-" * 30)

    try:
        # Pass the callback to the learn method
        # model.set_logger()
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            log_interval=10, # Log stats every 10 updates (an update may involve multiple gradient steps)
            progress_bar=True,
            callback=checkpoint_callback, # Use the checkpoint callback
            reset_num_timesteps=False # use only when loading model
        )
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        print("Attempting to save the model before exiting...")
        error_save_path = os.path.join(MODEL_CHECKPOINT_DIR, f"{rl_model_name}_humanoid_walker_ERROR_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        model.save(error_save_path)
        # If using VecNormalize, save its stats too
        # if isinstance(vec_env, VecNormalize):
        #    vec_env.save(os.path.join(MODEL_CHECKPOINT_DIR, f"{rl_model_name}_vecnormalize_ERROR.pkl"))
        print(f"Model saved due to error at: {error_save_path}.zip") # SB3 adds .zip automatically

    finally: # Ensure environment closure even if error occurs
        print("-" * 30)
        print("--- Training Process End ---")
        end_time_dt = datetime.datetime.now()
        end_time_str = end_time_dt.strftime("%Y-%m-%d %H:%M:%S")
        print(f"End Time: {end_time_str}")
        training_duration = end_time_dt - start_time_dt
        print(f"Total Training Duration: {training_duration}")
        print("-" * 30)

        # --- Final Save (Optional but Recommended) ---
        # Although checkpoints are saved, saving the final model explicitly is good practice
        final_model_name = f"{rl_model_name}_humanoid_walker_final_{TOTAL_TIMESTEPS}_steps"
        final_model_path = os.path.join(MODEL_CHECKPOINT_DIR, final_model_name)
        print(f"Saving final model to {final_model_path}.zip")
        model.save(final_model_path)
        # Save VecNormalize stats if used
        # if isinstance(vec_env, VecNormalize):
        #    vec_env.save(os.path.join(MODEL_CHECKPOINT_DIR, f"{rl_model_name}_vecnormalize_final.pkl"))
        print("Final model saved.")
        # --------------------------------------------

        print("Closing environment...")
        vec_env.close()
        print("Done.")
