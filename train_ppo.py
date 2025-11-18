import torch
from stable_baselines3 import PPO, SAC # Choose one
# Import CheckpointCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
import datetime
import os
# import re # No longer needed for folder incrementing during training
from HumanoidWalkEnv import HumanoidWalkEnv # Ensure this is correctly importable

# --- Configuration ---
rl_model_name = "ppo" # Use a consistent name
SAVE_FREQ = 2_000_000 # Save every 2 million timesteps
TOTAL_TIMESTEPS = 20_000_000 # Total training duration

# Define base directories
TENSORBOARD_LOG_DIR = f"./tensorboard/{rl_model_name}_humanoid_logs/"
# Directory specifically for periodic checkpoints
MODEL_CHECKPOINT_DIR = f"./models/{rl_model_name}_humanoid_checkpoints/"

# Create directories if they don't exist
os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)
os.makedirs(MODEL_CHECKPOINT_DIR, exist_ok=True)

# Environment setup
num_envs = 4
vec_env = make_vec_env(lambda: HumanoidWalkEnv(), n_envs=num_envs)

# --- Device Selection ---
if torch.cuda.is_available():
    print(f"CUDA available! Using GPU: {torch.cuda.get_device_name(0)}")
    device = "cuda"
else:
    print("CUDA not available, using CPU.")
    device = "cpu"

# --- Instantiate the Model ---
print(f"Instantiating {rl_model_name.upper()} model...")
model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    tensorboard_log=TENSORBOARD_LOG_DIR, # Log to the specific tensorboard dir
    device=device
)
print("Model and Environment Setup Complete.")
print(f"Observation Space: {vec_env.observation_space}")
print(f"Action Space: {vec_env.action_space}")


# --- Setup Checkpoint Callback ---
# This callback will save the model every SAVE_FREQ steps
checkpoint_callback = CheckpointCallback(
  save_freq=max(SAVE_FREQ // num_envs, 1), # Checkpoint frequency based on total steps
  save_path=MODEL_CHECKPOINT_DIR,
  name_prefix=f"{rl_model_name}_humanoid_walker", # Prefix for saved models
  save_replay_buffer=True, # Save replay buffer for off-policy algorithms like SAC/TD3
  save_vecnormalize=True, # Save VecNormalize statistics if used
)
print(f"Model checkpoints will be saved every {SAVE_FREQ:,} timesteps to: {MODEL_CHECKPOINT_DIR}")
# Note: save_freq in CheckpointCallback is based on the number of calls to the callback.
# In SB3, for on-policy algorithms like PPO, the callback's _on_step is called every env.step().
# So, we need to adjust the frequency based on the number of environments.
# The callback internally checks self.num_timesteps % (self.save_freq * self.n_envs) effectively for total steps.
# Let's adjust save_freq for clarity: save_freq is the number of *total* steps across all envs.
checkpoint_callback_corrected = CheckpointCallback(
  save_freq=SAVE_FREQ, # Save checkpoints every SAVE_FREQ total timesteps
  save_path=MODEL_CHECKPOINT_DIR,
  name_prefix=f"{rl_model_name}_humanoid_walker",
  save_replay_buffer=True,
  save_vecnormalize=True,
  verbose=1 # Print message when saving
)
print(f"Corrected: Model checkpoints will be saved every {SAVE_FREQ:,} total timesteps to: {MODEL_CHECKPOINT_DIR}")


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
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        log_interval=10,
        progress_bar=True,
        callback=checkpoint_callback_corrected # Use the checkpoint callback
    )
except Exception as e:
    print(f"An error occurred during training: {e}")
    print("Attempting to save the model before exiting...")
    error_save_path = os.path.join(MODEL_CHECKPOINT_DIR, f"{rl_model_name}_humanoid_walker_ERROR_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip")
    model.save(error_save_path)
    print(f"Model saved due to error at: {error_save_path}")
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
    final_model_path = os.path.join(MODEL_CHECKPOINT_DIR, f"{rl_model_name}_humanoid_walker_final_{TOTAL_TIMESTEPS}_steps.zip")
    print(f"Saving final model to {final_model_path}")
    model.save(final_model_path)
    print("Final model saved.")
    # --------------------------------------------

    print("Closing environment...")
    vec_env.close()
    print("Done.")