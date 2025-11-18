import os
import torch
import argparse
import numpy as np

from sb3_contrib import TQC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, VecVideoRecorder
from HumanoidWalkEnv import HumanoidWalkEnv


def evaluate(model_path, vecnormalize_path=None, episodes=5, render=True, record_dir=None, video_length=1000, track_camera=True):
    """
    Evaluate a trained humanoid model.

    Args:
    model_path: Path to trained .zip model.
    vecnormalize_path: Path to saved VecNormalize stats (if any).
    episodes: Number of episodes to run.
    render: If True, render simulation in real-time.
    record_dir: Directory to save video recordings.
    video_length: Maximum length of recorded video in steps.
    track_camera: If True, camera will follow the humanoid during recording.
    """

    render_mode = ""
    if record_dir:
        render_mode = "rgb_array"
    elif render:
        render_mode = "human"
    else:
        render_mode = None

    # Configure camera settings for tracking
    camera_config = None
    if track_camera:
        # Use MuJoCo's built-in tracking camera configuration
        camera_config = {
            'type': 1,  # mjCAMERA_TRACKING
            'trackbodyid': 1,  # Track the torso (body id 1 is typically the main body)
            'distance': 4.0,
            'elevation': -15,
            'azimuth': 180,
        }

    # Load environment with camera tracking configuration
    env = DummyVecEnv([lambda: HumanoidWalkEnv(
        render_mode=render_mode,
        xml_file="./assets/humanoid_180_75.xml",
        training_phase="walking",  # At eval always walking
        default_camera_config=camera_config if track_camera else None,
        camera_name="track" if track_camera else None
    )])

    # Optionally load VecNormalize stats
    if vecnormalize_path and os.path.exists(vecnormalize_path):
        print(f"Loading VecNormalize stats from {vecnormalize_path}")
        env = VecNormalize.load(vecnormalize_path, env)
        env.training = False
        env.norm_reward = False
    else:
        print("⚠️ No VecNormalize stats found. Using raw env.")

    # --- Wrap with video recorder if requested
    if record_dir:
        os.makedirs(record_dir, exist_ok=True)
        env = VecVideoRecorder(
            env,
            video_folder=record_dir,
            record_video_trigger=lambda step: step == 0,  # first episode
            video_length=video_length,
            name_prefix="humanoid_eval"
        )
        if track_camera:
            print(f"Recording evaluation video(s) with camera tracking to {record_dir}")
        else:
            print(f"Recording evaluation video(s) to {record_dir}")

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {model_path} on device {device}")
    model = TQC.load(model_path, env=env, device=device)

    all_rewards = []

    for ep in range(episodes):
        obs = env.reset()
        ep_reward = 0.0
        state = None

        while True:
            action, _ = model.predict(obs, deterministic=True, state=state)
            obs, reward, done, info = env.step(action)

            # Unwrap vectorized outputs (n_envs=1)
            r = float(reward[0]) if isinstance(reward, np.ndarray) else float(reward)
            d = bool(done[0]) if isinstance(done, np.ndarray) else bool(done)

            ep_reward += r

            if render:
                env.render()

            if d:
                break

        print(f"Episode {ep+1}: reward={ep_reward:.2f}")
        all_rewards.append(ep_reward)

    env.close()
    print(f"\nMean reward over {episodes} episodes: {np.mean(all_rewards):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to humanoid model .zip file")
    parser.add_argument("--vecnorm-path", type=str, default=None, help="Optional path to vecnormalize stats .pkl file")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--no-render", action="store_true", help="Disable real-time rendering window")
    parser.add_argument("--record-dir", help="Folder to save evaluation videos")
    parser.add_argument("--video-length", type=int, default=10000, help="Max length of recorded video in steps")
    parser.add_argument("--no-track-camera", action="store_true", help="Disable camera tracking during video recording")

    args = parser.parse_args()

    evaluate(
        model_path=args.model_path,
        vecnormalize_path=args.vecnorm_path,
        episodes=args.episodes,
        render=not args.no_render,
        record_dir=args.record_dir,
        video_length=args.video_length,
        track_camera=not args.no_track_camera
    )