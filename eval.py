import os
import re
import torch
import argparse
import numpy as np

from sb3_contrib import TQC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from HumanoidWalkEnv import HumanoidWalkEnv

# ============================================================
#  EASY CONFIGURATION — Change these before running
# ============================================================
VERSION         = 18                                           # Model version number (XX)
MODEL_FILE      = "tqc_humanoid_walker_final_20000016_steps"  # .zip added automatically by TQC.load()
VECNORM_FILE    = "tqc_vecnormalize_final.pkl"
VIDEO_TIMESTEPS = 10000                                         # Total steps to record (also caps eval length when recording)
# ============================================================

# Derived defaults (don't need to change these)
_MODEL_PATH   = os.path.join("models", f"tqc_{VERSION:02d}", MODEL_FILE)
_VECNORM_PATH = os.path.join("models", f"tqc_{VERSION:02d}", VECNORM_FILE)
_RECORD_DIR   = os.path.join("video_simulations", f"TQC{VERSION:02d}")


def parse_model_timesteps(model_file: str) -> str:
    """Extract timestep count from filename -> short form e.g. '20M', '5M'."""
    match = re.search(r'(\d{6,})', model_file)
    if match:
        steps = int(match.group(1))
        if steps % 1_000_000 == 0:
            return f"{steps // 1_000_000}M"
        elif steps % 1_000 == 0:
            return f"{steps // 1_000}K"
        return str(steps)
    return "unknown"


def next_video_index(record_dir: str, base_name: str) -> int:
    """Return the next available counter so existing videos are never overwritten."""
    if not os.path.exists(record_dir):
        return 1
    indices = []
    for f in os.listdir(record_dir):
        m = re.search(re.escape(base_name) + r'_(\d+)', f)
        if m:
            indices.append(int(m.group(1)))
    return max(indices, default=0) + 1


def save_video(frames: list, path: str, fps: int = 100):
    """Save RGB frames to an mp4 file using imageio."""
    try:
        import imageio
    except ImportError:
        raise ImportError("Run: pip install imageio[ffmpeg]")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    imageio.mimwrite(path, frames, fps=fps, codec="libx264", quality=8)
    print(f"  Saved: {path}")


def evaluate(model_path, vecnormalize_path=None, episodes=5, render=True,
             record_dir=None, video_length=3000, track_camera=True,
             track_camera_name="track", switch_cameras=False):
    """
    Evaluate a trained humanoid model.

    When recording, video_length acts as a hard cap on total steps across all
    episodes — evaluation stops as soon as that many steps have been collected,
    even if the humanoid hasn't fallen.

    Args:
        model_path:        Path to trained .zip model.
        vecnormalize_path: Path to saved VecNormalize stats (if any).
        episodes:          Max number of episodes to run.
        render:            If True, render in real-time (ignored when recording).
        record_dir:        Directory to save video (None = no recording).
        video_length:      Hard step cap when recording.
        track_camera:      Use the XML tracking camera during recording.
        track_camera_name: Camera name to use (ignored when switch_cameras=True).
        switch_cameras:    If True, cycle through track → track_side → track_front
                           in equal thirds of video_length. Recording only.
    """

    # rgb_array mode is required to capture frames manually.
    # We do NOT use VecVideoRecorder — it ignores camera_name and breaks tracking.
    if record_dir:
        render_mode = "rgb_array"
    elif render:
        render_mode = "human"
    else:
        render_mode = None

    # camera_name picks one of the XML-defined trackcom cameras.
    # "back" = behind humanoid, "side" = side view. Both follow automatically via trackcom.
    # We store a direct reference to the base env to call render() without going through
    # DummyVecEnv/VecNormalize which drop camera_name in their render chain.
    _base_env = []
    def _make_env():
        e = HumanoidWalkEnv(
            render_mode=render_mode,
            xml_file="./assets/humanoid_180_75.xml",
            training_phase="walking",
            camera_name="back" if track_camera else None,   # "back" or "side" — both use trackcom
        )
        _base_env.append(e)
        return e

    env = DummyVecEnv([_make_env])

    # Load VecNormalize stats
    if vecnormalize_path and os.path.exists(vecnormalize_path):
        print(f"  VecNorm  : {vecnormalize_path}")
        env = VecNormalize.load(vecnormalize_path, env)
        env.training    = False
        env.norm_reward = False
    else:
        print("  No VecNormalize stats found — using raw env.")

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device   : {device}")
    model = TQC.load(model_path, env=env, device=device)

    # Prepare video output path
    video_path = None
    if record_dir:
        model_ts  = parse_model_timesteps(MODEL_FILE)
        base_name = f"TQC{VERSION:02d}_{model_ts}_{video_length}"
        counter   = next_video_index(record_dir, base_name)
        video_path = os.path.join(record_dir, f"{base_name}_{counter}.mp4")
        print(f"  Recording: {video_path}  (stops after {video_length} steps)")

    all_rewards = []
    frames      = []
    total_steps = 0

    for ep in range(episodes):
        # Don't start a new episode if we've already hit the step cap
        if record_dir and total_steps >= video_length:
            break

        obs       = env.reset()
        ep_reward = 0.0
        state     = None

        while True:
            action, _ = model.predict(obs, deterministic=True, state=state)
            obs, reward, done, info = env.step(action)

            r = float(reward[0]) if isinstance(reward, np.ndarray) else float(reward)
            d = bool(done[0])    if isinstance(done,   np.ndarray) else bool(done)
            ep_reward += r
            total_steps += 1

            if record_dir:
                # Set camera_id directly on the renderer before calling render().
                # This is the correct API — passing camera_id as a kwarg to render()
                # is not supported in newer Gymnasium versions.
                base = _base_env[0]
                if track_camera:
                    try:
                        if switch_cameras:
                            # Divide video_length into 3 equal thirds
                            # 0–33%: track (back), 33–66%: track_side, 66–100%: track_front
                            third = video_length / 3
                            if total_steps < third:
                                cam_name = "track"
                            elif total_steps < 2 * third:
                                cam_name = "track_side"
                            else:
                                cam_name = "track_front"
                        else:
                            cam_name = track_camera_name
                        base.mujoco_renderer.camera_id = base.model.camera(cam_name).id
                    except Exception:
                        base.mujoco_renderer.camera_id = -1

                frame = base.render()
                if frame is not None:
                    if isinstance(frame, np.ndarray) and frame.ndim == 4:
                        frame = frame[0]
                    frames.append(frame)
            elif render:
                _base_env[0].render()

            # Hard stop at step cap (humanoid didn't fall — we cut it here)
            if record_dir and total_steps >= video_length:
                break

            if d:
                break

        print(f"  Episode {ep + 1}: reward = {ep_reward:.2f}  (steps so far: {total_steps})")
        all_rewards.append(ep_reward)

    env.close()
    print(f"\n  Mean reward: {np.mean(all_rewards):.2f}  ({len(all_rewards)} episode(s))")

    if frames and video_path:
        print(f"  Saving {len(frames)} frames ...")
        save_video(frames, video_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path",      type=str, default=None, help="Path to model .zip (overrides VERSION variable)")
    parser.add_argument("--vecnorm-path",    type=str, default=None, help="Path to vecnorm .pkl (overrides VERSION variable)")
    parser.add_argument("--record",          action="store_true",    help="Record video (saved to video_simulations/TQCXX/)")
    parser.add_argument("--record-dir",      type=str, default=None, help="Custom save folder (implies --record)")
    parser.add_argument("--episodes",        type=int, default=5,    help="Max episodes to run")
    parser.add_argument("--video-length",    type=int, default=None, help="Hard step cap when recording (overrides VIDEO_TIMESTEPS)")
    parser.add_argument("--no-render",       action="store_true",    help="Disable live render window")
    parser.add_argument("--no-track-camera", action="store_true",    help="Disable camera tracking")
    parser.add_argument("--camera",          type=str, default="track",
                        choices=["track", "track_side", "track_front", "back", "side", "egocentric"],
                        help="Camera to use when recording (default: track). Ignored when --switch is set.")
    parser.add_argument("--switch",          action="store_true",
                        help="Cycle cameras in thirds: track → track_side → track_front. Recording only.")
    args = parser.parse_args()

    # Parser takes priority; fall back to top-of-file variables
    model_path   = args.model_path   or _MODEL_PATH
    vecnorm_path = args.vecnorm_path or _VECNORM_PATH
    video_length = args.video_length or VIDEO_TIMESTEPS

    if args.record_dir:
        record_dir = args.record_dir
    elif args.record:
        record_dir = _RECORD_DIR
    else:
        record_dir = None

    print(f"\n{'='*55}")
    print(f"  TQC Evaluation — Version {VERSION}")
    print(f"{'='*55}")
    print(f"  Model    : {model_path}")
    print(f"  VecNorm  : {vecnorm_path}")
    print(f"  Video dir: {record_dir or 'disabled'}")
    print(f"  Camera   : {'switch (track → track_side → track_front)' if args.switch else args.camera}")
    print(f"  Steps    : {video_length}")
    print(f"{'='*55}\n")

    evaluate(
        model_path=model_path,
        vecnormalize_path=vecnorm_path,
        episodes=args.episodes,
        render=not args.no_render,
        record_dir=record_dir,
        video_length=video_length,
        track_camera=not args.no_track_camera,
        track_camera_name=args.camera,
        switch_cameras=args.switch,
    )