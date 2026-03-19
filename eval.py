import os
import re
import csv
import math
import torch
import argparse
import numpy as np

from sb3_contrib import TQC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from HumanoidWalkEnv import HumanoidWalkEnv

# ============================================================
#  EASY CONFIGURATION — Change these before running
# ============================================================
VERSION         = 24                                           # Model version number (XX)
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
             track_camera_name="track", switch_cameras=False, log_csv=False):
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

    # ── CSV logging setup ────────────────────────────────────────────────────
    csv_file   = None
    csv_writer = None
    CSV_COLUMNS = [
        'episode', 'step', 'sim_time',
        'pos_x', 'pos_y', 'pos_z', 'forward_vel',
        'torso_quat_w', 'torso_roll_deg', 'torso_pitch_deg',
        'hip_y_right_deg', 'hip_y_left_deg',
        'knee_right_deg',  'knee_left_deg',
        'ankle_y_right_deg', 'ankle_y_left_deg',
        'ankle_x_right_deg', 'ankle_x_left_deg',
        'heel_right', 'heel_left',
        'toe_right',  'toe_left',
        'left_contact', 'right_contact',
        'single_support', 'both_contact', 'no_contact',
        'episode_reward',
    ]
    if log_csv:
        if video_path:
            # Same name/folder as the video — just swap .mp4 for .csv
            log_path = video_path.replace('.mp4', '.csv')
        else:
            # No video — save to logs/TQCXX/ with same naming convention
            model_ts = parse_model_timesteps(MODEL_FILE)
            base_name = f"TQC{VERSION:02d}_{model_ts}_{video_length}"
            log_dir   = os.path.join("video_simulations", f"TQC{VERSION:02d}")
            os.makedirs(log_dir, exist_ok=True)
            counter   = next_video_index(log_dir, base_name)
            log_path  = os.path.join(log_dir, f"{base_name}_{counter}.csv")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        csv_file   = open(log_path, 'w', newline='')
        csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        csv_writer.writeheader()
        print(f"  CSV log  : {log_path}")

    all_rewards = []
    frames      = []
    total_steps = 0

    # Sensor + body IDs — resolved once after first env creation
    _sensor_ids_resolved = [False]
    _heel_r_adr = [0]; _heel_l_adr = [0]
    _toe_r_adr  = [0]; _toe_l_adr  = [0]
    _torso_id   = [0]
    _prev_pos_x = [0.0]

    def _resolve_sensor_ids():
        if _sensor_ids_resolved[0] or not _base_env:
            return
        import mujoco
        m = _base_env[0].model
        def _adr(name):
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, name)
            return int(m.sensor_adr[sid])
        _heel_r_adr[0] = _adr('foot_right_touch')
        _heel_l_adr[0] = _adr('foot_left_touch')
        _toe_r_adr[0]  = _adr('toe_right_touch')
        _toe_l_adr[0]  = _adr('toe_left_touch')
        _torso_id[0]   = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'torax'))
        _sensor_ids_resolved[0] = True

    for ep in range(episodes):
        # Don't start a new episode if we've already hit the step cap
        if record_dir and total_steps >= video_length:
            break

        obs       = env.reset()
        ep_reward = 0.0
        state     = None
        ep_step   = 0
        _resolve_sensor_ids()
        _prev_pos_x[0] = float(_base_env[0].data.qpos[0]) if _base_env else 0.0

        while True:
            action, _ = model.predict(obs, deterministic=True, state=state)
            obs, reward, done, info = env.step(action)

            r = float(reward[0]) if isinstance(reward, np.ndarray) else float(reward)
            d = bool(done[0])    if isinstance(done,   np.ndarray) else bool(done)
            ep_reward += r
            total_steps += 1
            ep_step     += 1

            # ── Per-step CSV logging ──────────────────────────────────────────
            if log_csv and csv_writer and _base_env:
                base  = _base_env[0]
                mdata = base.data
                qpos  = mdata.qpos

                # Position and velocity
                pos_x = float(qpos[0])
                pos_y = float(qpos[1])
                pos_z = float(qpos[2])
                fwd_vel = (pos_x - _prev_pos_x[0]) / base.dt
                _prev_pos_x[0] = pos_x

                # Torso orientation from quaternion (xquat[torso_id])
                tq = mdata.xquat[_torso_id[0]]
                qw, qx, qy, qz = float(tq[0]), float(tq[1]), float(tq[2]), float(tq[3])
                roll_rad  = math.atan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy))
                pitch_rad = math.asin(max(-1.0, min(1.0, 2*(qw*qy - qz*qx))))

                # Joint angles — convert rad to degrees
                def deg(idx): return float(qpos[idx]) * 180.0 / math.pi

                # Sensor readings
                sd = mdata.sensordata
                heel_r = float(sd[_heel_r_adr[0]])
                heel_l = float(sd[_heel_l_adr[0]])
                toe_r  = float(sd[_toe_r_adr[0]])
                toe_l  = float(sd[_toe_l_adr[0]])

                # Geom contacts
                lc = base._check_foot_contact_mujoco(base.left_foot_geoms)
                rc = base._check_foot_contact_mujoco(base.right_foot_geoms)

                csv_writer.writerow({
                    'episode':           ep + 1,
                    'step':              ep_step,
                    'sim_time':          round(float(mdata.time), 4),
                    'pos_x':             round(pos_x, 4),
                    'pos_y':             round(pos_y, 4),
                    'pos_z':             round(pos_z, 4),
                    'forward_vel':       round(fwd_vel, 4),
                    'torso_quat_w':      round(qw, 4),
                    'torso_roll_deg':    round(roll_rad  * 180.0 / math.pi, 2),
                    'torso_pitch_deg':   round(pitch_rad * 180.0 / math.pi, 2),
                    'hip_y_right_deg':   round(deg(12), 2),
                    'hip_y_left_deg':    round(deg(18), 2),
                    'knee_right_deg':    round(deg(13), 2),
                    'knee_left_deg':     round(deg(19), 2),
                    'ankle_y_right_deg': round(deg(14), 2),
                    'ankle_y_left_deg':  round(deg(20), 2),
                    'ankle_x_right_deg': round(deg(15), 2),
                    'ankle_x_left_deg':  round(deg(21), 2),
                    'heel_right':        round(heel_r, 3),
                    'heel_left':         round(heel_l, 3),
                    'toe_right':         round(toe_r,  3),
                    'toe_left':          round(toe_l,  3),
                    'left_contact':      int(lc),
                    'right_contact':     int(rc),
                    'single_support':    int(lc != rc),
                    'both_contact':      int(lc and rc),
                    'no_contact':        int(not lc and not rc),
                    'episode_reward':    round(ep_reward, 2),
                })

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
    if csv_file:
        csv_file.close()
        print(f"  CSV saved.")
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
    parser.add_argument("--log",             action="store_true",    help="Save per-step CSV log (saved alongside video or to logs/TQCXX/)")
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
        log_csv=args.log,
    )