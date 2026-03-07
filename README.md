# 🤖 Humanoid Robot Walking — Reinforcement Learning

> ⚠️ **This project is currently in progress and not yet complete.**

A master's thesis project focused on training a humanoid robot to walk naturally using deep reinforcement learning in a MuJoCo physics simulation. The goal is to develop a model that produces human-like bipedal locomotion — going beyond basic forward movement to achieve natural gait patterns with proper stride length, arm swing coordination, and upright posture.

---

## 🎯 Project Goal

Train a humanoid robot in MuJoCo to:

- Walk forward at a target velocity of **~0.5 m/s**
- Produce **natural stride patterns** (65–80 cm step length)
- Maintain **proper upright posture** without leaning or collapsing
- Coordinate **arm swing** in opposition to leg movement
- Avoid exploitation behaviors like "flamingo stance," skating, or excessive leaning

---

## 🧠 Algorithm

This project uses **TQC (Truncated Quantile Critics)** from `stable-baselines3-contrib`, an off-policy actor-critic algorithm well suited for continuous control tasks requiring precise and stable learning.

---

## 📁 Project Structure

```
.
├── assets/                  # Static assets (textures, meshes, etc.)
├── completed_models/        # Saved model checkpoints that have finished training
├── models/                  # Active model checkpoints during training
├── tensorboard/             # TensorBoard logs for training metrics
├── video_simulations/       # Recorded simulation videos for evaluation
├── HumanoidWalkEnv.py       # Custom MuJoCo Gym environment
├── train_tqc.py             # Main training script using TQC
├── eval.py                  # Model evaluation script
├── view_model.py            # Visual playback of trained models
├── flowchart.txt            # Reward/curriculum logic flowchart notes
├── requirements.txt         # Python dependencies
└── temp.py                  # Temporary/scratch script
```

---

## 🏋️ Environment — `HumanoidWalkEnv`

A custom `gymnasium`-compatible environment wrapping MuJoCo's humanoid model. Key features:

- **Curriculum learning** progressing through: balance → standing → walking phases
- **19+ reward components** targeting natural gait, velocity, posture, and joint behavior
- **Gait cycle detection** to measure stride length, step alternation, and airborne time
- **Anti-exploitation mechanisms** to prevent reward hacking behaviors
- **Joint constraints** for physiologically plausible movement
- Parallel environment support (24–32 envs) for GPU-accelerated training

---

## 🏆 Reward Design Philosophy

Reward engineering is the core challenge of this project. The guiding principles are:

1. **Forward progress is mandatory** — highest weight, always active
2. **Step alternation is essential** — rewarded as a primary signal
3. **Gait quality is secondary** — rewarded only when forward movement is present
4. **All contextual rewards are conditional** on forward motion to prevent static exploitation
5. **Symmetry is critical** — asymmetric penalties cause persistent directional bias

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (developed on RTX 5070 Ti)
- MuJoCo (via `mujoco` Python package)

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd <repo-folder>

# Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
python train_tqc.py
```

Training metrics are logged to `tensorboard/`. To view:

```bash
tensorboard --logdir tensorboard/
```

### Evaluation

```bash
python eval.py
```

### Visual Playback

```bash
python view_model.py
```

---

## 📊 Metrics Tracked

Training is monitored across 70+ metrics including:

- Episode length and reward
- Walking velocity (target: 0.5 m/s)
- Stride length and step alternation rate
- Airborne time percentage
- Curriculum phase progression
- Per-joint torque and position errors

---

## 🛠️ Hardware

| Component | Spec |
|-----------|------|
| CPU | Intel Core i5-12400 |
| RAM | 32 GB DDR4 3200 MHz |
| GPU | NVIDIA RTX 5070 Ti |
| OS | Windows (with WSL2 / Linux env) |

---

## 📌 Status

| Feature | Status |
|---------|--------|
| Custom MuJoCo environment | ✅ Complete |
| TQC training pipeline | ✅ Complete |
| Curriculum learning system | ✅ Complete |
| Gait cycle detection | ✅ Complete |
| Natural walking gait | 🔄 In Progress |
| Target velocity (0.5 m/s) | 🔄 In Progress |
| Thesis write-up | 🔄 In Progress |

---

## 📝 Notes

- This is a **clean rewrite branch** of the project. Previous experimental iterations (V17–V26) have been used to inform the current architecture and reward design.
- The `completed_models/` directory contains the best-performing checkpoints from prior runs.
- Video recordings in `video_simulations/` are captured from multiple camera angles (side, front, rear, isometric, top-down).

---

## 📄 License

This project is part of an academic master's thesis. All rights reserved.