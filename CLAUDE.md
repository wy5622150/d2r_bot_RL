# D2R Warlock Travincal RL Bot

## Project Overview
Reinforcement learning bot for Diablo 2 Resurrected. Uses pure pixel input (screen capture) and simulated mouse/keyboard to learn the full Travincal farming loop: main menu → create game → run to waypoint → teleport → kill council → loot → leave game → repeat.

## Architecture

### Action Space: `MultiDiscrete([144, 4, N_keys])`
- **[0] Mouse position**: 16×9 grid = 144 cells over game window
- **[1] Mouse action**: 0=none, 1=left_hold(walk), 2=left_click(pickup/menu), 3=right_click(skill)
- **[2] Keyboard**: Dynamically discovered from recorded data via `keymap.json`

### Observation: `(90, 160, 4)` uint8
- 160×90 grayscale (16:9 aspect ratio preserved), 4-frame stack
- Resized from 4K recording / 720p training resolution

### Training Pipeline
1. **Record** (Windows, 4K): `collect_data.py` → `data/ep_XXXX.npz`
2. **Build keymap**: `build_keymap.py` → `models/keymap.json`
3. **BC pretrain**: `train.py --phase bc` → `models/bc_pretrained.pth`
4. **PPO train**: `train.py --phase ppo` → `models/ppo_final`
5. **Deploy** (Windows, 720p): `run_bot.py`

## File Descriptions
- `collect_data.py` — Records human demos on Windows (F10=start, F11=stop). Captures mouse position/clicks + all keyboard keys as raw strings. Saves as `.npz`.
- `build_keymap.py` — Scans all `data/ep_*.npz`, discovers unique keys, generates `models/keymap.json` with key→index mapping.
- `d2r_warlock_env.py` — Gymnasium environment. Loads `keymap.json` for dynamic keyboard action space. Template matching for rewards. Screen capture via `mss`, input via `pyautogui`.
- `train.py` — BC pretraining (3-head CNN: grid/mouse/key) + PPO with custom NatureCNN feature extractor. BC weights transfer to PPO's feature extractor.
- `run_bot.py` — Loads trained PPO model and runs inference loop.
- `calibrate.py` — Screenshot + grid overlay + 160×90 preview for calibration.

## Current Status
- **Recording**: 15 episodes recorded on Windows (16,661 frames, ~18.5 min)
- **Data location**: `data/` directory (not in git, must be copied manually)
- **Keymap**: Generated at `models/keymap.json` (12 keys discovered: ctrl_l, space, w, e, q, esc, alt_l, 2, 4, 3, i)
- **Next step**: Install PyTorch ROCm + dependencies on Ubuntu, run BC pretraining, then PPO

## Hardware
- **GPU**: AMD Radeon RX 7900 XTX (24GB VRAM)
- **Recording**: Windows 11, 4K display
- **Training**: Ubuntu (ROCm for AMD GPU)

## Ubuntu Setup Instructions
```bash
# 1. Clone repo
git clone git@github.com:wy5622150/d2r_bot_RL.git
cd d2r_bot_RL

# 2. Copy data from Windows (USB drive, scp, etc.)
# Copy the entire data/ folder and models/keymap.json into the repo

# 3. Install ROCm PyTorch (for 7900 XTX)
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2

# 4. Install other dependencies
pip install stable-baselines3 gymnasium numpy opencv-python tensorboard tqdm

# 5. Verify GPU
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# 6. Run BC pretraining
python train.py --phase bc

# 7. Run PPO training (long, GPU-accelerated)
python train.py --phase ppo
```

## Key Design Decisions
- **No memory reading**: Pure pixel input, no game memory hacking
- **Dynamic keyboard mapping**: Recording captures raw key names, `build_keymap.py` auto-discovers used keys
- **Left mouse hold state**: Correctly tracks continuous hold for walking (not just click events)
- **Full game loop**: Model learns entire cycle including menu navigation, not just combat
- **Template matching rewards**: Uses grayscale template images in `templates/` directory for reward signals (not yet created — needed for PPO phase)
