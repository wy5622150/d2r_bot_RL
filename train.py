# train.py
# 训练管线：BC 预训练 + PPO 强化学习
# 动作空间: MultiDiscrete([144, 4, N]) — N 从 keymap.json 动态加载
#
# 用法:
#   python train.py --phase bc          # 只做 BC 预训练
#   python train.py --phase ppo         # 只做 PPO（加载 BC 权重）
#   python train.py --phase all         # BC + PPO
#   python train.py --phase ppo --no-bc # PPO 从头训练
#
# 前置: python build_keymap.py （生成 models/keymap.json）

import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym

from d2r_warlock_env import D2RWarlockEnv, N_GRID, N_MOUSE_ACTIONS, load_keymap

# ====================== 配置 ======================
DATA_DIR = "data"
MODEL_DIR = "models"
KEYMAP_PATH = os.path.join(MODEL_DIR, "keymap.json")
BC_MODEL_PATH = os.path.join(MODEL_DIR, "bc_pretrained.pth")
PPO_MODEL_PATH = os.path.join(MODEL_DIR, "ppo_final")
PPO_CHECKPOINT_DIR = os.path.join(MODEL_DIR, "ppo_checkpoints")

BC_EPOCHS = 30
BC_BATCH_SIZE = 128
BC_LR = 1e-4

PPO_TOTAL_TIMESTEPS = 50_000_000
PPO_N_ENVS = 4
PPO_N_STEPS = 2048
PPO_BATCH_SIZE = 256
PPO_N_EPOCHS = 4
PPO_LR = 3e-4
PPO_GAMMA = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_CLIP_RANGE = 0.2
PPO_ENT_COEF = 0.01
PPO_VF_COEF = 0.5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ===================================================

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PPO_CHECKPOINT_DIR, exist_ok=True)


def _load_keymap_config():
    """加载 keymap.json 配置"""
    with open(KEYMAP_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
#  自定义 CNN 特征提取器
# ============================================================
class D2RCNN(BaseFeaturesExtractor):
    """NatureCNN: (90, 160, 4) → 512 维特征"""

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)

        # observation_space.shape = (H, W, C) = (90, 160, 4)
        obs_h, obs_w, n_channels = observation_space.shape

        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # 动态计算 flatten 维度（适应任意 H×W）
        with torch.no_grad():
            sample = torch.zeros(1, n_channels, obs_h, obs_w)
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # SB3 传入 (B, H, W, C) → (B, C, H, W)
        x = observations.permute(0, 3, 1, 2).float() / 255.0
        return self.linear(self.cnn(x))


# ============================================================
#  Phase 1: Behavioral Cloning
# ============================================================
class DemoDataset(Dataset):
    """
    加载 data/*.npz
    录制数据中 act_key 是字符串数组，需要用 keymap 转为 int
    """

    def __init__(self, data_dir, keymap):
        """keymap: {"": 0, "w": 1, "e": 2, ...}"""
        self.observations = []
        self.act_grid = []
        self.act_mouse = []
        self.act_key = []

        files = sorted(glob.glob(os.path.join(data_dir, "ep_*.npz")))
        if not files:
            raise FileNotFoundError(f"在 {data_dir} 中找不到 ep_*.npz 文件")

        print(f"加载 {len(files)} 个演示文件...")
        for fpath in files:
            data = np.load(fpath, allow_pickle=True)
            self.observations.append(data["obs"])
            self.act_grid.append(data["act_grid"])
            self.act_mouse.append(data["act_mouse"])

            # 字符串按键名 → 整数索引
            raw_keys = data["act_key"]
            key_indices = np.array(
                [keymap.get(str(k), 0) for k in raw_keys],
                dtype=np.int64,
            )
            self.act_key.append(key_indices)

        self.observations = np.concatenate(self.observations, axis=0)
        self.act_grid = np.concatenate(self.act_grid, axis=0)
        self.act_mouse = np.concatenate(self.act_mouse, axis=0)
        self.act_key = np.concatenate(self.act_key, axis=0)

        n = len(self.observations)
        n_key_classes = max(keymap.values()) + 1
        print(f"总帧数: {n}")
        print(f"鼠标位置: min={self.act_grid.min()} max={self.act_grid.max()}")
        print(f"鼠标动作: {np.bincount(self.act_mouse, minlength=N_MOUSE_ACTIONS)}")
        print(f"键盘动作 ({n_key_classes} 类): {np.bincount(self.act_key, minlength=n_key_classes)}")

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        obs = torch.from_numpy(self.observations[idx]).float()
        obs = obs.permute(2, 0, 1) / 255.0
        ag = torch.tensor(self.act_grid[idx], dtype=torch.long)
        am = torch.tensor(self.act_mouse[idx], dtype=torch.long)
        ak = torch.tensor(self.act_key[idx], dtype=torch.long)
        return obs, ag, am, ak


class BCModel(nn.Module):
    """
    BC 模型：共享 CNN + 三个分类头
    key_head 大小从 keymap 动态决定
    """

    def __init__(self, n_key_actions):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            # 输入: (B, C=4, H=90, W=160)
            n_flatten = self.cnn(torch.zeros(1, 4, 90, 160)).shape[1]

        self.feature = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
        )

        self.grid_head = nn.Linear(512, N_GRID)            # 144
        self.mouse_head = nn.Linear(512, N_MOUSE_ACTIONS)   # 4
        self.key_head = nn.Linear(512, n_key_actions)        # 动态

    def forward(self, x):
        feat = self.feature(self.cnn(x))
        return self.grid_head(feat), self.mouse_head(feat), self.key_head(feat)


def train_bc():
    """Phase 1: Behavioral Cloning"""
    config = _load_keymap_config()
    keymap = config["keymap"]
    n_key_actions = config["n_key_actions"]

    print("\n" + "=" * 60)
    print("Phase 1: Behavioral Cloning 预训练")
    print(f"动作空间: grid={N_GRID} mouse={N_MOUSE_ACTIONS} key={n_key_actions}")
    print(f"按键映射: {keymap}")
    print("=" * 60)

    dataset = DemoDataset(DATA_DIR, keymap)
    dataloader = DataLoader(
        dataset,
        batch_size=BC_BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    model = BCModel(n_key_actions=n_key_actions).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=BC_LR)

    def make_weighted_ce(labels, n_classes):
        counts = np.bincount(labels, minlength=n_classes).astype(np.float32)
        w = 1.0 / (counts + 1.0)
        w = w / w.sum() * n_classes
        return nn.CrossEntropyLoss(weight=torch.from_numpy(w).to(DEVICE))

    ce_grid = make_weighted_ce(dataset.act_grid, N_GRID)
    ce_mouse = make_weighted_ce(dataset.act_mouse, N_MOUSE_ACTIONS)
    ce_key = make_weighted_ce(dataset.act_key, n_key_actions)

    best_loss = float('inf')

    for epoch in range(BC_EPOCHS):
        model.train()
        total_loss = 0.0
        correct_grid = correct_mouse = correct_key = 0
        total = 0

        for obs, ag, am, ak in dataloader:
            obs = obs.to(DEVICE)
            ag, am, ak = ag.to(DEVICE), am.to(DEVICE), ak.to(DEVICE)

            logits_g, logits_m, logits_k = model(obs)

            loss = 2.0 * ce_grid(logits_g, ag) + ce_mouse(logits_m, am) + ce_key(logits_k, ak)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = obs.size(0)
            total_loss += loss.item() * bs
            correct_grid += (logits_g.argmax(1) == ag).sum().item()
            correct_mouse += (logits_m.argmax(1) == am).sum().item()
            correct_key += (logits_k.argmax(1) == ak).sum().item()
            total += bs

        avg_loss = total_loss / total
        print(f"Epoch {epoch+1}/{BC_EPOCHS}  Loss: {avg_loss:.4f}  "
              f"Acc grid:{correct_grid/total:.3f} mouse:{correct_mouse/total:.3f} key:{correct_key/total:.3f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), BC_MODEL_PATH)
            print(f"  -> 保存 (loss={best_loss:.4f})")

    print(f"\nBC 完成, 最佳 loss: {best_loss:.4f}")


def load_bc_weights_into_ppo(ppo_model, bc_model_path):
    bc_state = torch.load(bc_model_path, map_location=DEVICE, weights_only=True)
    ppo_fe_state = ppo_model.policy.features_extractor.state_dict()

    matched = 0
    for bc_key, bc_val in bc_state.items():
        if bc_key.startswith("cnn."):
            if bc_key in ppo_fe_state and ppo_fe_state[bc_key].shape == bc_val.shape:
                ppo_fe_state[bc_key] = bc_val
                matched += 1
        elif bc_key.startswith("feature."):
            ppo_key = bc_key.replace("feature.", "linear.")
            if ppo_key in ppo_fe_state and ppo_fe_state[ppo_key].shape == bc_val.shape:
                ppo_fe_state[ppo_key] = bc_val
                matched += 1

    ppo_model.policy.features_extractor.load_state_dict(ppo_fe_state)
    print(f"已加载 {matched} 个 BC 权重到 PPO features_extractor")


# ============================================================
#  Phase 2: PPO
# ============================================================
def make_env(rank, seed=0):
    def _init():
        env = D2RWarlockEnv(mode='train')
        env.reset(seed=seed + rank)
        return env
    return _init


def train_ppo(use_bc_pretrain=True):
    config = _load_keymap_config()
    n_key_actions = config["n_key_actions"]

    print("\n" + "=" * 60)
    print("Phase 2: PPO 在线强化学习")
    print(f"动作空间: MultiDiscrete([{N_GRID}, {N_MOUSE_ACTIONS}, {n_key_actions}])")
    print(f"并行: {PPO_N_ENVS} 环境 | 总步数: {PPO_TOTAL_TIMESTEPS:,} | {DEVICE}")
    print("=" * 60)

    env = SubprocVecEnv([make_env(i) for i in range(PPO_N_ENVS)])
    env = VecMonitor(env)

    policy_kwargs = {
        "features_extractor_class": D2RCNN,
        "features_extractor_kwargs": {"features_dim": 512},
        "net_arch": dict(pi=[256], vf=[256]),
    }

    model = PPO(
        policy="CnnPolicy",
        env=env,
        n_steps=PPO_N_STEPS,
        batch_size=PPO_BATCH_SIZE,
        n_epochs=PPO_N_EPOCHS,
        learning_rate=PPO_LR,
        gamma=PPO_GAMMA,
        gae_lambda=PPO_GAE_LAMBDA,
        clip_range=PPO_CLIP_RANGE,
        ent_coef=PPO_ENT_COEF,
        vf_coef=PPO_VF_COEF,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=os.path.join(MODEL_DIR, "tb_logs"),
        device=DEVICE,
        policy_kwargs=policy_kwargs,
    )

    if use_bc_pretrain and os.path.exists(BC_MODEL_PATH):
        load_bc_weights_into_ppo(model, BC_MODEL_PATH)
    elif use_bc_pretrain:
        print(f"[WARN] {BC_MODEL_PATH} 不存在，从头训练")

    checkpoint_cb = CheckpointCallback(
        save_freq=100_000 // PPO_N_ENVS,
        save_path=PPO_CHECKPOINT_DIR,
        name_prefix="ppo_warlock",
    )

    try:
        model.learn(
            total_timesteps=PPO_TOTAL_TIMESTEPS,
            callback=CallbackList([checkpoint_cb]),
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n训练中断，保存模型...")

    model.save(PPO_MODEL_PATH)
    print(f"\nPPO 已保存: {PPO_MODEL_PATH}")
    env.close()


# ============================================================
def main():
    parser = argparse.ArgumentParser(description="D2R Warlock Trav RL 训练")
    parser.add_argument("--phase", default="all", choices=["bc", "ppo", "all"])
    parser.add_argument("--no-bc", action="store_true")
    args = parser.parse_args()

    if args.phase in ("bc", "all"):
        train_bc()
    if args.phase in ("ppo", "all"):
        train_ppo(use_bc_pretrain=not args.no_bc)


if __name__ == "__main__":
    main()
