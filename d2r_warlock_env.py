# d2r_warlock_env.py
# Gymnasium 环境：D2R Warlock Travincal
# 动作空间：MultiDiscrete([144, 4, N]) — N 从 keymap.json 动态加载
# 覆盖完整循环：主菜单→建游戏→刷Trav→离开→回主菜单

import os
import json
import gymnasium as gym
import numpy as np
import cv2
import time
from collections import deque
import mss
import pyautogui
from gymnasium.spaces import Box, MultiDiscrete

pyautogui.PAUSE = 0.0
pyautogui.FAILSAFE = False

# ====================== 固定配置 ======================
GRID_W = 16
GRID_H = 9
N_GRID = GRID_W * GRID_H  # 144

MOUSE_NONE = 0
MOUSE_LEFT_HOLD = 1
MOUSE_LEFT_CLICK = 2
MOUSE_RIGHT_CLICK = 3
N_MOUSE_ACTIONS = 4

TEMPLATE_REWARDS = {
    'trav_entrance': 70.0,
    'council_dead': 100.0,
    'item_glow': 35.0,
    'main_menu': 150.0,
}

MONITOR_720P = {"top": 100, "left": 100, "width": 1280, "height": 720}
MONITOR_4K = {"top": 0, "left": 0, "width": 3840, "height": 2160}

OBS_W = 160
OBS_H = 90
FRAME_STACK = 4
MATCH_THRESHOLD = 0.82
STEP_PENALTY = -0.005
MAX_STEPS = 8000

KEYMAP_PATH = os.path.join("models", "keymap.json")


def load_keymap(path=KEYMAP_PATH):
    """加载按键映射配置，返回 (n_key_actions, idx_to_key_dict)"""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"找不到 {path}，请先运行: python build_keymap.py"
        )
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)

    n_key_actions = config["n_key_actions"]
    # idx_to_key: {0: "", 1: "w", 2: "e", ...}
    idx_to_key = {int(k): v for k, v in config["idx_to_key"].items()}
    return n_key_actions, idx_to_key


class D2RWarlockEnv(gym.Env):
    """
    D2R Warlock Travincal 完整循环 RL 环境

    观测: (90, 160, 4) uint8 灰度帧栈 (H, W, C)
    动作: MultiDiscrete([144, 4, N_keys])
        [0] 鼠标网格位置 (16x9)
        [1] 鼠标动作: 0=无, 1=左键按住, 2=左键单击, 3=右键
        [2] 键盘: 0=无, 1~N=从录制数据中自动发现的按键
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, mode='train', template_dir='templates',
                 keymap_path=KEYMAP_PATH, render_mode=None):
        super().__init__()

        # 加载动态按键映射
        self.n_key_actions, self.idx_to_key = load_keymap(keymap_path)

        self.observation_space = Box(
            low=0, high=255,
            shape=(OBS_H, OBS_W, FRAME_STACK),
            dtype=np.uint8,
        )
        self.action_space = MultiDiscrete([N_GRID, N_MOUSE_ACTIONS, self.n_key_actions])

        self.mode = mode
        self.render_mode = render_mode
        self.template_dir = template_dir

        self.monitor = MONITOR_720P if mode in ('train', 'run') else MONITOR_4K
        self.cell_w = self.monitor["width"] // GRID_W
        self.cell_h = self.monitor["height"] // GRID_H

        self.sct = None
        self.templates = {}
        self._load_templates()

        self.frame_stack = deque(maxlen=FRAME_STACK)
        self.step_count = 0
        self.episode_reward = 0.0
        self.milestones_hit = set()
        self._left_held = False

    def _ensure_sct(self):
        if self.sct is None:
            self.sct = mss.mss()

    def _load_templates(self):
        for name in TEMPLATE_REWARDS:
            path = os.path.join(self.template_dir, f"{name}.png")
            if os.path.exists(path):
                tpl = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if tpl is not None:
                    self.templates[name] = tpl

    def _grid_to_pixel(self, grid_idx):
        gy = grid_idx // GRID_W
        gx = grid_idx % GRID_W
        px = self.monitor["left"] + gx * self.cell_w + self.cell_w // 2
        py = self.monitor["top"] + gy * self.cell_h + self.cell_h // 2
        return px, py

    def _grab_gray(self):
        self._ensure_sct()
        img = np.array(self.sct.grab(self.monitor))
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        return cv2.resize(gray, (OBS_W, OBS_H))

    def _grab_full_gray(self):
        self._ensure_sct()
        img = np.array(self.sct.grab(self.monitor))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    def _get_stacked_obs(self):
        return np.stack(list(self.frame_stack), axis=-1)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0
        self.episode_reward = 0.0
        self.milestones_hit.clear()
        self.frame_stack.clear()
        self._left_held = False

        pyautogui.mouseUp(button='left')
        time.sleep(1.0)

        for _ in range(FRAME_STACK):
            self.frame_stack.append(self._grab_gray())
            time.sleep(0.05)

        return self._get_stacked_obs(), {}

    def step(self, action):
        grid_idx = int(action[0])
        mouse_act = int(action[1])
        key_act = int(action[2])

        # 1. 鼠标移动
        px, py = self._grid_to_pixel(grid_idx)
        pyautogui.moveTo(px, py, duration=0)

        # 2. 鼠标动作
        if mouse_act == MOUSE_NONE:
            if self._left_held:
                pyautogui.mouseUp(button='left')
                self._left_held = False
        elif mouse_act == MOUSE_LEFT_HOLD:
            if not self._left_held:
                pyautogui.mouseDown(button='left')
                self._left_held = True
        elif mouse_act == MOUSE_LEFT_CLICK:
            if self._left_held:
                pyautogui.mouseUp(button='left')
                self._left_held = False
            pyautogui.click(px, py, button='left')
        elif mouse_act == MOUSE_RIGHT_CLICK:
            if self._left_held:
                pyautogui.mouseUp(button='left')
                self._left_held = False
            pyautogui.click(px, py, button='right')

        # 3. 键盘动作（从动态映射查找按键名）
        key_name = self.idx_to_key.get(key_act, "")
        if key_name:
            pyautogui.press(key_name)

        # 4. 等一帧
        time.sleep(1.0 / 15)

        self.step_count += 1

        # 5. 观测 & 奖励
        self.frame_stack.append(self._grab_gray())
        obs = self._get_stacked_obs()
        reward = self._compute_reward()
        terminated = self._check_terminated()
        truncated = self.step_count >= MAX_STEPS

        self.episode_reward += reward

        info = {
            "step": self.step_count,
            "episode_reward": self.episode_reward,
            "milestones": list(self.milestones_hit),
        }

        return obs, reward, terminated, truncated, info

    def _compute_reward(self):
        reward = STEP_PENALTY
        if not self.templates:
            return reward

        full_gray = self._grab_full_gray()

        for name, tpl in self.templates.items():
            repeatable = name in ('item_glow',)
            if not repeatable and name in self.milestones_hit:
                continue

            th, tw = tpl.shape[:2]
            fh, fw = full_gray.shape[:2]
            if th > fh or tw > fw:
                continue

            res = cv2.matchTemplate(full_gray, tpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)

            if max_val > MATCH_THRESHOLD:
                reward += TEMPLATE_REWARDS[name]
                self.milestones_hit.add(name)

        return reward

    def _check_terminated(self):
        if 'main_menu' not in self.templates:
            return False

        full_gray = self._grab_full_gray()
        tpl = self.templates['main_menu']
        th, tw = tpl.shape[:2]
        fh, fw = full_gray.shape[:2]
        if th > fh or tw > fw:
            return False

        res = cv2.matchTemplate(full_gray, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)

        return max_val > 0.85 and self.step_count > 100

    def render(self):
        if self.render_mode == "human" and self.frame_stack:
            cv2.imshow("D2R Warlock Env", self.frame_stack[-1])
            cv2.waitKey(1)

    def close(self):
        if self._left_held:
            pyautogui.mouseUp(button='left')
            self._left_held = False
        if self.sct is not None:
            self.sct.close()
            self.sct = None
        cv2.destroyAllWindows()
