# collect_data.py
# 录制人类演示数据（鼠标位置 + 鼠标按键 + 所有键盘按键）
# 键盘不预设映射，记录原始按键名，训练时自动构建动作空间
#
# 用法：python collect_data.py
#   - 游戏窗口保持 4K 全屏
#   - 按 F10 开始录制当前 episode
#   - 手动玩一整局
#   - 按 F11 结束当前 episode
#   - 按 Ctrl+C 结束全部录制

import os
import time
import threading
import winsound
import numpy as np
import cv2
from collections import deque
import mss
from pynput import keyboard, mouse

# ====================== 配置 ======================
RECORD_MONITOR = {"top": 0, "left": 0, "width": 3840, "height": 2160}
SAVE_DIR = "data"
FPS = 15
OBS_W = 160
OBS_H = 90

GRID_W = 16
GRID_H = 9
CELL_W = RECORD_MONITOR["width"] // GRID_W
CELL_H = RECORD_MONITOR["height"] // GRID_H
N_GRID = GRID_W * GRID_H

START_KEY = keyboard.Key.f10   # 开始录制
STOP_KEY = keyboard.Key.f11    # 结束当前 episode
# ===================================================

os.makedirs(SAVE_DIR, exist_ok=True)


def beep_start():
    """录制开始提示音：两声短促高音"""
    winsound.Beep(1000, 150)
    winsound.Beep(1500, 150)


def beep_stop():
    """录制结束提示音：一声低音"""
    winsound.Beep(600, 300)


def key_to_name(key):
    """将 pynput key 转为统一的字符串名称"""
    try:
        if key.char is not None:
            return key.char.lower()
    except AttributeError:
        pass
    name = str(key).replace("Key.", "")
    return name


class DemoRecorder:
    def __init__(self):
        self.sct = mss.mss()
        self.lock = threading.Lock()

        self.mouse_x = 0
        self.mouse_y = 0
        self.left_held = False
        self.left_clicked = False
        self.right_clicked = False
        self.current_key_name = ""
        self.should_stop = False
        self.should_start = False

    # ---------- 鼠标回调 ----------
    def _on_move(self, x, y):
        with self.lock:
            self.mouse_x = x
            self.mouse_y = y

    def _on_click(self, x, y, button, pressed):
        with self.lock:
            self.mouse_x = x
            self.mouse_y = y
            if button == mouse.Button.left:
                self.left_held = pressed
                if pressed:
                    self.left_clicked = True
            elif button == mouse.Button.right:
                if pressed:
                    self.right_clicked = True

    # ---------- 键盘回调 ----------
    def _on_press(self, key):
        if key == STOP_KEY:
            self.should_stop = True
            return
        if key == START_KEY:
            self.should_start = True
            return

        name = key_to_name(key)
        with self.lock:
            self.current_key_name = name

    def _on_release(self, key):
        if key in (STOP_KEY, START_KEY):
            return

        name = key_to_name(key)
        with self.lock:
            if self.current_key_name == name:
                self.current_key_name = ""

    # ---------- 工具 ----------
    def _pos_to_grid(self, x, y):
        rx = x - RECORD_MONITOR["left"]
        ry = y - RECORD_MONITOR["top"]
        gx = max(0, min(rx // CELL_W, GRID_W - 1))
        gy = max(0, min(ry // CELL_H, GRID_H - 1))
        return int(gy * GRID_W + gx)

    def _sample_action(self):
        with self.lock:
            grid = self._pos_to_grid(self.mouse_x, self.mouse_y)

            if self.right_clicked:
                ma = 3
                self.right_clicked = False
            elif self.left_clicked:
                ma = 2
                self.left_clicked = False
            elif self.left_held:
                ma = 1
            else:
                ma = 0

            key_name = self.current_key_name
        return grid, ma, key_name

    def _grab_frame(self):
        img = np.array(self.sct.grab(RECORD_MONITOR))
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        return cv2.resize(gray, (OBS_W, OBS_H))

    # ---------- 录制 ----------
    def record_episode(self, episode_id):
        observations = []
        act_grid = []
        act_mouse = []
        act_keys = []
        frame_stack = deque(maxlen=4)

        self.should_stop = False
        self.should_start = False
        self.left_held = False
        self.left_clicked = False
        self.right_clicked = False
        self.current_key_name = ""

        print(f"\n=== Episode {episode_id} ===")
        print("按 F10 开始录制，F11 结束本局")

        # 启动监听器（等待 F10）
        m_listener = mouse.Listener(on_move=self._on_move, on_click=self._on_click)
        k_listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        m_listener.start()
        k_listener.start()

        # 等待 F10
        while not self.should_start:
            time.sleep(0.05)

        # 声音提示：开始录制
        beep_start()
        print(">>> 录制中...")

        frame_interval = 1.0 / FPS

        try:
            while not self.should_stop:
                t0 = time.perf_counter()

                frame = self._grab_frame()
                frame_stack.append(frame)

                if len(frame_stack) == 4:
                    obs = np.stack(list(frame_stack), axis=-1)
                    g, m, k = self._sample_action()

                    observations.append(obs)
                    act_grid.append(g)
                    act_mouse.append(m)
                    act_keys.append(k)

                elapsed = time.perf_counter() - t0
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        finally:
            m_listener.stop()
            k_listener.stop()

        # 声音提示：录制结束
        beep_stop()

        n_frames = len(observations)
        if n_frames < 50:
            print(f"帧数太少 ({n_frames})，已丢弃")
            return

        key_arr = np.array(act_keys, dtype="U20")

        save_path = os.path.join(SAVE_DIR, f"ep_{episode_id:04d}.npz")
        np.savez_compressed(
            save_path,
            obs=np.array(observations, dtype=np.uint8),
            act_grid=np.array(act_grid, dtype=np.int64),
            act_mouse=np.array(act_mouse, dtype=np.int64),
            act_key=key_arr,
        )

        size_mb = os.path.getsize(save_path) / 1e6
        print(f"已保存: {save_path}  ({n_frames} 帧, {size_mb:.1f} MB)")

        am = np.array(act_mouse)
        print(f"  鼠标: 无={np.sum(am==0)} 左按住={np.sum(am==1)} "
              f"左单击={np.sum(am==2)} 右键={np.sum(am==3)}")

        unique_keys, counts = np.unique(key_arr, return_counts=True)
        key_stats = sorted(zip(unique_keys, counts), key=lambda x: -x[1])
        print(f"  键盘: {len(unique_keys)} 种按键")
        for kn, cnt in key_stats:
            label = kn if kn else "(无)"
            print(f"    {label}: {cnt}")


def main():
    recorder = DemoRecorder()
    episode_id = 0

    existing = [f for f in os.listdir(SAVE_DIR) if f.startswith("ep_") and f.endswith(".npz")]
    if existing:
        episode_id = max(int(f[3:7]) for f in existing) + 1
        print(f"检测到已有 {len(existing)} 个 episode，从 {episode_id} 继续")

    print("=" * 60)
    print("D2R 人类演示录制器")
    print(f"录制区域: {RECORD_MONITOR['width']}x{RECORD_MONITOR['height']}")
    print(f"网格: {GRID_W}x{GRID_H} = {N_GRID} 格")
    print(f"帧率: {FPS} fps | 观测: {OBS_W}x{OBS_H}")
    print()
    print("录制所有键盘操作，无需预设键位")
    print("鼠标: 0=无 1=左键按住 2=左键单击 3=右键")
    print()
    print("F10 = 开始录制 | F11 = 结束当前 episode | Ctrl+C = 退出")
    print("=" * 60)

    try:
        while episode_id < 999:
            recorder.record_episode(episode_id)
            episode_id += 1
    except KeyboardInterrupt:
        print(f"\n录制结束，共 {episode_id} 个 episode")


if __name__ == "__main__":
    main()
