# build_keymap.py
# 扫描 data/*.npz，自动发现所有用过的键盘按键，生成 keymap.json
# 训练和推理时加载此文件来确定键盘动作空间
#
# 用法: python build_keymap.py
# 输出: models/keymap.json

import os
import glob
import json
import numpy as np

DATA_DIR = "data"
OUTPUT_PATH = os.path.join("models", "keymap.json")

os.makedirs("models", exist_ok=True)


def build():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "ep_*.npz")))
    if not files:
        print(f"在 {DATA_DIR} 中找不到 ep_*.npz 文件")
        return

    print(f"扫描 {len(files)} 个录制文件...")

    # 统计所有按键
    all_keys = {}
    total_frames = 0

    for fpath in files:
        data = np.load(fpath, allow_pickle=True)
        keys = data["act_key"]  # string array
        total_frames += len(keys)

        for k in keys:
            k = str(k)
            all_keys[k] = all_keys.get(k, 0) + 1

    # "" (无按键) 始终是 index 0
    none_count = all_keys.pop("", 0)

    # 按使用频率排序（高频按键靠前，方便调试）
    sorted_keys = sorted(all_keys.items(), key=lambda x: -x[1])

    # 构建映射: "" → 0, 最常用键 → 1, 次常用 → 2, ...
    keymap = {"": 0}
    for i, (key_name, _) in enumerate(sorted_keys):
        keymap[key_name] = i + 1

    n_keys = len(keymap)  # 包含 "" (无按键)

    # 反向映射（index → key name，供 env 使用）
    idx_to_key = {v: k for k, v in keymap.items()}

    config = {
        "keymap": keymap,           # key_name → index
        "idx_to_key": {str(k): v for k, v in idx_to_key.items()},  # index → key_name (JSON key must be str)
        "n_key_actions": n_keys,
        "n_grid": 144,
        "n_mouse_actions": 4,
        "grid_w": 16,
        "grid_h": 9,
        "total_frames": total_frames,
        "total_episodes": len(files),
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"\n已生成: {OUTPUT_PATH}")
    print(f"总帧数: {total_frames}")
    print(f"总 episode: {len(files)}")
    print(f"键盘动作空间: {n_keys} 种（含'无按键'）")
    print(f"\n按键映射:")
    print(f"  0: (无按键)  [{none_count} 次]")
    for key_name, count in sorted_keys:
        idx = keymap[key_name]
        # 将 pyautogui 可用的键名显示出来
        display = key_name if key_name else "(空)"
        print(f"  {idx}: '{display}'  [{count} 次, {count/total_frames*100:.1f}%]")

    print(f"\n动作空间总大小: MultiDiscrete([144, 4, {n_keys}])")


if __name__ == "__main__":
    build()
