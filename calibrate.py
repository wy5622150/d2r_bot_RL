# calibrate.py
# 标定工具：截图 + 绘制网格叠加 + 160x90 预览
# 用法: python calibrate.py

import time
import numpy as np
import cv2
import mss

# ===== 修改这里 =====
MONITOR = {"top": 0, "left": 0, "width": 3840, "height": 2160}
# ====================

GRID_W = 16
GRID_H = 9
OBS_W = 160
OBS_H = 90

sct = mss.mss()

print("3 秒后截图，请切换到 D2R 游戏窗口...")
time.sleep(3)

img = np.array(sct.grab(MONITOR))
img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

# 1. 保存原始截图
cv2.imwrite("calibrate_screenshot.png", img_bgr)
print(f"原始截图: calibrate_screenshot.png ({img.shape[1]}x{img.shape[0]})")

# 2. 绘制网格叠加
grid_img = img_bgr.copy()
cell_w = MONITOR["width"] // GRID_W
cell_h = MONITOR["height"] // GRID_H

for i in range(1, GRID_W):
    x = i * cell_w
    cv2.line(grid_img, (x, 0), (x, MONITOR["height"]), (0, 255, 0), 2)

for j in range(1, GRID_H):
    y = j * cell_h
    cv2.line(grid_img, (0, y), (MONITOR["width"], y), (0, 255, 0), 2)

for j in range(GRID_H):
    for i in range(GRID_W):
        idx = j * GRID_W + i
        cx = i * cell_w + cell_w // 2
        cy = j * cell_h + cell_h // 2
        cv2.putText(grid_img, str(idx), (cx - 20, cy + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

grid_preview = cv2.resize(grid_img, (1920, 1080))
cv2.imwrite("calibrate_grid.png", grid_preview)
print(f"网格叠加: calibrate_grid.png (16x9 网格, 每格 {cell_w}x{cell_h} px)")

# 3. 160x90 灰度预览（模型实际看到的画面）
gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
small = cv2.resize(gray, (OBS_W, OBS_H))
# 放大 4 倍方便人眼查看
preview = cv2.resize(small, (OBS_W * 4, OBS_H * 4), interpolation=cv2.INTER_NEAREST)
cv2.imwrite("calibrate_obs_preview.png", preview)
print(f"模型视角预览: calibrate_obs_preview.png ({OBS_W}x{OBS_H} 放大4倍)")

print("\n请检查:")
print("  1. calibrate_screenshot.png - 是否截到了完整游戏画面")
print("  2. calibrate_grid.png - 网格是否覆盖了整个游戏窗口")
print("  3. calibrate_obs_preview.png - 模型视角下是否能分辨菜单按钮、怪物、地形")
