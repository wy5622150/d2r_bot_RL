# boxing-game —— 拳击俱乐部战斗 MVP

React 19 + Phaser 4 + TypeScript。与仓库根目录的 D2R Python 项目完全独立，
在这个子目录里干活时不要碰仓库根目录的 Python 代码。

## 铁律

1. **`src/core/` 是纯 TypeScript。** 不许 import Phaser、React 或任何浏览器 API。
   它要能 headless 运行、能被 vitest 直接测。
2. **战斗规则只写在 `src/core/`。** `src/game/`（Phaser）只负责把事件流演出来，
   `src/ui/`（React）只负责界面。渲染层出现任何"判定"逻辑都是 bug。
3. **数值只改 `src/core/moves.ts` 和 `src/core/fighters.ts`。**
   这两个文件里当前是**占位数值**，等拿到原版 Punch Club 的招式表后整体替换。
   目标是完全对标原版，**不要自己发明机制或调平衡**。
4. **不要往战斗系统里加原作没有的机制。** 已经因为这个删过一轮
   （telegraph 起手、刺拳破防）。原作有的：伤害、体力消耗、命中率、格挡、闪避、
   反击、抽体力、震慑、体力归零惩罚。
5. **确定性不能破。** 所有随机数走 `src/core/rng.ts` 的 seeded RNG，
   状态存在 `FightState.rng` 里。`Math.random()` 一律禁止 —— 有单测盯着。

## 架构速记

```
core/   引擎：simulateRound() 同步算完一整个回合 → RoundEvent[]
game/   Phaser 按时间线回放 RoundEvent[]
ui/     React：选对手 / 配槽 / HUD / 结算
        三者只通过 game/EventBus.ts 通信
```

"先模拟后播放"是刻意的：战斗结果与渲染完全解耦，同 seed 同结果，
加速/跳过只改回放节奏而不影响判定。

## 常用命令

```bash
npm run dev          # 开发服务器
npm test             # core/ 单测（27 个）
npm run build        # tsc --noEmit + vite build
npm run sync:skills  # 重新同步 Phaser 官方 skill（升级 phaser 后跑）
```

## Phaser 4

`.claude/skills/` 下有 Phaser 官方随 npm 包发布的 28 个 skill（MIT，原样拷贝，勿手改）。
写 Phaser 代码前先查那里，特别是 `v3-to-v4-migration`（v4 换了渲染器，
`Create.GenerateTexture` / `TextureManager.generate` 已移除、Shape 没有 `setTint()`、
`Math.TAU` 变成了 PI×2）。
