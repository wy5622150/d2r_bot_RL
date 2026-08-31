# boxing-game —— 拳击俱乐部一代战斗 MVP

React 19 + Phaser 4 + TypeScript。与仓库根目录的 D2R Python 项目完全独立，
在这个子目录里干活时不要碰仓库根目录的 Python 代码。

## 铁律

1. **对标《拳击俱乐部》一代（The Dark Fist），不是二代。**
   一代**没有**先手值 initiative、没有每招的 initiative 消耗、没有进攻槽/防守槽分开、
   没有跨回合出招游标 —— 这些都是二代的。这套代码最初误按二代模型写过一版，已整体重写，
   不要再退回去。

2. **`src/core/unknowns.ts` 是唯一允许出现非原版数值的地方。**
   其它文件只能用 `data/punch-club-source.md` 里已确证的公式。任何未知量必须从
   `unknowns.ts` 取，并在那里写清楚「原版怎么说的」和「我们为什么这么填」。
   往别处塞一个"感觉合理"的常数 = 破坏了整个项目的可信度。

3. **不要自行调平衡。** 数值全部来自原版资料。如果打出来的手感不对，
   正确做法是去查证是哪条公式的**读法**错了（例如 `armorMode` 那一项），
   而不是改一个数字让它"好玩一点"。

4. **`src/core/` 是纯 TypeScript。** 不许 import Phaser、React 或任何浏览器 API。
   要能 headless 运行、能被 vitest 直接测。

5. **战斗规则只写在 `src/core/`。** `src/game/`（Phaser）只负责把事件流演出来，
   `src/ui/`（React）只负责界面。渲染层出现任何"判定"逻辑都是 bug。

6. **确定性不能破。** 所有随机数走 `src/core/rng.ts` 的 seeded RNG，
   状态存在 `FightState.rng` 里。`Math.random()` 一律禁止 —— 有单测盯着。

## 已确证 vs 未知

已确证（照搬，不许改）：baseHP/HP、`ACC = 3·AGI/(STR+AGI+STM)`、`REG = 5+STM×1.5`、
`ARM = STM×1.3`、技能的 `base + 系数×STR` 与 `base + 系数×ACC`、体力归零 +10 伤害并击倒、
闪避成功完全免伤、最多 20 回合、5 个共享技能槽、出招顺序无意义。

未知（在 `unknowns.ts`）：闪避率公式、格挡减伤与成功率、防守选取算法、回合计时器时长、
读分公式、战斗内回复的时机与量、击倒时长与起身回复、暴击（一代没找到任何暴击公式）、
以及 **ARM 是直接相减还是按比例** —— 最后这项对手感影响最大，注释里有两种读法的实测数据。

## 架构速记

```
core/   引擎：simulateRound() 同步算完一整个回合 → RoundEvent[]
game/   Phaser 按时间线回放 RoundEvent[]
ui/     React：选对手 / 选技能 / HUD / 结算
        三者只通过 game/EventBus.ts 通信
```

"先模拟后播放"是刻意的：战斗结果与渲染完全解耦，同 seed 同结果，
加速/跳过只改回放节奏而不影响判定。

## 常用命令

```bash
npm run dev          # 开发服务器
npm test             # core/ 单测
npm run build        # tsc --noEmit + vite build
npm run sync:skills  # 重新同步 Phaser 官方 skill（升级 phaser 后跑）
```

## Phaser 4

`.claude/skills/` 下有 Phaser 官方随 npm 包发布的 28 个 skill（MIT，原样拷贝，勿手改）。
写 Phaser 代码前先查那里，特别是 `v3-to-v4-migration`（v4 换了渲染器，
`Create.GenerateTexture` / `TextureManager.generate` 已移除、Shape 没有 `setTint()`、
`Math.TAU` 变成了 PI×2）。
