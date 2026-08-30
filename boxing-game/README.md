# 拳击俱乐部 · 战斗 MVP

用 **React 19 + Phaser 4 + TypeScript** 复刻《拳击俱乐部》一代（Punch Club: The Dark Fist，2016）的战斗场景。

原作战斗的核心不是操作，而是**配置**：从学到的技能里挑 5 个装上，战斗随后全自动播放，
每个回合之间可以重新挑一次来针对对手。本项目就是把这套决策闭环完整跑通。

```bash
npm install
npm run dev        # http://localhost:5173
npm test           # 战斗引擎单测
npm run build      # tsc --noEmit + vite build
```

## 数据来源与可信度

所有数值都逐字转录自 **`data/punch-club-source.md`**（人工核对版）与
**`data/punch-club-source.json`**（结构化版），出处逐条标注到 wiki.gg / Fandom / Steam 页面。
代码里没有一个"看起来合理"的数字。

一代的战斗结算只有一部分被公开逆向出来。**已确证**的部分直接照搬：

| 项 | 原版公式 |
|---|---|
| 血量 | `baseHP = 38 + 6·STR + 3·AGI + 8·STM + 16·min(STR,AGI,STM)`；`HP = baseHP/2 + baseHP·Health/2` |
| 命中系数 | `ACC = 3·AGI / (STR+AGI+STM)` |
| 体力回复 | `REG = 5 + STM×1.5`（属性页显示值） |
| 护甲 | `ARM = STM×1.3` |
| 技能伤害 / 体力 | `base + 系数 × STR`，常规四舍五入 |
| 技能命中 | `base + 系数 × ACC`（百分比） |
| 体力归零 | 挨打额外 **+10 伤害**并被击倒 |
| 闪避 | 成功则该次攻击完全不造成伤害 |
| 回合数 | 最多 **20 回合** |
| 技能槽 | **一组共享槽位，最多 5 个**（攻守混装） |
| 出招顺序 | **顺序无意义**（Lazy Bear 开发者原话），随机抽取 |
| 体力上限 | 100 —— ⚠ 唯一来源是 Steam 攻略，作者自承部分内容为推测 |

**未公开**的部分全部集中在 `src/core/unknowns.ts`，每条都写明「原版怎么说的」与「我们为什么这么填」：
闪避率公式、格挡减伤量与成功率、防守技能的选取算法、回合计时器时长、20 回合后的判胜公式、
战斗内体力回复的时机与量、击倒持续时间与起身回复量、暴击（一代没找到任何暴击公式）。

> **最值得优先查证的一项**：`ARM` 是「直接相减」还是「按比例减伤」。
> wiki 把直接相减标注为初步判断（tentative）。实测（各 600 场）：
> 直接相减 → KO 率 0%，每场都拖到 20 回合读分；按比例 → KO 率 100%，平均 12–16 回合。
> 真实的一代两种结局都常见，说明正确公式在两者之间。默认保留 wiki 字面的直接相减，
> 开关在 `unknowns.ts` 的 `armorMode`。

### 与二代的界线

一代**没有**数值化的先手值 initiative、没有每招的 initiative 消耗、没有进攻槽/防守槽分开、
没有跨回合的出招游标 —— 这些全是二代的机制。本项目一个都没有实现。
（这套代码最初按二代模型写过一版，拿到一代资料后整体重写。）

### 技能收录范围

资料里共 109 条 ability/skill 记录（105 个技能树节点 + 4 个初始战斗能力），但只有
**11 个攻击技能同时给出了伤害/体力/命中三条公式**，另有 Block 与 Dodge 给出体力公式。
`src/core/moves.ts` 只收录这 13 个可直接结算的技能。被有意排除的：

- **Knee Crush** —— 有伤害与体力公式，但没有命中公式，补上就是编造
- **Power Block / Sidestep** —— 只有体力公式，额外效果的量级未知
- 全部被动（38）、修饰（26）、流派节点（3）—— 不是可装备的战斗技能

同理，对手只有两个：一代公开资料里能查到完整属性的只有 **Silver**（5/5/5）与
**Big Bobo**（8/5/3）。

## 玩法

1. **选对手** —— Silver 或 Big Bobo
2. **赛前选技能** —— 从 13 个技能里挑 5 个（可以重复挑同一个来提高它被抽中的概率）
3. **自动对打** —— 最多 20 回合，可 1x / 2x / 4x 或直接跳过
4. **回合间重新选** —— 你和对手都会改
5. **结算** —— KO，或打满 20 回合读分

## 架构：三层严格分离

```
src/core/    纯 TypeScript 战斗引擎 —— 零 Phaser / 零 React 依赖，headless 可测
src/game/    Phaser 4 回放层        —— 只演出事件流，不含任何战斗规则
src/ui/      React 界面层           —— 选对手 / 选技能 / HUD / 结算
```

**关键设计：先模拟，后播放。** `simulateRound()` 同步把一整个回合算完并返回 `RoundEvent[]`，
Phaser 场景按时间线回放它。于是：

- 战斗结果与渲染完全解耦，同 seed 必定同结果（有单测钉住）
- 1x / 2x / 4x / 跳过 是免费的 —— 只改回放节奏，不影响判定
- 血条和解说跟着「画面播到了哪一条」走，数字不会跑在画面前面

### React ↔ Phaser 通信

两边都不持有对方的对象，只走 `src/game/EventBus.ts`：

| 方向 | 事件 | 载荷 |
|---|---|---|
| React → Phaser | `fight:setup` | 双方外观 / 名字 / 初始状态 |
| React → Phaser | `fight:play-round` | `RoundEvent[]` |
| React → Phaser | `fight:set-speed` / `fight:skip` | 播放速度 / 跳过本回合 |
| Phaser → React | `scene:ready` | 场景 create 完毕 |
| Phaser → React | `fight:event-shown` | 画面播到了哪一条事件 |
| Phaser → React | `fight:round-done` | 本回合播完 |

## 美术

全部由 `src/game/view/BoxerView.ts` 和 `ArenaScene.drawArena()` 用 Phaser 的 Shape
程序化画出来，**零外部素材依赖**。换真素材时把 `BoxerView` 换成 Sprite + 序列帧即可，
`punch/defend/hurt/fall/...` 这组接口保持不变；技能上的 `anim` 字段就是给序列帧用的动画键。

## Phaser 4 官方 skill

`.claude/skills/` 下是 **Phaser 官方随 npm 包一起发布的 28 个 Claude Code skill**
（`node_modules/phaser/skills/`，Phaser Studio Inc.，MIT），原样拷贝，未做修改。
升级 phaser 版本后重新同步：

```bash
npm run sync:skills     # 也会在 npm install 后自动跑
```

## 测试

```bash
npm test
```

`src/core/__tests__/` 覆盖：

- **原版公式转录**（`stats.test.ts`）—— baseHP / HP / ACC / REG / ARM 与各技能的三条公式
  逐个钉死具体数值；资料里记录的源内部矛盾（Low Kick 命中、Cutthroat 数值行）也断言其被保留
- **攻防结算**（`resolve.test.ts`）—— 体力消耗公式、随机抽招、装多份提高概率、闪避免伤、
  格挡减伤、体力归零 +10 与击倒
- **回合结构**（`engine.test.ts`）—— 双方严格交替（没有先手值）、最多 20 回合、
  KO 立即终止、读分与平局、体力耗尽只能喘气、技能配置校验（必须正好 5 个、至少一个攻击技能）
- **确定性** —— 同 seed 同配置 → 事件流逐条相等

## 目录

```
data/         原版资料（转录版 + 结构化版），代码里所有数字的出处
src/core/     types / rng / stats / moves / fighters / resolve / engine / ai / match
              unknowns.ts ← 唯一允许出现非原版数值的地方
src/game/     EventBus / PhaserGame.tsx / scenes/ArenaScene / view/BoxerView
src/ui/       store(zustand) / screens / components / styles.css
.claude/skills/  Phaser 官方 skill（同步自 npm 包）
```
