# 拳击俱乐部 · 战斗 MVP

用 **React 19 + Phaser 4 + TypeScript** 复刻《拳击俱乐部》(Punch Club) 的战斗场景。

原作战斗的核心不是操作，而是**配置**：把学到的招式塞进有限的进攻槽 / 防守槽，
战斗随后全自动播放，每个回合之间可以重新配槽来针对对手。本项目就是把这套
「配槽 → 自动结算 → 中场调整」的决策闭环完整跑通。

```bash
npm install
npm run dev        # http://localhost:5173
npm test           # 战斗引擎单测
npm run build      # tsc --noEmit + vite build
```

## ⚠️ 数值现状

`src/core/moves.ts` 和 `src/core/fighters.ts` 里的数字是**占位值，不是原版数值**。
目标是第一版就完全对标原版的招式表（Basic / Bear / Tiger / Turtle 四系共 27 个技能的
伤害、体力消耗、命中率、先手消耗），但开发沙箱的出站策略封了
`punchclub.wiki.gg` / `punch-club.fandom.com` / `steamcommunity.com`，暂时取不到原表。

拿到原版数据后**只需要整体替换这两个数据文件**，引擎、回放层、界面都不用动 ——
这也是把数值全部集中在 `core/` 数据文件里的原因。

机制上只保留了原作真实存在的那些：伤害、体力消耗、命中率、格挡、闪避、反击、
抽体力、震慑、体力归零惩罚。没有自己发明的机制。

## 玩法

1. **选对手** —— 三种风格：压迫型 / 游斗型 / 铁壁型
2. **赛前配槽** —— 4 个进攻槽 + 3 个防守槽
3. **自动对打** —— 3 个回合，可 1x / 2x / 4x 或直接跳过
4. **回合间重新配槽** —— 你和对手都会改
5. **结算** —— KO，或者打满三回合按剩余血量百分比读分

## 战斗规则

复刻自原作的几条关键机制：

| 机制 | 说明 |
|---|---|
| **进攻槽循环** | 4 个槽按顺序循环出手，**游标跨回合延续**（不会每回合从头开始） |
| **槽位权重** | 同一招放多份 = 被打出来的频率更高 |
| **空槽回体力** | 空的进攻槽被轮到时不出手，改为回复体力 |
| **防守槽随机抽取** | 挨打时从 3 个防守槽里随机抽一个应对；**空槽也参与抽取**，等于有概率完全不设防 |
| **先手值 initiative** | 敏捷决定连续出手次数：我方打完 N 拍才轮到对方。4 vs 2 就是 2:1 的出手比 |
| **体力归零惩罚** | 挨打瞬间体力见底 → 额外 +10 伤害并被打倒，下一拍起不来 |
| **胜负** | 血量归零 KO；打满 3 回合按剩余血量**百分比**读分 |

派生属性（`src/core/stats.ts`）：

```
maxHp      = 100 + 耐力 × 10        maxEnergy  = 50 + 耐力 × 5
initiative = 2 + ⌊敏捷 / 3⌋          damageMult = 1 + 力量 × 0.05
armor      = ⌊耐力 / 3⌋              critChance = min(0.03 + 敏捷 × 0.01, 0.35)
dodgeBonus = min(敏捷 × 0.015, 0.25)  roundRegen = ⌊maxEnergy × 0.2⌋ + 耐力
```

伤害结算顺序：闪避判定 → 命中判定 → `伤害 = max(1, round(基础伤害 × 力量系数 × 暴击) × (1 − 减伤%) − 护甲)`
→ 反击 → 抽体力 → 震慑 → 体力归零惩罚。

## 架构：三层严格分离

```
src/core/    纯 TypeScript 战斗引擎  —— 零 Phaser / 零 React 依赖，可 headless 运行、可单测
src/game/    Phaser 4 回放层         —— 只负责把引擎产出的事件流演出来，不含任何规则
src/ui/      React 界面层            —— 选对手 / 配槽 / HUD / 结算
```

**关键设计：先模拟，后播放。**
引擎不是每帧驱动的 —— `simulateRound()` 同步把一整个回合算完，吐出一条 `RoundEvent[]`
事件流；Phaser 场景按时间线回放它。带来三个好处：

- 战斗结果 100% 确定（同 seed 同结果），可以写单测、可以做战报回放
- 1x / 2x / 4x / 跳过 是免费的 —— 只改回放节奏，不影响结果
- 渲染层写崩了也不会影响战斗平衡

因为回合之间需要玩家重新配槽，所以按**回合**粒度模拟：
`模拟第 N 回合 → 回放 → 暂停配槽 → 模拟第 N+1 回合`。

### React ↔ Phaser 通信

两边都不持有对方的对象，只走 `src/game/EventBus.ts`：

| 方向 | 事件 | 载荷 |
|---|---|---|
| React → Phaser | `fight:setup` | 双方外观 / 名字 / 初始状态 |
| React → Phaser | `fight:play-round` | `RoundEvent[]` |
| React → Phaser | `fight:set-speed` / `fight:skip` | 播放速度 / 跳过本回合 |
| Phaser → React | `scene:ready` | 场景 create 完毕 |
| Phaser → React | `fight:event-shown` | 画面播到了哪一条事件（HUD 据此更新，不提前剧透） |
| Phaser → React | `fight:round-done` | 本回合播完 |

血条和解说跟着 `fight:event-shown` 走，所以数字永远和画面同步。

## 美术

全部由 `src/game/view/BoxerView.ts` 和 `ArenaScene.drawArena()` 用 Phaser 的 Shape
程序化画出来，**零外部素材依赖**，clone 下来就能跑。换真素材时把 `BoxerView`
换成 Sprite + 序列帧即可，`punch/defend/hurt/fall/...` 这组接口保持不变；
招式上的 `anim` 字段就是给序列帧用的动画键。

## Phaser 4 官方 skill

`.claude/skills/` 下是 **Phaser 官方随 npm 包一起发布的 28 个 Claude Code skill**
（`node_modules/phaser/skills/`，Phaser Studio Inc.，MIT），原样拷贝，未做修改。
升级 phaser 版本后重新同步：

```bash
npm run sync:skills     # 也会在 npm install 后自动跑
```

来源与清单见 `.claude/skills/SOURCE.md`。

## 测试

```bash
npm test
```

`src/core/__tests__/` 覆盖：

- **确定性** —— 同 seed 同配槽 → 事件流逐条相等
- **先手值** —— 4 vs 2 时动作数正好 2:1；先手高的一方先出手
- **进攻槽** —— 空槽回体力、游标跨回合延续、体力不足踉跄
- **防守槽** —— 份数越多抽中概率越高（3:1 ≈ 3 倍）、空槽 = 不设防、体力不够防守失败
- **攻防结算** —— 百分比减伤 + 护甲、招架反击、闪避零伤害、击腹抽体力
- **体力归零惩罚** —— +10 伤害并触发倒地
- **结束条件** —— KO 立即终止；打满三回合按血量百分比读分 / 平局
- **平衡冒烟** —— 只保证「配槽有意义、每个对手都打得过」，不锁具体胜率（数值调优等原版数据到位后再做）

## 目录

```
src/core/     types / rng / stats / moves / fighters / resolve / engine / ai / match
src/game/     EventBus / PhaserGame.tsx / scenes/ArenaScene / view/BoxerView
src/ui/       store(zustand) / screens / components / styles.css
scripts/      sync-phaser-skills.mjs
.claude/skills/  Phaser 官方 skill（同步自 npm 包）
```
