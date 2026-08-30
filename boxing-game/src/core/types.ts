/**
 * 战斗核心类型定义 —— 对标《拳击俱乐部》一代（The Dark Fist）。
 * 本目录（core/）是纯 TypeScript，不依赖 Phaser 或 React，可以 headless 运行与单测。
 *
 * 一代与二代的关键差别（二代的东西一律不要出现在这里）：
 * - 一代是**一组共享技能槽**（最多 5 个），不是二代的进攻槽/防守槽分开
 * - 一代**没有**数值化的先手值 initiative 与每招的 initiative 消耗
 * - 技能槽顺序无意义（开发者原话），出招是随机抽取，没有跨回合游标
 */

export type Side = 'player' | 'opponent';

/** 一代的四个流派 */
export type School = 'Basic' | 'Bear' | 'Tiger' | 'Turtle';

/** 只保留能进战斗的两类；被动/修饰/流派节点不在 MVP 范围内 */
export type MoveKind = 'attack' | 'defense';

/** 渲染层挑动画用；换真素材时对着这个表接序列帧 */
export type AnimKey =
  | 'punch'
  | 'high_punch'
  | 'uppercut'
  | 'crosspunch'
  | 'backhand'
  | 'kick'
  | 'high_kick'
  | 'low_kick'
  | 'knee'
  | 'chop'
  | 'block'
  | 'dodge';

/**
 * 原版数值的形态：`base + perStr × STR`。
 * 例：Punch 的伤害写作 `1+0.7(str)` → { base: 1, perStr: 0.7 }
 */
export interface StatFormula {
  base: number;
  perStr: number;
}

/**
 * 命中率的形态：`base + perAcc × ACC`（百分比）。
 * 例：Punch 的命中写作 `70+20(Hit%)` → { base: 70, perAcc: 20 }
 * 其中 ACC = 3·AGI/(STR+AGI+STM)，见 stats.ts。
 */
export interface AccuracyFormula {
  base: number;
  perAcc: number;
}

export interface Move {
  id: string;
  nameEn: string;
  /** 第三方 2016 年中文资料的译名，非官方本地化 */
  nameZhUnofficial: string;
  school: School;
  kind: MoveKind;
  /** 防守技能的两种形态：格挡（减伤）与闪避（成功则完全免伤） */
  defenseKind?: 'block' | 'dodge';
  /** 原样保留的效果描述 */
  effects: string[];
  energyCost: StatFormula;
  damage?: StatFormula;
  accuracy?: AccuracyFormula;
  /** 该条数值的出处 */
  sourceUrl: string;
  /** 资料里记录的源内部矛盾，界面上会提示 */
  conflict?: string;
  anim: AnimKey;
}

/** 一代用 STM 指代耐力，这里跟随原始资料的命名以减少转录误差 */
export interface Stats {
  str: number;
  agi: number;
  stm: number;
}

export interface Derived {
  /** baseHP = 38 + 6·STR + 3·AGI + 8·STM + 16·min(STR,AGI,STM) */
  baseHp: number;
  /** HP = baseHP/2 + baseHP·Health/2 */
  maxHp: number;
  maxEnergy: number;
  /** ACC = 3·AGI / (STR+AGI+STM)，招式命中率公式里的 Hit% */
  acc: number;
  /** REG = 5 + STM×1.5（属性页显示值） */
  reg: number;
  /** ARM = STM×1.3 */
  arm: number;
}

/** 一代：一组共享技能槽，攻击与防守技能混装 */
export type Loadout = string[];

export type AiStyle = 'manual' | 'aggressive' | 'defensive' | 'balanced';

export interface FighterDef {
  id: string;
  name: string;
  /** 渲染用主色 */
  color: number;
  tagline: string;
  stats: Stats;
  /** 健康度 0..1，进入 HP 公式 */
  health: number;
  style: AiStyle;
  /** 可用技能池 */
  pool: string[];
  /** 初始装备的技能 */
  loadout: Loadout;
  /** 属性/技能的出处，界面上会显示 */
  sourceUrl?: string;
  sourceNote?: string;
}

export interface FighterState {
  def: FighterDef;
  derived: Derived;
  hp: number;
  energy: number;
  /** 被击倒后要跳过的阶段数 */
  lostPhases: number;
  loadout: Loadout;
}

export type FightMethod = 'ko' | 'decision';

export interface FightResult {
  /** null = 平局 */
  winner: Side | null;
  method: FightMethod;
  round: number;
}

export interface FightState {
  round: number;
  player: FighterState;
  opponent: FighterState;
  /** 当前进攻方 */
  attacker: Side;
  /** 可序列化的 RNG 状态，保证同 seed 同结果 */
  rng: number;
  over: boolean;
  result: FightResult | null;
}

/** 每个事件都带一份双方血量/体力快照，HUD 据此与画面同步 */
export interface EventSnapshot {
  hp: Record<Side, number>;
  energy: Record<Side, number>;
}

interface Base extends EventSnapshot {
  /** 回合内序号 */
  tick: number;
  /** 解说文案，由引擎生成，渲染层只管显示 */
  text: string;
}

export type RoundEvent =
  | (Base & { type: 'round_start'; round: number })
  | (Base & { type: 'attack'; side: Side; move: string; defenseMove: string | null })
  | (Base & { type: 'dodge'; side: Side; move: string })
  | (Base & { type: 'miss'; side: Side; move: string })
  | (Base & {
      type: 'hit';
      side: Side;
      target: Side;
      move: string;
      damage: number;
      blocked: number;
      /** 守方体力见底时的额外惩罚伤害 */
      exhaustBonus: boolean;
    })
  | (Base & { type: 'exhausted'; side: Side; energyGain: number })
  | (Base & { type: 'skip'; side: Side })
  | (Base & { type: 'knockdown'; side: Side })
  | (Base & { type: 'ko'; side: Side })
  | (Base & { type: 'round_end'; round: number });

export type RoundEventType = RoundEvent['type'];
