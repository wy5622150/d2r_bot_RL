/**
 * 战斗核心类型定义。
 * 本目录（core/）是纯 TypeScript，不依赖 Phaser 或 React，可以 headless 运行与单测。
 */

export type Side = 'player' | 'opponent';

export type MoveKind = 'attack' | 'defense' | 'rest';

/** 渲染层用来挑动画的提示键，替换真素材时对着这个表接序列帧即可 */
export type AnimKey =
  | 'jab'
  | 'cross'
  | 'hook'
  | 'uppercut'
  | 'overhand'
  | 'bodyshot'
  | 'haymaker'
  | 'guard'
  | 'slip'
  | 'parry'
  | 'clinch'
  | 'breathe';

export interface Move {
  id: string;
  name: string;
  kind: MoveKind;
  desc: string;
  /** 使用该招式消耗的能量 */
  energyCost: number;

  // --- attack ---
  baseDamage?: number;
  /** 基础命中率 0..1 */
  accuracy?: number;
  /** 命中后使对手下个动作作废的概率 */
  stunChance?: number;
  /** 命中后额外抽干对手的能量 */
  energyDrain?: number;

  // --- defense ---
  /** 百分比减伤 0..1 */
  blockPct?: number;
  /** 基础闪避率，会叠加防守方的 dodgeBonus */
  dodge?: number;
  /** 反击系数：把挡下的伤害按比例回敬给攻方 */
  counter?: number;

  // --- rest ---
  energyRestore?: number;

  anim: AnimKey;
}

export interface Stats {
  /** 力量：伤害 */
  str: number;
  /** 敏捷：先手值、暴击、闪避 */
  agi: number;
  /** 耐力：血量、能量池与回复、减伤 */
  sta: number;
}

export interface Derived {
  maxHp: number;
  maxEnergy: number;
  /** 连续出手次数 */
  initiative: number;
  damageMult: number;
  /** 固定减伤 */
  armor: number;
  critChance: number;
  /** 叠加到闪避招式上的额外闪避率 */
  dodgeBonus: number;
  /** 每回合开始回复的能量 */
  roundRegen: number;
}

/** 槽位：招式 id 或 null（空槽） */
export type Slot = string | null;

export interface Loadout {
  /** 进攻槽，长度 OFFENSE_SLOTS。同一招放多份 = 出现频率更高；空槽 = 回能量 */
  offense: Slot[];
  /** 防守槽，长度 DEFENSE_SLOTS。被攻击时按槽位权重随机抽一个应对 */
  defense: Slot[];
}

export type AiStyle = 'pressure' | 'outboxer' | 'wall' | 'manual';

export interface FighterDef {
  id: string;
  name: string;
  /** 渲染用主色 */
  color: number;
  /** 一句话人设，选对手界面显示 */
  tagline: string;
  stats: Stats;
  style: AiStyle;
  /** 可用招式池（招式 id） */
  pool: string[];
  /** 初始配槽 */
  loadout: Loadout;
}

export interface FighterState {
  def: FighterDef;
  derived: Derived;
  hp: number;
  energy: number;
  /** 进攻槽游标，跨回合延续 */
  offCursor: number;
  /** 下个动作作废（被震慑） */
  stunned: boolean;
  /** 下个动作作废（被击倒爬起） */
  knockedDown: boolean;
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
  /** 当前攻方 */
  turn: Side;
  /** 当前攻方还剩几个连续动作 */
  turnActionsLeft: number;
  /** 可序列化的 RNG 状态，保证同 seed 同结果 */
  rng: number;
  over: boolean;
  result: FightResult | null;
}

/** 每个事件都带一份双方血量/能量快照，HUD 据此与画面同步 */
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
  | (Base & { type: 'round_start'; round: number; regen: Record<Side, number> })
  | (Base & { type: 'turn_switch'; side: Side; actions: number })
  | (Base & { type: 'empty_slot'; side: Side; energyGain: number })
  | (Base & { type: 'rest'; side: Side; move: string; energyGain: number })
  | (Base & { type: 'exhausted'; side: Side; energyGain: number })
  | (Base & { type: 'skip'; side: Side; cause: 'stun' | 'knockdown' })
  | (Base & { type: 'attack'; side: Side; move: string; defenseMove: string | null })
  | (Base & { type: 'dodge'; side: Side; move: string })
  | (Base & { type: 'miss'; side: Side; move: string })
  | (Base & {
      type: 'hit';
      side: Side;
      target: Side;
      move: string;
      damage: number;
      crit: boolean;
      blocked: number;
      /** 守方能量见底时的额外惩罚伤害 */
      exhaustBonus: boolean;
    })
  | (Base & { type: 'counter'; side: Side; target: Side; damage: number })
  | (Base & { type: 'stun'; side: Side })
  | (Base & { type: 'knockdown'; side: Side })
  | (Base & { type: 'ko'; side: Side })
  | (Base & { type: 'round_end'; round: number });

export type RoundEventType = RoundEvent['type'];
