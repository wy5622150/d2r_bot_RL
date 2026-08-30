import type { Derived, Stats } from './types';

/** 一场比赛的回合数 */
export const ROUNDS = 3;
/** 每回合双方合计的动作数 */
export const ACTIONS_PER_ROUND = 14;
export const OFFENSE_SLOTS = 4;
export const DEFENSE_SLOTS = 3;

/** 空进攻槽被抽中时回复的能量（原作机制：空槽 = 喘口气） */
export const EMPTY_SLOT_ENERGY = 10;
/** 能量不足打不出招式时的踉跄回能 */
export const STAGGER_ENERGY = 6;
/** 守方能量归零时挨打的额外伤害 */
export const EXHAUST_BONUS_DAMAGE = 10;
export const CRIT_MULT = 1.5;
/** 闪避率上限：再灵活的选手也不该完全无法被击中 */
export const MAX_DODGE = 0.85;
/** 任何命中至少造成 1 点伤害 */
export const MIN_DAMAGE = 1;

export const STAT_MIN = 1;
export const STAT_MAX = 20;

export function derive(stats: Stats): Derived {
  const { str, agi, sta } = stats;
  const maxEnergy = 50 + sta * 5;
  return {
    maxHp: 100 + sta * 10,
    maxEnergy,
    initiative: 2 + Math.floor(agi / 3),
    damageMult: 1 + str * 0.05,
    armor: Math.floor(sta / 3),
    critChance: Math.min(0.03 + agi * 0.01, 0.35),
    dodgeBonus: Math.min(agi * 0.015, 0.25),
    roundRegen: Math.round(maxEnergy * 0.2) + sta,
  };
}

export function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}
