import type { AccuracyFormula, Derived, StatFormula, Stats } from './types';

/**
 * 一代已确证的战斗常量与派生公式。
 * 这里**只允许**出现有来源的数字；未公开的部分一律去 unknowns.ts 取。
 * 出处见 data/punch-club-source.md 的「B. Combat formulas / flow」。
 */

/** 一场比赛最多 20 回合（wiki.gg Fighting / Wikipedia） */
export const MAX_ROUNDS = 20;

/** 一代是一组共享技能槽，最多 5 个（Guides_and_Tips_(Punch_Club_1)） */
export const ABILITY_SLOTS_MAX = 5;

/**
 * 体力归零时挨打的额外伤害并被击倒
 * （Guides_and_Tips_(Punch_Club_1)/Game_Mechanics）
 */
export const EXHAUST_BONUS_DAMAGE = 10;

/**
 * 体力上限 100。
 * 注意：这条来自 Steam 社区攻略（id=598795240），作者本人声明攻略部分内容是推测，
 * 不是 wiki 的正式公式 —— 是本文件里唯一一个来源等级较低的数字。
 */
export const MAX_ENERGY = 100;

export const STAT_MIN = 1;
export const STAT_MAX = 20;

/**
 * baseHP = 38 + 6·STR + 3·AGI + 8·STM + 16·min(STR,AGI,STM)
 * HP     = baseHP/2 + baseHP·Health/2
 * ACC    = 3·AGI / (STR+AGI+STM)      —— 招式命中公式里的 Hit%
 * REG    = 5 + STM×1.5                 —— 属性页显示值
 * ARM    = STM×1.3
 */
export function derive(stats: Stats, health = 1): Derived {
  const { str, agi, stm } = stats;
  const baseHp = 38 + 6 * str + 3 * agi + 8 * stm + 16 * Math.min(str, agi, stm);
  return {
    baseHp,
    maxHp: baseHp / 2 + (baseHp * health) / 2,
    maxEnergy: MAX_ENERGY,
    acc: (3 * agi) / (str + agi + stm),
    reg: 5 + stm * 1.5,
    arm: stm * 1.3,
  };
}

/** 招式数值：`base + perStr × STR`，例 Punch 伤害 `1+0.7(str)` */
export function evalFormula(f: StatFormula, str: number): number {
  return f.base + f.perStr * str;
}

/**
 * 命中率：`base + perAcc × ACC`，例 Punch 命中 `70+20(Hit%)`。
 * 返回 0..1 的概率（原始公式是百分数）。
 */
export function evalAccuracy(a: AccuracyFormula, acc: number): number {
  return clamp((a.base + a.perAcc * acc) / 100, 0, 1);
}

export function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

/** 把 `1+0.7(str)` 这样的原始字符串还原成可读形式，界面上直接显示原版写法 */
export function formulaText(f: StatFormula): string {
  return `${f.base}+${f.perStr}(str)`;
}

export function accuracyText(a: AccuracyFormula): string {
  return `${a.base}+${a.perAcc}(Hit%)`;
}
