import { describe, expect, it } from 'vitest';
import { getMove, MOVES } from '../moves';
import { derive, evalAccuracy, evalFormula, MAX_ENERGY } from '../stats';

/**
 * 这些断言直接钉住 data/punch-club-source.md 里逐字转录的原版公式。
 * 任何一条挂掉，要么是转录被改坏了，要么是有人偷偷"调平衡"。
 */
describe('原版派生公式', () => {
  it('baseHP = 38 + 6·STR + 3·AGI + 8·STM + 16·min(STR,AGI,STM)', () => {
    // 5/5/5 → 38 + 30 + 15 + 40 + 80 = 203
    expect(derive({ str: 5, agi: 5, stm: 5 }).baseHp).toBe(203);
    // 8/5/3 → 38 + 48 + 15 + 24 + 48 = 173
    expect(derive({ str: 8, agi: 5, stm: 3 }).baseHp).toBe(173);
  });

  it('HP = baseHP/2 + baseHP·Health/2，满健康时等于 baseHP', () => {
    expect(derive({ str: 5, agi: 5, stm: 5 }, 1).maxHp).toBe(203);
    // 半健康 → 203/2 + 203*0.5/2 = 101.5 + 50.75
    expect(derive({ str: 5, agi: 5, stm: 5 }, 0.5).maxHp).toBeCloseTo(152.25, 5);
  });

  it('ACC = 3·AGI / (STR+AGI+STM)', () => {
    expect(derive({ str: 5, agi: 5, stm: 5 }).acc).toBe(1);
    expect(derive({ str: 8, agi: 5, stm: 3 }).acc).toBeCloseTo(15 / 16, 10);
  });

  it('REG = 5 + STM×1.5，ARM = STM×1.3', () => {
    const d = derive({ str: 5, agi: 5, stm: 4 });
    expect(d.reg).toBeCloseTo(11, 10);
    expect(d.arm).toBeCloseTo(5.2, 10);
  });

  it('体力上限固定 100', () => {
    expect(derive({ str: 1, agi: 1, stm: 1 }).maxEnergy).toBe(MAX_ENERGY);
    expect(derive({ str: 20, agi: 20, stm: 20 }).maxEnergy).toBe(MAX_ENERGY);
  });
});

describe('招式数值转录', () => {
  it('Punch = 伤害 1+0.7(str) / 体力 0+1(str) / 命中 70+20(Hit%)', () => {
    const punch = getMove('punch');
    expect(punch.damage).toEqual({ base: 1, perStr: 0.7 });
    expect(punch.energyCost).toEqual({ base: 0, perStr: 1 });
    expect(punch.accuracy).toEqual({ base: 70, perAcc: 20 });
    // STR 5 → 1 + 3.5 = 4.5；体力 5
    expect(evalFormula(punch.damage!, 5)).toBeCloseTo(4.5, 10);
    expect(evalFormula(punch.energyCost, 5)).toBe(5);
    // ACC 1 → 70 + 20 = 90%
    expect(evalAccuracy(punch.accuracy!, 1)).toBeCloseTo(0.9, 10);
    // ACC 0.5 → 70 + 10 = 80%
    expect(evalAccuracy(punch.accuracy!, 0.5)).toBeCloseTo(0.8, 10);
  });

  it('Backhand High Punch 是最吃敏捷的一招：20+60(Hit%)', () => {
    const m = getMove('backhand_high_punch');
    expect(evalAccuracy(m.accuracy!, 0)).toBeCloseTo(0.2, 10);
    expect(evalAccuracy(m.accuracy!, 1)).toBeCloseTo(0.8, 10);
  });

  it('每个技能都带出处；攻击技能三条公式齐全，防守技能有体力公式', () => {
    for (const m of MOVES) {
      expect(m.sourceUrl).toMatch(/^https:\/\//);
      expect(m.energyCost).toBeDefined();
      if (m.kind === 'attack') {
        expect(m.damage, `${m.nameEn} 缺伤害公式`).toBeDefined();
        expect(m.accuracy, `${m.nameEn} 缺命中公式`).toBeDefined();
      } else {
        expect(m.defenseKind, `${m.nameEn} 缺防守形态`).toBeDefined();
      }
    }
  });

  it('资料里记录的源内部矛盾被保留下来，没有被"修正"掉', () => {
    expect(getMove('low_kick').conflict).toContain('71..80');
    expect(getMove('cutthroat').conflict).toContain('Karate Chop');
  });
});
