import { describe, expect, it } from 'vitest';
import { getMove } from '../moves';
import { createRng } from '../rng';
import type { Emit } from '../resolve';
import { pickDefense, resolveAttack } from '../resolve';
import { EXHAUST_BONUS_DAMAGE, MIN_DAMAGE } from '../stats';
import type { RoundEvent } from '../types';
import { defOf, IDLE_LOADOUT, loadoutOf, stateOf } from './helpers';

function run(
  seed: number,
  moveId: string,
  attackerPatch = {},
  defenderPatch = {},
  defenderDefense: (string | null)[] = [null, null, null],
) {
  const atkDef = defOf('atk', { str: 10, agi: 1, sta: 6 }, IDLE_LOADOUT);
  const defDef = defOf('def', { str: 5, agi: 1, sta: 6 }, loadoutOf([null], defenderDefense));
  const attacker = stateOf(atkDef, attackerPatch);
  const defender = stateOf(defDef, defenderPatch);
  const events: RoundEvent[] = [];
  const emit: Emit = (e) => {
    events.push({ ...e, tick: events.length, hp: { player: 0, opponent: 0 }, energy: { player: 0, opponent: 0 } } as RoundEvent);
  };
  resolveAttack(
    {
      attacker,
      defender,
      attackerSide: 'player',
      defenderSide: 'opponent',
      rng: createRng(seed),
      emit,
    },
    moveId,
  );
  return { events, attacker, defender };
}

describe('体力归零惩罚', () => {
  it('守方体力为 0 时挨打，额外 +10 伤害并被击倒', () => {
    let checked = 0;
    for (let seed = 1; seed <= 200; seed++) {
      const { events, defender } = run(seed, 'cross', {}, { energy: 0 });
      const hit = events.find((e) => e.type === 'hit');
      if (!hit || hit.type !== 'hit' || hit.crit) continue;

      // cross: 14 基础伤害 × 攻方 damageMult(1.5) = 21，减去守方 armor(2) = 19，再 +10
      const expected = Math.max(MIN_DAMAGE, Math.round(14 * 1.5) - 2) + EXHAUST_BONUS_DAMAGE;
      expect(hit.exhaustBonus).toBe(true);
      expect(hit.damage).toBe(expected);
      expect(events.some((e) => e.type === 'knockdown')).toBe(true);
      expect(defender.knockedDown).toBe(true);
      checked++;
      if (checked >= 5) break;
    }
    expect(checked).toBeGreaterThan(0);
  });

  it('守方体力充足时没有惩罚伤害', () => {
    for (let seed = 1; seed <= 200; seed++) {
      const { events } = run(seed, 'cross');
      const hit = events.find((e) => e.type === 'hit');
      if (!hit || hit.type !== 'hit' || hit.crit) continue;
      expect(hit.exhaustBonus).toBe(false);
      expect(hit.damage).toBe(Math.max(MIN_DAMAGE, Math.round(14 * 1.5) - 2));
      return;
    }
    throw new Error('200 个 seed 里一次都没命中，说明命中率算错了');
  });
});

describe('防守槽抽取', () => {
  it('同一招放的份数越多，被抽中的概率越高（3:1 ≈ 3 倍）', () => {
    const defender = stateOf(
      defOf('d', { str: 5, agi: 5, sta: 10 }, loadoutOf([null], ['guard', 'guard', 'guard'])),
    );
    const mixed = stateOf(
      defOf('d', { str: 5, agi: 5, sta: 10 }, loadoutOf([null], ['guard', 'slip', 'slip'])),
    );
    const rng = createRng(99);
    let allGuard = 0;
    let mixedGuard = 0;
    const N = 6000;
    for (let i = 0; i < N; i++) {
      if (pickDefense(defender, rng)?.id === 'guard') allGuard++;
      if (pickDefense(mixed, rng)?.id === 'guard') mixedGuard++;
    }
    expect(allGuard / N).toBeCloseTo(1, 2);
    expect(mixedGuard / N).toBeGreaterThan(0.28);
    expect(mixedGuard / N).toBeLessThan(0.39);
  });

  it('空防守槽会被抽中 → 完全不设防', () => {
    const defender = stateOf(
      defOf('d', { str: 5, agi: 5, sta: 10 }, loadoutOf([null], ['guard', null, null])),
    );
    const rng = createRng(5);
    let none = 0;
    for (let i = 0; i < 3000; i++) if (pickDefense(defender, rng) === null) none++;
    expect(none / 3000).toBeGreaterThan(0.6);
  });

  it('能量不够时防守失败', () => {
    const defender = stateOf(
      defOf('d', { str: 5, agi: 5, sta: 10 }, loadoutOf([null], ['parry', 'parry', 'parry'])),
      { energy: 1 },
    );
    expect(pickDefense(defender, createRng(1))).toBeNull();
  });
});

describe('防守效果', () => {
  it('格挡先按百分比减伤，再扣固定护甲', () => {
    for (let seed = 1; seed <= 300; seed++) {
      const { events } = run(seed, 'cross', {}, {}, ['guard', 'guard', 'guard']);
      const hit = events.find((e) => e.type === 'hit');
      if (!hit || hit.type !== 'hit' || hit.crit) continue;
      const raw = Math.round(14 * 1.5);
      expect(hit.damage).toBe(
        Math.max(MIN_DAMAGE, Math.round(raw * (1 - getMove('guard').blockPct!)) - 2),
      );
      expect(hit.blocked).toBe(raw - hit.damage);
      return;
    }
    throw new Error('没有采到一次非暴击命中');
  });

  it('招架会把挡下的伤害按比例反击回去', () => {
    for (let seed = 1; seed <= 300; seed++) {
      const { events, attacker } = run(seed, 'hook', {}, {}, ['parry', 'parry', 'parry']);
      const hit = events.find((e) => e.type === 'hit');
      const counter = events.find((e) => e.type === 'counter');
      if (!hit || hit.type !== 'hit' || !counter || counter.type !== 'counter') continue;
      expect(counter.damage).toBe(Math.max(MIN_DAMAGE, Math.round(hit.blocked * getMove('parry').counter!)));
      expect(attacker.hp).toBe(attacker.derived.maxHp - counter.damage);
      return;
    }
    throw new Error('没有采到一次招架反击');
  });

  it('闪避成功时不产生任何伤害', () => {
    for (let seed = 1; seed <= 300; seed++) {
      const { events, defender } = run(seed, 'cross', {}, {}, ['slip', 'slip', 'slip']);
      if (!events.some((e) => e.type === 'dodge')) continue;
      expect(events.some((e) => e.type === 'hit')).toBe(false);
      expect(defender.hp).toBe(defender.derived.maxHp);
      return;
    }
    throw new Error('没有采到一次闪避');
  });
});

describe('招式附加效果', () => {
  it('击腹会抽干守方体力', () => {
    for (let seed = 1; seed <= 300; seed++) {
      const { events, defender } = run(seed, 'bodyshot', {}, { energy: 60 });
      if (!events.some((e) => e.type === 'hit')) continue;
      expect(defender.energy).toBe(60 - getMove('bodyshot').energyDrain!);
      return;
    }
    throw new Error('没有采到一次击腹命中');
  });

  it('攻方要付出招式的能量消耗', () => {
    const { attacker } = run(1, 'haymaker', { energy: 50 });
    expect(attacker.energy).toBe(50 - getMove('haymaker').energyCost);
  });
});
