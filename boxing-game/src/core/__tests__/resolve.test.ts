import { describe, expect, it } from 'vitest';
import { getMove } from '../moves';
import { createRng } from '../rng';
import type { Emit } from '../resolve';
import { costOf, pickAttack, pickDefense, resolveAttack } from '../resolve';
import { EXHAUST_BONUS_DAMAGE, evalFormula } from '../stats';
import type { RoundEvent } from '../types';
import { defOf, stateOf } from './helpers';

const ATK = ['punch', 'high_punch', 'uppercut', 'block', 'dodge'];

function run(
  seed: number,
  moveId: string,
  defenderLoadout: string[],
  attackerPatch = {},
  defenderPatch = {},
) {
  const attacker = stateOf(defOf('atk', { str: 5, agi: 5, stm: 5 }, ATK), attackerPatch);
  const defender = stateOf(
    defOf('def', { str: 4, agi: 4, stm: 4 }, defenderLoadout),
    defenderPatch,
  );
  const events: RoundEvent[] = [];
  const emit: Emit = (e) => {
    events.push({
      ...e,
      tick: events.length,
      hp: { player: 0, opponent: 0 },
      energy: { player: 0, opponent: 0 },
    } as RoundEvent);
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

describe('体力消耗', () => {
  it('按原版公式 base + 系数×STR 扣体力', () => {
    const f = stateOf(defOf('a', { str: 5, agi: 5, stm: 5 }, ATK));
    // uppercut 体力 1+1.5(str) → STR 5 = 8.5
    expect(costOf(getMove('uppercut'), f)).toBeCloseTo(8.5, 10);
    // block 体力 0.5+0.3(str) → 2
    expect(costOf(getMove('block'), f)).toBeCloseTo(2, 10);
  });

  it('攻方要付出技能的体力消耗', () => {
    const { attacker } = run(1, 'uppercut', ['block'], { energy: 50 });
    expect(attacker.energy).toBeCloseTo(50 - 8.5, 10);
  });
});

describe('抽招', () => {
  it('攻击阶段只抽得起的攻击技能，抽不起就返回 null', () => {
    const broke = stateOf(defOf('a', { str: 5, agi: 5, stm: 5 }, ATK), { energy: 0 });
    expect(pickAttack(broke, createRng(1))).toBeNull();

    // 体力 5 时只付得起 punch（5），付不起 high_punch（8）与 uppercut（8.5）
    const low = stateOf(defOf('a', { str: 5, agi: 5, stm: 5 }, ATK), { energy: 5 });
    for (let s = 1; s <= 30; s++) expect(pickAttack(low, createRng(s))?.id).toBe('punch');
  });

  it('装备里同一技能放多份 → 被抽中的概率成比例提高', () => {
    const one = stateOf(defOf('a', { str: 1, agi: 1, stm: 1 }, ['punch', 'kick', 'kick', 'kick', 'block']));
    const rng = createRng(7);
    let punches = 0;
    const N = 6000;
    for (let i = 0; i < N; i++) if (pickAttack(one, rng)?.id === 'punch') punches++;
    expect(punches / N).toBeGreaterThan(0.2);
    expect(punches / N).toBeLessThan(0.3);
  });

  it('防守阶段从已装备的防守技能里抽；体力不足则防守失败', () => {
    const f = stateOf(defOf('d', { str: 5, agi: 5, stm: 5 }, ['punch', 'block', 'dodge']));
    const ids = new Set<string>();
    const rng = createRng(3);
    for (let i = 0; i < 200; i++) {
      const m = pickDefense(f, rng);
      if (m) ids.add(m.id);
    }
    expect(ids).toEqual(new Set(['block', 'dodge']));

    const broke = stateOf(defOf('d', { str: 5, agi: 5, stm: 5 }, ['punch', 'block']), { energy: 0 });
    expect(pickDefense(broke, createRng(1))).toBeNull();
  });

  it('没装防守技能 → 完全不设防', () => {
    const f = stateOf(defOf('d', { str: 5, agi: 5, stm: 5 }, ['punch', 'kick']));
    expect(pickDefense(f, createRng(1))).toBeNull();
  });
});

describe('攻防结算', () => {
  it('命中伤害 = round(base + 系数×STR) − ARM，下限 1', () => {
    for (let seed = 1; seed <= 300; seed++) {
      const { events } = run(seed, 'uppercut', ['punch']); // 守方不带防守技能
      const hit = events.find((e) => e.type === 'hit');
      if (!hit || hit.type !== 'hit') continue;
      // uppercut 伤害 1.5+1.7(str)，STR 5 → 10；守方 STM 4 → ARM 5.2
      const raw = Math.round(evalFormula(getMove('uppercut').damage!, 5));
      expect(raw).toBe(10);
      expect(hit.damage).toBe(Math.max(1, Math.round(raw - 5.2)));
      expect(hit.exhaustBonus).toBe(false);
      return;
    }
    throw new Error('300 个 seed 里一次都没命中');
  });

  it('闪避成功则完全免伤', () => {
    for (let seed = 1; seed <= 300; seed++) {
      const { events, defender } = run(seed, 'punch', ['dodge']);
      if (!events.some((e) => e.type === 'dodge')) continue;
      expect(events.some((e) => e.type === 'hit')).toBe(false);
      expect(defender.hp).toBe(defender.derived.maxHp);
      return;
    }
    throw new Error('没有采到一次闪避');
  });

  it('格挡会减少伤害（同 seed 下带格挡比不带挨得少）', () => {
    let compared = 0;
    for (let seed = 1; seed <= 400 && compared < 3; seed++) {
      const bare = run(seed, 'uppercut', ['punch']).events.find((e) => e.type === 'hit');
      const blocked = run(seed, 'uppercut', ['block']).events.find((e) => e.type === 'hit');
      if (!bare || bare.type !== 'hit' || !blocked || blocked.type !== 'hit') continue;
      expect(blocked.damage).toBeLessThan(bare.damage);
      expect(blocked.blocked).toBeGreaterThan(0);
      compared++;
    }
    expect(compared).toBeGreaterThan(0);
  });
});

describe('体力归零惩罚（原版已确证）', () => {
  it('守方体力为 0 时挨打额外 +10 伤害并被击倒', () => {
    let checked = 0;
    for (let seed = 1; seed <= 300 && checked < 3; seed++) {
      const { events, defender } = run(seed, 'punch', ['punch'], {}, { energy: 0 });
      const hit = events.find((e) => e.type === 'hit');
      if (!hit || hit.type !== 'hit') continue;
      // punch 伤害 1+0.7×5 = 4.5 → round 5；ARM 5.2 → max(1, round(5-5.2)) = 1；再 +10
      expect(hit.exhaustBonus).toBe(true);
      expect(hit.damage).toBe(1 + EXHAUST_BONUS_DAMAGE);
      expect(events.some((e) => e.type === 'knockdown')).toBe(true);
      expect(defender.lostPhases).toBeGreaterThan(0);
      checked++;
    }
    expect(checked).toBeGreaterThan(0);
  });
});
