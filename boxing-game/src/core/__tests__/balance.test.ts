import { describe, expect, it } from 'vitest';
import { getOpponent, OPPONENTS, PLAYER } from '../fighters';
import { playFullFight } from '../match';
import type { FighterDef, Loadout } from '../types';

/**
 * 冒烟级的平衡断言 —— 只保证"配槽这件事有意义、每个对手都打得过"，
 * 不锁定具体胜率。数值调优留到 MVP 跑通之后再做，届时再把阈值收紧。
 */
const N = 200;

const NAIVE: Loadout = {
  offense: ['hook', 'cross', 'hook', 'cross'],
  defense: ['guard', 'guard', 'guard'],
};

const COUNTERS: Record<string, Loadout> = {
  carl: { offense: ['bodyshot', null, 'bodyshot', 'bodyshot'], defense: ['guard', 'slip', 'parry'] },
  ray: { offense: ['jab', 'breathe', 'haymaker', 'uppercut'], defense: ['parry', 'slip', 'parry'] },
  otto: {
    offense: ['uppercut', 'uppercut', 'breathe', 'uppercut'],
    defense: ['clinch', 'parry', 'slip'],
  },
};

function winRate(opponent: FighterDef, loadout: Loadout): number {
  let wins = 0;
  for (let seed = 1; seed <= N; seed++) {
    const { state } = playFullFight(PLAYER, opponent, seed * 2654435761, loadout);
    if (state.result?.winner === 'player') wins++;
  }
  return wins / N;
}

describe('平衡冒烟', () => {
  for (const opponent of OPPONENTS) {
    it(`${opponent.name} 存在打得过的配槽，且明显好过通用配槽`, () => {
      const naive = winRate(opponent, NAIVE);
      const countered = winRate(opponent, COUNTERS[opponent.id]!);
      expect(countered).toBeGreaterThan(0.55);
      expect(countered - naive).toBeGreaterThan(0.15);
    });
  }

  it('配槽是"针对性"的：打卡尔的配槽拿去打奥托会明显变差', () => {
    const vsCarl = winRate(getOpponent('carl'), COUNTERS['carl']!);
    const vsOtto = winRate(getOpponent('otto'), COUNTERS['carl']!);
    expect(vsCarl - vsOtto).toBeGreaterThan(0.3);
  });

  it('全空槽必输：不出手就赢不了', () => {
    const idle: Loadout = { offense: [null, null, null, null], defense: ['guard', 'guard', 'guard'] };
    for (const o of OPPONENTS) expect(winRate(o, idle)).toBe(0);
  });

  it('KO 和读分两种结局都会出现', () => {
    const methods = new Set<string>();
    for (const o of OPPONENTS) {
      for (let seed = 1; seed <= 80; seed++) {
        const { state } = playFullFight(PLAYER, o, seed * 7919, NAIVE);
        if (state.result) methods.add(state.result.method);
      }
    }
    expect(methods).toEqual(new Set(['ko', 'decision']));
  });
});
