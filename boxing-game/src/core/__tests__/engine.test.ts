import { describe, expect, it } from 'vitest';
import { createFight, simulateFight, simulateRound } from '../engine';
import { OPPONENTS, PLAYER } from '../fighters';
import { ACTIONS_PER_ROUND, EMPTY_SLOT_ENERGY, ROUNDS } from '../stats';
import type { RoundEvent, Side } from '../types';
import { defOf, IDLE_LOADOUT, loadoutOf } from './helpers';

const carl = OPPONENTS[0]!;

function countActions(events: RoundEvent[], side: Side): number {
  return events.filter(
    (e) =>
      (e.type === 'attack' ||
        e.type === 'rest' ||
        e.type === 'empty_slot' ||
        e.type === 'exhausted' ||
        e.type === 'skip') &&
      e.side === side,
  ).length;
}

describe('确定性', () => {
  it('同 seed 同配槽 → 事件流逐条相等', () => {
    const a = simulateFight(createFight(PLAYER, carl, 12345));
    const b = simulateFight(createFight(PLAYER, carl, 12345));
    expect(JSON.stringify(a.rounds)).toBe(JSON.stringify(b.rounds));
    expect(a.state.result).toEqual(b.state.result);
  });

  it('不同 seed → 战报不同', () => {
    const a = simulateFight(createFight(PLAYER, carl, 1));
    const b = simulateFight(createFight(PLAYER, carl, 999));
    expect(JSON.stringify(a.rounds)).not.toBe(JSON.stringify(b.rounds));
  });
});

describe('先手值 initiative', () => {
  it('4 vs 2 时动作数正好是 2:1', () => {
    // 双方全空槽 → 不会掉血提前 KO，纯看动作分配
    const fast = defOf('fast', { str: 5, agi: 6, sta: 5 }, IDLE_LOADOUT); // initiative 4
    const slow = defOf('slow', { str: 5, agi: 1, sta: 5 }, IDLE_LOADOUT); // initiative 2
    const { rounds } = simulateFight(createFight(fast, slow, 7));
    const all = rounds.flat();
    expect(countActions(all, 'player')).toBe(ROUNDS * ACTIONS_PER_ROUND * (2 / 3));
    expect(countActions(all, 'opponent')).toBe(ROUNDS * ACTIONS_PER_ROUND * (1 / 3));
  });

  it('先手值高的一方先出手', () => {
    const slowPlayer = defOf('p', { str: 5, agi: 1, sta: 5 }, IDLE_LOADOUT);
    const fastFoe = defOf('o', { str: 5, agi: 9, sta: 5 }, IDLE_LOADOUT);
    const { events } = simulateRound(createFight(slowPlayer, fastFoe, 3));
    const first = events.find((e) => e.type === 'empty_slot');
    expect(first && 'side' in first ? first.side : null).toBe('opponent');
  });
});

describe('进攻槽', () => {
  it('空槽会回复能量并产生 empty_slot 事件', () => {
    const def = defOf('p', { str: 5, agi: 5, sta: 5 }, loadoutOf([null, null, null, null], []));
    const foe = defOf('o', { str: 5, agi: 5, sta: 5 }, IDLE_LOADOUT);
    const start = createFight(def, foe, 42);
    start.player.energy = 10; // 留出回能空间
    const { events, state } = simulateRound(start);
    const empties = events.filter((e) => e.type === 'empty_slot' && e.side === 'player');
    expect(empties.length).toBeGreaterThan(0);
    expect(state.player.energy).toBeGreaterThan(10);
    expect(empties[0]).toMatchObject({ energyGain: EMPTY_SLOT_ENERGY });
  });

  it('出招游标跨回合延续（不会每回合从第一个槽重来）', () => {
    const def = defOf('p', { str: 5, agi: 5, sta: 5 }, IDLE_LOADOUT);
    const foe = defOf('o', { str: 5, agi: 5, sta: 5 }, IDLE_LOADOUT);
    let state = createFight(def, foe, 5);
    const r1 = simulateRound(state);
    const afterR1 = r1.state.player.offCursor;
    expect(afterR1).toBe(countActions(r1.events, 'player'));
    state = r1.state;
    const r2 = simulateRound(state);
    expect(r2.state.player.offCursor).toBe(afterR1 + countActions(r2.events, 'player'));
  });

  it('能量不足时踉跄，不会打出招式', () => {
    const def = defOf('p', { str: 5, agi: 5, sta: 5 }, loadoutOf(['haymaker'], [null, null, null]));
    const foe = defOf('o', { str: 5, agi: 5, sta: 5 }, IDLE_LOADOUT);
    const start = createFight(def, foe, 11);
    start.player.energy = 1;
    const { events } = simulateRound(start);
    expect(events.some((e) => e.type === 'exhausted' && e.side === 'player')).toBe(true);
  });
});

describe('结束条件', () => {
  it('KO 后立即结束，不再产生后续动作事件', () => {
    const start = createFight(PLAYER, carl, 2024);
    start.opponent.hp = 3;
    const { events, state } = simulateRound(start);
    expect(state.over).toBe(true);
    expect(state.result?.method).toBe('ko');
    const koIdx = events.findIndex((e) => e.type === 'ko');
    expect(koIdx).toBeGreaterThanOrEqual(0);
    // KO 之后只允许有一个 round_end
    expect(events.slice(koIdx + 1).map((e) => e.type)).toEqual(['round_end']);
  });

  it('打满三回合按剩余血量百分比读分', () => {
    const a = defOf('p', { str: 5, agi: 5, sta: 5 }, IDLE_LOADOUT);
    const b = defOf('o', { str: 5, agi: 5, sta: 5 }, IDLE_LOADOUT);
    const start = createFight(a, b, 8);
    start.opponent.hp = start.opponent.derived.maxHp / 2;
    const { state } = simulateFight(start);
    expect(state.result).toEqual({ winner: 'player', method: 'decision', round: ROUNDS });
  });

  it('血量百分比相同 → 平局', () => {
    const a = defOf('p', { str: 5, agi: 5, sta: 5 }, IDLE_LOADOUT);
    const b = defOf('o', { str: 5, agi: 5, sta: 5 }, IDLE_LOADOUT);
    const { state } = simulateFight(createFight(a, b, 8));
    expect(state.result).toEqual({ winner: null, method: 'decision', round: ROUNDS });
  });
});

describe('事件快照', () => {
  it('每个事件都带双方血量/体力快照，且与最终状态吻合', () => {
    const { rounds, state } = simulateFight(createFight(PLAYER, carl, 777));
    const all = rounds.flat();
    expect(all.length).toBeGreaterThan(0);
    for (const e of all) {
      expect(e.hp.player).toBeGreaterThanOrEqual(0);
      expect(e.energy.opponent).toBeGreaterThanOrEqual(0);
      expect(typeof e.text).toBe('string');
    }
    const last = all[all.length - 1]!;
    expect(last.hp.player).toBe(state.player.hp);
    expect(last.hp.opponent).toBe(state.opponent.hp);
  });
});
