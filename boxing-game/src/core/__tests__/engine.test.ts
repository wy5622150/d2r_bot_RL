import { describe, expect, it } from 'vitest';
import { validateLoadout } from '../ai';
import { createFight, simulateFight, simulateRound } from '../engine';
import { OPPONENTS, PLAYER } from '../fighters';
import { playFullFight } from '../match';
import { ABILITY_SLOTS_MAX, MAX_ROUNDS } from '../stats';
import type { RoundEvent, Side } from '../types';
import { UNK } from '../unknowns';
import { defOf } from './helpers';

const silver = OPPONENTS[0]!;
const bobo = OPPONENTS[1]!;

function phasesOf(events: RoundEvent[], side: Side): number {
  return events.filter(
    (e) => (e.type === 'attack' || e.type === 'exhausted' || e.type === 'skip') && e.side === side,
  ).length;
}

describe('确定性', () => {
  it('同 seed 同配置 → 事件流逐条相等', () => {
    const a = playFullFight(PLAYER, bobo, 12345);
    const b = playFullFight(PLAYER, bobo, 12345);
    expect(JSON.stringify(a.rounds)).toBe(JSON.stringify(b.rounds));
    expect(a.state.result).toEqual(b.state.result);
  });

  it('不同 seed → 战报不同', () => {
    const a = playFullFight(PLAYER, bobo, 1);
    const b = playFullFight(PLAYER, bobo, 999);
    expect(JSON.stringify(a.rounds)).not.toBe(JSON.stringify(b.rounds));
  });
});

describe('回合结构', () => {
  it('双方严格交替出手，没有先手值这回事', () => {
    const { events } = simulateRound(createFight(PLAYER, silver, 7));
    const sides = events
      .filter((e) => e.type === 'attack' || e.type === 'exhausted' || e.type === 'skip')
      .map((e) => (e.type === 'attack' || e.type === 'exhausted' || e.type === 'skip' ? e.side : ''));
    expect(sides.length).toBe(UNK.phasesPerRound);
    for (let i = 1; i < sides.length; i++) expect(sides[i]).not.toBe(sides[i - 1]);
  });

  it('一个回合的阶段数在双方之间平分', () => {
    const { events } = simulateRound(createFight(PLAYER, silver, 11));
    expect(phasesOf(events, 'player') + phasesOf(events, 'opponent')).toBe(UNK.phasesPerRound);
  });

  it('最多打 20 个回合', () => {
    // 双方都没有攻击技能 → 谁也打不死谁，必然走到读分
    const pacifist = defOf('a', { str: 5, agi: 5, stm: 5 }, ['block', 'dodge']);
    const pacifist2 = defOf('b', { str: 5, agi: 5, stm: 5 }, ['block', 'dodge']);
    const { rounds, state } = simulateFight(createFight(pacifist, pacifist2, 5));
    expect(rounds.length).toBe(MAX_ROUNDS);
    expect(state.round).toBe(MAX_ROUNDS);
    expect(state.result?.method).toBe('decision');
  });
});

describe('结束条件', () => {
  it('KO 后立即结束，之后只剩一个 round_end', () => {
    const start = createFight(PLAYER, bobo, 2024);
    start.opponent.hp = 1;
    const { events, state } = simulateRound(start);
    expect(state.over).toBe(true);
    expect(state.result?.method).toBe('ko');
    expect(state.result?.winner).toBe('player');
    const koIdx = events.findIndex((e) => e.type === 'ko');
    expect(koIdx).toBeGreaterThanOrEqual(0);
    expect(events.slice(koIdx + 1).map((e) => e.type)).toEqual(['round_end']);
  });

  it('打满 20 回合按剩余血量百分比读分', () => {
    const a = defOf('a', { str: 5, agi: 5, stm: 5 }, ['block', 'dodge']);
    const b = defOf('b', { str: 5, agi: 5, stm: 5 }, ['block', 'dodge']);
    const start = createFight(a, b, 8);
    start.opponent.hp = start.opponent.derived.maxHp / 2;
    const { state } = simulateFight(start);
    expect(state.result).toEqual({ winner: 'player', method: 'decision', round: MAX_ROUNDS });
  });

  it('血量百分比相同 → 平局', () => {
    const a = defOf('a', { str: 5, agi: 5, stm: 5 }, ['block', 'dodge']);
    const b = defOf('b', { str: 5, agi: 5, stm: 5 }, ['block', 'dodge']);
    const { state } = simulateFight(createFight(a, b, 8));
    expect(state.result).toEqual({ winner: null, method: 'decision', round: MAX_ROUNDS });
  });
});

describe('体力耗尽', () => {
  it('打不出任何已装备的攻击技能时，该阶段只能喘气回体力', () => {
    const start = createFight(PLAYER, silver, 42);
    start.player.energy = 0;
    const { events } = simulateRound(start);
    const gasp = events.find((e) => e.type === 'exhausted' && e.side === 'player');
    expect(gasp).toBeDefined();
    expect(gasp && gasp.type === 'exhausted' ? gasp.energyGain : 0).toBeGreaterThan(0);
  });
});

describe('技能配置校验', () => {
  it('必须正好装备 5 个技能（一代不允许留空）', () => {
    expect(validateLoadout(['punch', 'block'], PLAYER.pool)).toContain(`${ABILITY_SLOTS_MAX}`);
    expect(validateLoadout(PLAYER.loadout, PLAYER.pool)).toBeNull();
  });

  it('至少要带一个攻击技能', () => {
    const allDefense = ['block', 'dodge', 'block', 'dodge', 'block'];
    expect(validateLoadout(allDefense, PLAYER.pool)).toContain('攻击技能');
  });

  it('不能带池子外的技能', () => {
    expect(validateLoadout(['punch', 'punch', 'punch', 'punch', 'nope'], PLAYER.pool)).toContain(
      'nope',
    );
  });
});

describe('事件快照', () => {
  it('每个事件都带双方血量/体力快照，末条与最终状态一致', () => {
    const { rounds, state } = playFullFight(PLAYER, bobo, 777);
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
