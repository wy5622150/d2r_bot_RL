import { cloneLoadout } from './fighters';
import { createRng, type Rng } from './rng';
import type { Emit, EventInput } from './resolve';
import { pickAttack, resolveAttack } from './resolve';
import { derive, MAX_ROUNDS } from './stats';
import type { FighterDef, FighterState, FightState, Loadout, RoundEvent, Side } from './types';
import { UNK } from './unknowns';

export function other(side: Side): Side {
  return side === 'player' ? 'opponent' : 'player';
}

export function fighterOf(state: FightState, side: Side): FighterState {
  return side === 'player' ? state.player : state.opponent;
}

function makeFighter(def: FighterDef, loadout?: Loadout): FighterState {
  const derived = derive(def.stats, def.health);
  return {
    def,
    derived,
    hp: derived.maxHp,
    energy: derived.maxEnergy,
    lostPhases: 0,
    loadout: cloneLoadout(loadout ?? def.loadout),
  };
}

export function createFight(
  playerDef: FighterDef,
  opponentDef: FighterDef,
  seed: number,
  playerLoadout?: Loadout,
): FightState {
  return {
    round: 0,
    player: makeFighter(playerDef, playerLoadout),
    opponent: makeFighter(opponentDef),
    // 一代没有先手值系统，谁先出手也没有公开规则；固定由玩家先手
    attacker: 'player',
    rng: seed >>> 0,
    over: false,
    result: null,
  };
}

function cloneFighter(f: FighterState): FighterState {
  return { ...f, loadout: cloneLoadout(f.loadout) };
}

export function cloneState(s: FightState): FightState {
  return {
    ...s,
    player: cloneFighter(s.player),
    opponent: cloneFighter(s.opponent),
    result: s.result ? { ...s.result } : null,
  };
}

/** 界面在回合间改了技能配置后写回状态 */
export function applyLoadout(state: FightState, side: Side, loadout: Loadout): FightState {
  const next = cloneState(state);
  fighterOf(next, side).loadout = cloneLoadout(loadout);
  return next;
}

/**
 * 模拟一个回合，返回该回合的完整事件流 + 新状态。
 *
 * 一代的回合结构：回合内有一个倒计时器，双方交替进行攻防阶段，最多打 20 个回合。
 * 计时器时长没有公开，所以这里用「每回合固定若干个阶段」近似 —— 该值在 unknowns.ts 里。
 *
 * 引擎一次性把回合算完（不是每帧驱动），渲染层只负责按时间线回放这条事件流。
 */
export function simulateRound(prev: FightState): { events: RoundEvent[]; state: FightState } {
  const state = cloneState(prev);
  const events: RoundEvent[] = [];
  if (state.over) return { events, state };

  const rng = createRng(state.rng);
  let tick = 0;
  const emit: Emit = (e: EventInput) => {
    events.push({
      ...e,
      tick: tick++,
      hp: { player: state.player.hp, opponent: state.opponent.hp },
      energy: { player: state.player.energy, opponent: state.opponent.energy },
    } as RoundEvent);
  };

  state.round += 1;

  emit({ type: 'round_start', round: state.round, text: `第 ${state.round} 回合 —— 开始` });

  for (let phase = 0; phase < UNK.phasesPerRound && !state.over; phase++) {
    performPhase(state, state.attacker, rng, emit);
    state.attacker = other(state.attacker);
    checkKo(state, emit);
  }

  if (!state.over) {
    emit({ type: 'round_end', round: state.round, text: `第 ${state.round} 回合结束` });
    if (state.round >= MAX_ROUNDS) judge(state);
  }

  state.rng = rng.state();
  return { events, state };
}

function performPhase(state: FightState, side: Side, rng: Rng, emit: Emit): void {
  const self = fighterOf(state, side);

  // 战斗中的体力回复：wiki 描述为「不断浮出的小额数字」，所以放在每个自己的阶段开始时
  self.energy = Math.min(self.derived.maxEnergy, self.energy + UNK.inFightRegen(self.derived.reg));

  if (self.lostPhases > 0) {
    self.lostPhases -= 1;
    emit({ type: 'skip', side, text: `${self.def.name} 还在从地上爬起来` });
    return;
  }

  const move = pickAttack(self, rng);

  // 体力不足以打出任何已装备的攻击技能
  if (!move) {
    const gain = Math.min(UNK.exhaustedPhaseRegen, self.derived.maxEnergy - self.energy);
    self.energy += gain;
    emit({
      type: 'exhausted',
      side,
      energyGain: gain,
      text: `${self.def.name} 体力见底，这一拍只能喘气`,
    });
    return;
  }

  resolveAttack(
    {
      attacker: self,
      defender: fighterOf(state, other(side)),
      attackerSide: side,
      defenderSide: other(side),
      rng,
      emit,
    },
    move.id,
  );
}

function checkKo(state: FightState, emit: Emit): void {
  const playerDown = state.player.hp <= 0;
  const opponentDown = state.opponent.hp <= 0;
  if (!playerDown && !opponentDown) return;

  const loser: Side | null = playerDown && opponentDown ? null : playerDown ? 'player' : 'opponent';
  emit({
    type: 'ko',
    side: loser ?? 'player',
    text:
      loser === null
        ? '双方同时倒地 —— 双 KO！'
        : `${fighterOf(state, loser).def.name} 倒下了 —— KO！`,
  });
  emit({ type: 'round_end', round: state.round, text: `第 ${state.round} 回合结束` });

  state.over = true;
  state.result = { winner: loser === null ? null : other(loser), method: 'ko', round: state.round };
}

/**
 * 20 回合内双方都没倒下 → 由系统判定。
 * 原版的判定公式没有公开，这里按剩余血量百分比比较（见 unknowns.ts 的 decisionRule）。
 */
export function judge(state: FightState): void {
  const pr = state.player.hp / state.player.derived.maxHp;
  const or = state.opponent.hp / state.opponent.derived.maxHp;
  const winner: Side | null = pr > or ? 'player' : or > pr ? 'opponent' : null;
  state.over = true;
  state.result = { winner, method: 'decision', round: state.round };
}

/** 一直打到分出结果。headless 批量模拟用。 */
export function simulateFight(initial: FightState): {
  rounds: RoundEvent[][];
  state: FightState;
} {
  let state = initial;
  const rounds: RoundEvent[][] = [];
  let guard = 0;
  while (!state.over && guard++ < MAX_ROUNDS + 2) {
    const r = simulateRound(state);
    rounds.push(r.events);
    state = r.state;
  }
  return { rounds, state };
}
