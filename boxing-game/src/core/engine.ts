import { cloneLoadout } from './fighters';
import { tryGetMove } from './moves';
import { createRng, type Rng } from './rng';
import type { Emit, EventInput } from './resolve';
import { resolveAttack } from './resolve';
import { ACTIONS_PER_ROUND, derive, EMPTY_SLOT_ENERGY, ROUNDS, STAGGER_ENERGY } from './stats';
import type { FighterDef, FighterState, FightState, Loadout, RoundEvent, Side } from './types';

export function other(side: Side): Side {
  return side === 'player' ? 'opponent' : 'player';
}

export function fighterOf(state: FightState, side: Side): FighterState {
  return side === 'player' ? state.player : state.opponent;
}

function makeFighter(def: FighterDef, loadout?: Loadout): FighterState {
  const derived = derive(def.stats);
  return {
    def,
    derived,
    hp: derived.maxHp,
    energy: derived.maxEnergy,
    offCursor: 0,
    stunned: false,
    knockedDown: false,
    loadout: cloneLoadout(loadout ?? def.loadout),
  };
}

export function createFight(
  playerDef: FighterDef,
  opponentDef: FighterDef,
  seed: number,
  playerLoadout?: Loadout,
): FightState {
  const player = makeFighter(playerDef, playerLoadout);
  const opponent = makeFighter(opponentDef);
  // 先手值高的一方先出手，相同则玩家先手
  const turn: Side = opponent.derived.initiative > player.derived.initiative ? 'opponent' : 'player';
  return {
    round: 0,
    player,
    opponent,
    turn,
    turnActionsLeft: (turn === 'player' ? player : opponent).derived.initiative,
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

/** 界面在回合间改了配槽后写回状态 */
export function applyLoadout(state: FightState, side: Side, loadout: Loadout): FightState {
  const next = cloneState(state);
  fighterOf(next, side).loadout = cloneLoadout(loadout);
  return next;
}

/**
 * 模拟一个回合，返回该回合的完整事件流 + 新状态。
 * 引擎一次性把回合算完（不是每帧驱动），渲染层只负责按时间线回放这条事件流：
 * 战斗结果因此与渲染完全解耦，同 seed 必定同结果。
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

  const regen: Record<Side, number> = { player: 0, opponent: 0 };
  if (state.round > 1) {
    for (const side of ['player', 'opponent'] as const) {
      const f = fighterOf(state, side);
      const before = f.energy;
      f.energy = Math.min(f.derived.maxEnergy, f.energy + f.derived.roundRegen);
      regen[side] = f.energy - before;
    }
  }
  emit({
    type: 'round_start',
    round: state.round,
    regen,
    text: `第 ${state.round} 回合 —— 开始`,
  });

  let budget = ACTIONS_PER_ROUND;
  while (budget > 0 && !state.over) {
    if (state.turnActionsLeft <= 0) {
      state.turn = other(state.turn);
      const f = fighterOf(state, state.turn);
      state.turnActionsLeft = f.derived.initiative;
      emit({
        type: 'turn_switch',
        side: state.turn,
        actions: state.turnActionsLeft,
        text: `${f.def.name} 抢到主动权，可以连打 ${state.turnActionsLeft} 拍`,
      });
    }

    performAction(state, state.turn, rng, emit);
    state.turnActionsLeft -= 1;
    budget -= 1;

    checkKo(state, emit);
  }

  if (!state.over) {
    emit({ type: 'round_end', round: state.round, text: `第 ${state.round} 回合结束` });
    if (state.round >= ROUNDS) {
      judge(state);
    }
  }

  state.rng = rng.state();
  return { events, state };
}

function performAction(state: FightState, side: Side, rng: Rng, emit: Emit): void {
  const self = fighterOf(state, side);

  if (self.knockedDown) {
    self.knockedDown = false;
    emit({
      type: 'skip',
      side,
      cause: 'knockdown',
      text: `${self.def.name} 正从地上爬起来，这一拍没了`,
    });
    return;
  }
  if (self.stunned) {
    self.stunned = false;
    emit({ type: 'skip', side, cause: 'stun', text: `${self.def.name} 还没缓过来，出不了手` });
    return;
  }

  const slots = self.loadout.offense;
  const slotId = slots.length > 0 ? (slots[self.offCursor % slots.length] ?? null) : null;
  self.offCursor += 1;
  const move = tryGetMove(slotId);

  // 空槽：不出手，换一口气（原作机制）
  if (!move) {
    const gain = Math.min(EMPTY_SLOT_ENERGY, self.derived.maxEnergy - self.energy);
    self.energy += gain;
    emit({
      type: 'empty_slot',
      side,
      energyGain: gain,
      text: `${self.def.name} 空出一拍调整节奏，回复 ${gain} 体力`,
    });
    return;
  }

  if (move.kind === 'rest') {
    const gain = Math.min(move.energyRestore ?? 0, self.derived.maxEnergy - self.energy);
    self.energy += gain;
    emit({
      type: 'rest',
      side,
      move: move.id,
      energyGain: gain,
      text: `${self.def.name} ${move.name}，回复 ${gain} 体力`,
    });
    return;
  }

  if (self.energy < move.energyCost) {
    const gain = Math.min(STAGGER_ENERGY, self.derived.maxEnergy - self.energy);
    self.energy += gain;
    emit({
      type: 'exhausted',
      side,
      energyGain: gain,
      text: `${self.def.name} 体力不够打出${move.name}，踉跄了一下`,
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

  // 反击有可能让双方同时倒下 —— 判双 KO 平局
  const loser: Side | null = playerDown && opponentDown ? null : playerDown ? 'player' : 'opponent';
  const text =
    loser === null
      ? '双方同时倒地 —— 双 KO！'
      : `${fighterOf(state, loser).def.name} 倒下了，数到十也没能起来 —— KO！`;

  emit({ type: 'ko', side: loser ?? 'player', text });
  emit({ type: 'round_end', round: state.round, text: `第 ${state.round} 回合结束` });

  state.over = true;
  state.result = { winner: loser === null ? null : other(loser), method: 'ko', round: state.round };
}

/** 打满回合后按剩余血量百分比读分 */
export function judge(state: FightState): void {
  const pr = state.player.hp / state.player.derived.maxHp;
  const or = state.opponent.hp / state.opponent.derived.maxHp;
  const winner: Side | null = pr > or ? 'player' : or > pr ? 'opponent' : null;
  state.over = true;
  state.result = { winner, method: 'decision', round: state.round };
}

/** 一直打到分出结果，返回每回合的事件流。headless 批量模拟/调平衡用。 */
export function simulateFight(initial: FightState): {
  rounds: RoundEvent[][];
  state: FightState;
} {
  let state = initial;
  const rounds: RoundEvent[][] = [];
  let guard = 0;
  while (!state.over && guard++ < ROUNDS + 2) {
    const r = simulateRound(state);
    rounds.push(r.events);
    state = r.state;
  }
  return { rounds, state };
}
