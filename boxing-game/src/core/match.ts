import { aiReconfigure } from './ai';
import { applyLoadout, createFight, simulateRound } from './engine';
import { createRng } from './rng';
import { MAX_ROUNDS } from './stats';
import type { FighterDef, FightState, Loadout, RoundEvent } from './types';

/**
 * 一场比赛的推进逻辑：界面和 headless 批量模拟共用这一份。
 * 每个回合开始前，对手按自己的风格重新选技能（玩家那边由界面负责）。
 */
export function advanceRound(state: FightState): { events: RoundEvent[]; state: FightState } {
  if (state.over) return { events: [], state };

  let next = state;
  if (state.round >= 1) {
    // 用一条与战斗 rng 错开的流，避免 AI 的决策和当回合的判定共用同一批随机数
    const aiRng = createRng((state.rng ^ 0x9e3779b9) >>> 0);
    const loadout = aiReconfigure(next.opponent, next.player, aiRng);
    next = applyLoadout(next, 'opponent', loadout);
  }
  return simulateRound(next);
}

export interface MatchLog {
  rounds: RoundEvent[][];
  state: FightState;
}

/** 一路打到分出结果（对手全程自动换技能，玩家沿用同一套配置）。 */
export function playFullFight(
  playerDef: FighterDef,
  opponentDef: FighterDef,
  seed: number,
  playerLoadout?: Loadout,
): MatchLog {
  let state = createFight(playerDef, opponentDef, seed, playerLoadout);
  const rounds: RoundEvent[][] = [];
  let guard = 0;
  while (!state.over && guard++ < MAX_ROUNDS + 2) {
    const r = advanceRound(state);
    rounds.push(r.events);
    state = r.state;
  }
  return { rounds, state };
}
