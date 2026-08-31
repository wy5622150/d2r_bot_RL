import { create } from 'zustand';
import { applyLoadout, createFight } from '../core/engine';
import { cloneLoadout, getOpponent, OPPONENTS, PLAYER } from '../core/fighters';
import { advanceRound } from '../core/match';
import { ABILITY_SLOTS_MAX } from '../core/stats';
import type { FightState, Loadout, RoundEvent, Side } from '../core/types';
import { EventBus } from '../game/EventBus';

export type Screen = 'select' | 'loadout' | 'fight' | 'result';
/** 战斗内部的小状态机：等待开打 / 正在播放 / 回合间配槽 / 已结束 */
export type Phase = 'ready' | 'playing' | 'between' | 'over';

interface Hud {
  hp: Record<Side, number>;
  energy: Record<Side, number>;
}

interface UiState {
  screen: Screen;
  opponentId: string;
  loadout: Loadout;
  fight: FightState | null;
  phase: Phase;
  hud: Hud;
  log: string[];
  speed: number;

  chooseOpponent: (id: string) => void;
  backToSelect: () => void;
  setSlot: (index: number, move: string) => void;
  resetLoadout: () => void;
  startFight: () => void;
  playNextRound: () => void;
  onEventShown: (event: RoundEvent) => void;
  onRoundDone: () => void;
  setSpeed: (speed: number) => void;
  rematch: () => void;
}

const emptyHud = (): Hud => ({
  hp: { player: 0, opponent: 0 },
  energy: { player: 0, opponent: 0 },
});

function hudFrom(state: FightState): Hud {
  return {
    hp: { player: state.player.hp, opponent: state.opponent.hp },
    energy: { player: state.player.energy, opponent: state.opponent.energy },
  };
}

export const useGame = create<UiState>((set, get) => ({
  screen: 'select',
  opponentId: OPPONENTS[0]!.id,
  loadout: cloneLoadout(PLAYER.loadout),
  fight: null,
  phase: 'ready',
  hud: emptyHud(),
  log: [],
  speed: 1,

  chooseOpponent: (id) => set({ opponentId: id, screen: 'loadout' }),
  backToSelect: () => set({ screen: 'select', fight: null, log: [], phase: 'ready' }),

  setSlot: (index, move) =>
    set((s) => {
      const next = cloneLoadout(s.loadout);
      while (next.length < ABILITY_SLOTS_MAX) next.push(move);
      next[index] = move;
      return { loadout: next.slice(0, ABILITY_SLOTS_MAX) };
    }),

  resetLoadout: () => set({ loadout: cloneLoadout(PLAYER.loadout) }),

  startFight: () => {
    const { opponentId, loadout } = get();
    const opponent = getOpponent(opponentId);
    const fight = createFight(PLAYER, opponent, Date.now() >>> 0, loadout);
    // 只负责建状态；等 Phaser 场景就绪后由 FightScreen 发 setup 并起第一个回合
    set({ fight, screen: 'fight', phase: 'ready', log: [], hud: hudFrom(fight) });
  },

  /** 把玩家最新的配槽写回状态，算下一个回合，然后交给 Phaser 播放 */
  playNextRound: () => {
    const { fight, loadout } = get();
    if (!fight || fight.over) return;
    const withLoadout = applyLoadout(fight, 'player', loadout);
    const { events, state } = advanceRound(withLoadout);
    set({ fight: state, phase: 'playing' });
    EventBus.emit('fight:play-round', events);
  },

  onEventShown: (event) =>
    set((s) => ({
      hud: { hp: event.hp, energy: event.energy },
      log: [...s.log.slice(-60), event.text],
    })),

  onRoundDone: () => {
    const { fight } = get();
    if (!fight) return;
    if (fight.over) {
      set({ phase: 'over', screen: 'result' });
    } else {
      // 回合之间暂停，让玩家针对刚才打的内容重新配槽
      set({ phase: 'between' });
    }
  },

  setSpeed: (speed) => {
    set({ speed });
    EventBus.emit('fight:set-speed', speed);
  },

  rematch: () =>
    set({ screen: 'loadout', fight: null, phase: 'ready', log: [], hud: emptyHud() }),
}));
