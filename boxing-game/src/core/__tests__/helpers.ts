import { derive } from '../stats';
import type { FighterDef, FighterState, Loadout, Slot, Stats } from '../types';

export function defOf(
  id: string,
  stats: Stats,
  loadout: Loadout,
  extra: Partial<FighterDef> = {},
): FighterDef {
  return {
    id,
    name: id,
    color: 0xffffff,
    tagline: '',
    stats,
    style: 'manual',
    pool: [],
    loadout,
    ...extra,
  };
}

export function loadoutOf(offense: Slot[], defense: Slot[]): Loadout {
  return { offense, defense };
}

export function stateOf(def: FighterDef, patch: Partial<FighterState> = {}): FighterState {
  const derived = derive(def.stats);
  return {
    def,
    derived,
    hp: derived.maxHp,
    energy: derived.maxEnergy,
    offCursor: 0,
    stunned: false,
    knockedDown: false,
    loadout: { offense: [...def.loadout.offense], defense: [...def.loadout.defense] },
    ...patch,
  };
}

/** 全空槽的选手：不出手也不掉血，用来把某个机制单独隔离出来测 */
export const IDLE_LOADOUT: Loadout = {
  offense: [null, null, null, null],
  defense: [null, null, null],
};
