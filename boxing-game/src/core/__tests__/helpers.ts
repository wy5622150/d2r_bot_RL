import { derive } from '../stats';
import type { FighterDef, FighterState, Loadout, Stats } from '../types';

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
    health: 1,
    style: 'manual',
    pool: [...loadout],
    loadout,
    ...extra,
  };
}

export function stateOf(def: FighterDef, patch: Partial<FighterState> = {}): FighterState {
  const derived = derive(def.stats, def.health);
  return {
    def,
    derived,
    hp: derived.maxHp,
    energy: derived.maxEnergy,
    lostPhases: 0,
    loadout: [...def.loadout],
    ...patch,
  };
}
