import type { FighterDef, Loadout } from './types';

const loadout = (offense: (string | null)[], defense: (string | null)[]): Loadout => ({
  offense,
  defense,
});

/** 玩家：属性均衡，招式全解锁，胜负完全取决于配槽 */
export const PLAYER: FighterDef = {
  id: 'player',
  name: '你',
  color: 0x4fc3f7,
  tagline: '均衡型。没有短板，也没有强项——赢面全在配槽上。',
  stats: { str: 9, agi: 8, sta: 8 },
  style: 'manual',
  pool: [
    'jab',
    'cross',
    'hook',
    'bodyshot',
    'overhand',
    'uppercut',
    'haymaker',
    'guard',
    'clinch',
    'slip',
    'parry',
    'breathe',
  ],
  loadout: loadout(['jab', 'cross', 'hook', 'breathe'], ['guard', 'slip', 'parry']),
};

export const OPPONENTS: readonly FighterDef[] = [
  {
    id: 'carl',
    name: '铁锤 · 卡尔',
    color: 0xef5350,
    tagline: '压迫型。拳重、先手少、体力烧得快——熬过他的前两回合就是你的。',
    stats: { str: 13, agi: 4, sta: 7 },
    style: 'pressure',
    pool: ['cross', 'hook', 'overhand', 'haymaker', 'bodyshot', 'guard', 'clinch', 'breathe'],
    loadout: loadout(['overhand', 'hook', 'haymaker', 'cross'], ['guard', 'guard', 'clinch']),
  },
  {
    id: 'ray',
    name: '游鱼 · 雷',
    color: 0x66bb6a,
    tagline: '游斗型。先手极多、闪避极高，用刺拳把你磨到判定。别把体力浪费在打空上。',
    stats: { str: 6, agi: 11, sta: 6 },
    style: 'outboxer',
    pool: ['jab', 'cross', 'bodyshot', 'uppercut', 'slip', 'parry', 'guard', 'breathe'],
    loadout: loadout(['jab', 'jab', 'cross', 'breathe'], ['slip', 'slip', 'parry']),
  },
  {
    id: 'otto',
    name: '石墙 · 奥托',
    color: 0xffa726,
    tagline: '铁壁型。血厚、减伤高、体力深，专门拖到读分。硬碰硬是打不穿的。',
    stats: { str: 8, agi: 4, sta: 12 },
    style: 'wall',
    pool: ['cross', 'hook', 'bodyshot', 'uppercut', 'guard', 'clinch', 'parry', 'breathe'],
    loadout: loadout(['cross', 'hook', 'breathe', null], ['guard', 'clinch', 'parry']),
  },
];

export function getOpponent(id: string): FighterDef {
  const o = OPPONENTS.find((f) => f.id === id);
  if (!o) throw new Error(`未知对手: ${id}`);
  return o;
}

/** 深拷贝一份配槽，避免界面直接改到常量上 */
export function cloneLoadout(l: Loadout): Loadout {
  return { offense: [...l.offense], defense: [...l.defense] };
}
