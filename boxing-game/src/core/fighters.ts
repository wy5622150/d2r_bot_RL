import { ABILITY_SLOTS_MAX } from './stats';
import type { FighterDef, Loadout } from './types';

/**
 * 选手数据。
 *
 * 属性值凡是标了 sourceUrl 的都逐字来自 data/punch-club-source.md；
 * 技能配置则只有一部分有出处 —— 一代没有公开过对手的完整 5 技能装备表，
 * 每个选手的 sourceNote 会写清楚哪几个技能是资料里记载的、哪些是从 Basic 池里补的。
 */

/** 一代的玩家属性是自己练出来的，没有「官方玩家属性」这回事 */
export const PLAYER: FighterDef = {
  id: 'player',
  name: '你',
  color: 0x4fc3f7,
  tagline: '三维均衡的新人。一代里玩家的属性是练出来的，这里取一个中庸的起手。',
  stats: { str: 5, agi: 5, stm: 5 },
  health: 1,
  style: 'manual',
  pool: [
    'punch',
    'kick',
    'high_punch',
    'high_kick',
    'low_kick',
    'uppercut',
    'crosspunch',
    'backhand_punch',
    'backhand_high_punch',
    'karate_chop',
    'cutthroat',
    'block',
    'dodge',
  ],
  loadout: ['punch', 'high_punch', 'uppercut', 'block', 'dodge'],
  sourceNote: '属性 5/5/5 不是原版数据，是为 MVP 选的中庸起手值（一代玩家属性由训练决定）。',
};

export const OPPONENTS: readonly FighterDef[] = [
  {
    id: 'silver',
    name: 'Silver',
    color: 0x9aa4b2,
    tagline: '故事里的第一场对练。属性与你完全相同 —— 拼的纯粹是技能选择。',
    stats: { str: 5, agi: 5, stm: 5 },
    health: 1,
    style: 'balanced',
    pool: ['punch', 'kick', 'high_punch', 'uppercut', 'block', 'dodge'],
    loadout: ['punch', 'kick', 'high_punch', 'block', 'dodge'],
    sourceUrl: 'https://punch-club.fandom.com/ru/wiki/%D0%A1%D0%B8%D0%BB%D1%8C%D0%B2%D0%B5%D1%80',
    sourceNote:
      '属性 5/5/5 来自俄语 Fandom 的首次对练记录（该页说他战术中性，但没有给出稳定的技能列表）。技能配置由 Basic 池补齐。',
  },
  {
    id: 'big_bobo',
    name: 'Big Bobo',
    color: 0xef5350,
    tagline: '力量 8、耐力只有 3。拳很重，但体力池浅得可怕 —— 熬过他的前几拳。',
    stats: { str: 8, agi: 5, stm: 3 },
    health: 1,
    style: 'aggressive',
    pool: ['punch', 'high_punch', 'crosspunch', 'backhand_punch', 'block', 'dodge'],
    loadout: ['high_punch', 'punch', 'crosspunch', 'block', 'dodge'],
    sourceUrl: 'https://punch-club.fandom.com/ru/wiki/%D0%91%D0%BE%D0%BB%D1%8C%D1%88%D0%BE%D0%B9_%D0%91%D0%BE%D0%B1%D0%B0',
    sourceNote:
      '属性 8/5/3 来自俄语 Fandom 对手页（Steam 社区独立描述一致）。资料记载他会用 High Punch、一个独有的球棒攻击（无数值）与 Berserker 被动；球棒与被动没有可用数值，未实现，其余技能由 Basic 池补齐。',
  },
];

export function getOpponent(id: string): FighterDef {
  const o = OPPONENTS.find((f) => f.id === id);
  if (!o) throw new Error(`未知对手: ${id}`);
  return o;
}

export function cloneLoadout(l: Loadout): Loadout {
  return [...l];
}

/** 一代不允许带着少于已解锁槽位数的技能上场 */
export function isLoadoutComplete(l: Loadout): boolean {
  return l.length === ABILITY_SLOTS_MAX;
}
