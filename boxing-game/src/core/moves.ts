import type { Move } from './types';

/**
 * 招式表。
 *
 * ⚠️ 这里的数字是**占位值**，不是《拳击俱乐部》原版数值。
 * 第一版的目标是完全对标原版的招式表（Basic / Bear / Tiger / Turtle 四系共 27 个技能，
 * 各自的伤害、体力消耗、命中率、先手消耗），但本沙箱的出站策略封了 punchclub.wiki.gg /
 * fandom / steamcommunity，暂时取不到。拿到原版表之后整体替换这一个文件即可，
 * 引擎和界面都不需要改 —— 这也是把数值全部集中在这里的原因。
 *
 * 结构上只保留原版真实存在的机制：伤害、体力消耗、命中率、格挡、闪避、反击、抽体力、震慑。
 */
export const MOVES: readonly Move[] = [
  // ---------------- 进攻 ----------------
  {
    id: 'jab',
    name: '刺拳',
    kind: 'attack',
    desc: '最省力的试探拳，命中高、伤害低，顺带打乱对手呼吸。',
    energyCost: 4,
    baseDamage: 8,
    accuracy: 0.92,
    energyDrain: 2,
    anim: 'jab',
  },
  {
    id: 'cross',
    name: '直拳',
    kind: 'attack',
    desc: '标准后手直拳，伤害与消耗都很均衡。',
    energyCost: 7,
    baseDamage: 14,
    accuracy: 0.85,
    anim: 'cross',
  },
  {
    id: 'hook',
    name: '勾拳',
    kind: 'attack',
    desc: '横向绕过正面防守的重拳。',
    energyCost: 9,
    baseDamage: 18,
    accuracy: 0.8,
    anim: 'hook',
  },
  {
    id: 'bodyshot',
    name: '击腹',
    kind: 'attack',
    desc: '伤害一般，但能大量抽干对手的体力条。',
    energyCost: 8,
    baseDamage: 12,
    accuracy: 0.88,
    energyDrain: 9,
    anim: 'bodyshot',
  },
  {
    id: 'overhand',
    name: '摆拳',
    kind: 'attack',
    desc: '大幅度抡出的重拳，打中很疼，打空很亏。',
    energyCost: 10,
    baseDamage: 20,
    accuracy: 0.75,
    anim: 'overhand',
  },
  {
    id: 'uppercut',
    name: '上勾拳',
    kind: 'attack',
    desc: '从下颚穿上去的拳，有概率把对手打懵一拍。',
    energyCost: 11,
    baseDamage: 22,
    accuracy: 0.72,
    stunChance: 0.18,
    anim: 'uppercut',
  },
  {
    id: 'haymaker',
    name: '重摆拳',
    kind: 'attack',
    desc: '孤注一掷的一拳。命中率很低，但打中基本改变比赛走向。',
    energyCost: 15,
    baseDamage: 32,
    accuracy: 0.58,
    stunChance: 0.25,
    anim: 'haymaker',
  },

  // ---------------- 防守 ----------------
  {
    id: 'guard',
    name: '格挡',
    kind: 'defense',
    desc: '举拳护头，减伤 45%，消耗极低。对重拳最有效。',
    energyCost: 3,
    blockPct: 0.45,
    anim: 'guard',
  },
  {
    id: 'clinch',
    name: '抱缠',
    kind: 'defense',
    desc: '减伤 35%，还能顺回一点体力。最省力的防守。',
    energyCost: 2,
    blockPct: 0.35,
    energyRestore: 4,
    anim: 'clinch',
  },
  {
    id: 'slip',
    name: '闪避',
    kind: 'defense',
    desc: '侧身让开，有概率完全躲掉一拳。吃敏捷。',
    energyCost: 5,
    dodge: 0.35,
    anim: 'slip',
  },
  {
    id: 'parry',
    name: '招架反击',
    kind: 'defense',
    desc: '减伤 30%，并把挡下的力道按 60% 还击回去。',
    energyCost: 7,
    blockPct: 0.3,
    counter: 0.6,
    anim: 'parry',
  },

  // ---------------- 休整 ----------------
  {
    id: 'breathe',
    name: '调整呼吸',
    kind: 'rest',
    desc: '放弃一次出手机会，换回一大口体力。',
    energyCost: 0,
    energyRestore: 18,
    anim: 'breathe',
  },
];

const BY_ID = new Map(MOVES.map((m) => [m.id, m]));

export function getMove(id: string): Move {
  const m = BY_ID.get(id);
  if (!m) throw new Error(`未知招式: ${id}`);
  return m;
}

/** 槽位可能为空，取不到就返回 null */
export function tryGetMove(id: string | null | undefined): Move | null {
  return id ? (BY_ID.get(id) ?? null) : null;
}

export const ATTACK_MOVES = MOVES.filter((m) => m.kind === 'attack');
export const DEFENSE_MOVES = MOVES.filter((m) => m.kind === 'defense');
export const REST_MOVES = MOVES.filter((m) => m.kind === 'rest');

/** 进攻槽可放：攻击 + 休整 */
export const OFFENSE_POOL_KINDS: readonly Move['kind'][] = ['attack', 'rest'];
