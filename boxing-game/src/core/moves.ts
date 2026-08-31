import type { Move } from './types';

const WIKI_SKILLS = 'https://punchclub.wiki.gg/wiki/Skills_(Punch_Club_1)';

/**
 * 《拳击俱乐部》一代的可用技能 —— **逐字转录自 data/punch-club-source.md**，没有一个数字是我们编的。
 *
 * 选取标准：只收录**数值完整、可以直接结算**的技能。
 * 资料里共 109 条 ability/skill 记录（105 个技能树节点 + 4 个初始战斗能力），但其中
 * 只有 11 个攻击技能同时给出了伤害/体力/命中三条公式，另有 Block 与 Dodge 给出了体力公式。
 * 其余绝大多数只有文字描述（「替代 Punch，伤害更高」这类），没有可用数值。
 *
 * 被有意排除的：
 * - Knee Crush（Bear）—— 有伤害与体力公式，但**没有命中公式**，补上就等于编造
 * - Power Block / Sidestep（Bear）—— 只有体力公式，其额外效果（加力量 / 让对手多耗体力）
 *   的量级未知；若照搬只会得到一个与 Block / Dodge 完全等价的技能
 * - 全部被动（38 个）、修饰（26 个）与流派节点（3 个）—— 不是可装备的战斗技能
 *
 * 完整的 109 条记录保留在 data/punch-club-source.json 里，本文件只是它的可玩子集。
 */
export const MOVES: readonly Move[] = [
  // ---------------------------------------------------------------- Basic 攻击
  {
    id: 'punch',
    nameEn: 'Punch',
    nameZhUnofficial: '正拳',
    school: 'Basic',
    kind: 'attack',
    effects: [],
    damage: { base: 1, perStr: 0.7 },
    energyCost: { base: 0, perStr: 1 },
    accuracy: { base: 70, perAcc: 20 },
    sourceUrl: WIKI_SKILLS,
    anim: 'punch',
  },
  {
    id: 'kick',
    nameEn: 'Kick',
    nameZhUnofficial: '踢',
    school: 'Basic',
    kind: 'attack',
    effects: [],
    damage: { base: 3, perStr: 0.3 },
    energyCost: { base: 0, perStr: 1 },
    accuracy: { base: 30, perAcc: 30 },
    sourceUrl: WIKI_SKILLS,
    anim: 'kick',
  },
  {
    id: 'high_punch',
    nameEn: 'High Punch',
    nameZhUnofficial: '上拳',
    school: 'Basic',
    kind: 'attack',
    effects: ['比 Punch 更强，体力消耗更大。'],
    damage: { base: 3, perStr: 0.9 },
    energyCost: { base: 2, perStr: 1.2 },
    accuracy: { base: 60, perAcc: 20 },
    sourceUrl: WIKI_SKILLS,
    anim: 'high_punch',
  },
  {
    id: 'high_kick',
    nameEn: 'High Kick',
    nameZhUnofficial: '上踢',
    school: 'Basic',
    kind: 'attack',
    effects: ['比 Kick 更强，体力消耗更大。'],
    damage: { base: 4, perStr: 0.7 },
    energyCost: { base: 1, perStr: 1.2 },
    accuracy: { base: 15, perAcc: 40 },
    sourceUrl: WIKI_SKILLS,
    anim: 'high_kick',
  },
  {
    id: 'low_kick',
    nameEn: 'Low Kick',
    nameZhUnofficial: '下踢',
    school: 'Basic',
    kind: 'attack',
    effects: ['有概率降低对手的耐力与敏捷（触发概率与降幅未公开，未实现）。'],
    damage: { base: 0.5, perStr: 1.5 },
    energyCost: { base: 0, perStr: 1 },
    accuracy: { base: 15, perAcc: 40 },
    conflict: 'wiki 上印的命中公式是 15+40(Hit%)，但同页展示的 1–10 级数值行是 71..80，两者对不上。此处保留印出的公式。',
    sourceUrl: WIKI_SKILLS,
    anim: 'low_kick',
  },
  {
    id: 'uppercut',
    nameEn: 'Uppercut',
    nameZhUnofficial: '上勾拳',
    school: 'Basic',
    kind: 'attack',
    effects: ['强力攻击，对敏捷要求不高。'],
    damage: { base: 1.5, perStr: 1.7 },
    energyCost: { base: 1, perStr: 1.5 },
    accuracy: { base: 65, perAcc: 25 },
    sourceUrl: WIKI_SKILLS,
    anim: 'uppercut',
  },
  {
    id: 'crosspunch',
    nameEn: 'Crosspunch',
    nameZhUnofficial: '交叉拳',
    school: 'Basic',
    kind: 'attack',
    effects: ['重拳，对敏捷要求不高。'],
    damage: { base: 2, perStr: 2 },
    energyCost: { base: 1.5, perStr: 1.6 },
    accuracy: { base: 60, perAcc: 20 },
    sourceUrl: WIKI_SKILLS,
    anim: 'crosspunch',
  },
  {
    id: 'backhand_punch',
    nameEn: 'Backhand Punch',
    nameZhUnofficial: '反手正拳',
    school: 'Basic',
    kind: 'attack',
    effects: ['威力更大，代价更高。'],
    damage: { base: 2, perStr: 1.5 },
    energyCost: { base: 2, perStr: 1.5 },
    accuracy: { base: 45, perAcc: 35 },
    sourceUrl: WIKI_SKILLS,
    anim: 'backhand',
  },
  {
    id: 'backhand_high_punch',
    nameEn: 'Backhand High Punch',
    nameZhUnofficial: '反手上拳',
    school: 'Basic',
    kind: 'attack',
    effects: ['威力可观，但很吃敏捷。'],
    damage: { base: 3, perStr: 2 },
    energyCost: { base: 2, perStr: 1.5 },
    accuracy: { base: 20, perAcc: 60 },
    sourceUrl: WIKI_SKILLS,
    anim: 'backhand',
  },

  // ---------------------------------------------------------------- Tiger 攻击
  {
    id: 'karate_chop',
    nameEn: 'Karate Chop',
    nameZhUnofficial: '空手道正拳',
    school: 'Tiger',
    kind: 'attack',
    effects: ['替代 Punch：更少体力打出更高伤害。'],
    damage: { base: 2, perStr: 1.5 },
    energyCost: { base: 2, perStr: 1.2 },
    accuracy: { base: 20, perAcc: 55 },
    sourceUrl: WIKI_SKILLS,
    anim: 'chop',
  },
  {
    id: 'cutthroat',
    nameEn: 'Cutthroat',
    nameZhUnofficial: '暴徒',
    school: 'Tiger',
    kind: 'attack',
    effects: ['替代 High Punch：提高对手所有技能的体力消耗（提高多少未公开，未实现）。'],
    damage: { base: 3, perStr: 2 },
    energyCost: { base: 1.5, perStr: 1.7 },
    accuracy: { base: 20, perAcc: 60 },
    conflict: 'wiki 同页展示的数值行疑似从 Karate Chop 复制而来，与印出的公式不符。此处保留印出的公式。',
    sourceUrl: WIKI_SKILLS,
    anim: 'chop',
  },

  // ---------------------------------------------------------------- Basic 防守
  {
    id: 'block',
    nameEn: 'Block',
    nameZhUnofficial: '格挡',
    school: 'Basic',
    kind: 'defense',
    defenseKind: 'block',
    effects: ['格挡成功可减少受到的伤害（减伤量与成功率原版未公开，见 unknowns.ts）。'],
    energyCost: { base: 0.5, perStr: 0.3 },
    sourceUrl: WIKI_SKILLS,
    anim: 'block',
  },
  {
    id: 'dodge',
    nameEn: 'Dodge',
    nameZhUnofficial: '闪避',
    school: 'Basic',
    kind: 'defense',
    defenseKind: 'dodge',
    effects: ['闪避成功则该次攻击完全不造成伤害（闪避率公式原版未公开，见 unknowns.ts）。'],
    energyCost: { base: 0.5, perStr: 0.6 },
    sourceUrl: WIKI_SKILLS,
    anim: 'dodge',
  },
];

const BY_ID = new Map(MOVES.map((m) => [m.id, m]));

export function getMove(id: string): Move {
  const m = BY_ID.get(id);
  if (!m) throw new Error(`未知技能: ${id}`);
  return m;
}

export function tryGetMove(id: string | null | undefined): Move | null {
  return id ? (BY_ID.get(id) ?? null) : null;
}

export const ATTACK_MOVES = MOVES.filter((m) => m.kind === 'attack');
export const DEFENSE_MOVES = MOVES.filter((m) => m.kind === 'defense');
