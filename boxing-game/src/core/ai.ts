import { getMove } from './moves';
import type { Rng } from './rng';
import { ABILITY_SLOTS_MAX } from './stats';
import type { AiStyle, FighterState, Loadout } from './types';

/**
 * 对手在回合之间重新选技能。
 *
 * 一代允许在回合间更换装备的技能，这是玩家（和对手）唯一的操作。
 * 但原版并没有公开过 AI 的换装逻辑 —— 下面这套规则是我们自己写的对手行为，
 * 不是原版数据。它只决定「对手带哪 5 个技能」，不影响任何结算公式。
 */

/** 保证配置合法：正好 ABILITY_SLOTS_MAX 个、全部在自己的技能池里、至少带一个攻击技能 */
function normalize(picked: string[], pool: readonly string[]): Loadout {
  const valid = picked.filter((id) => pool.includes(id));
  const out = valid.slice(0, ABILITY_SLOTS_MAX);
  for (const id of pool) {
    if (out.length >= ABILITY_SLOTS_MAX) break;
    out.push(id);
  }
  if (!out.some((id) => getMove(id).kind === 'attack')) {
    const atk = pool.find((id) => getMove(id).kind === 'attack');
    if (atk) out[out.length - 1] = atk;
  }
  return out;
}

export function aiReconfigure(self: FighterState, foe: FighterState, rng: Rng): Loadout {
  const style: AiStyle = self.def.style;
  if (style === 'manual') return self.loadout;

  const pool = self.def.pool;
  const attacks = pool.filter((id) => getMove(id).kind === 'attack');
  const defenses = pool.filter((id) => getMove(id).kind === 'defense');
  const energyRatio = self.energy / self.derived.maxEnergy;
  const hpRatio = self.hp / self.derived.maxHp;

  /** 按体力消耗排序：省力的在前 */
  const byCost = [...attacks].sort(
    (a, b) =>
      getMove(a).energyCost.base +
      getMove(a).energyCost.perStr * self.def.stats.str -
      (getMove(b).energyCost.base + getMove(b).energyCost.perStr * self.def.stats.str),
  );

  let wantDefense = style === 'defensive' ? 2 : style === 'balanced' ? 2 : 1;
  // 血量告急就多带防守；体力见底则换省力的攻击技能
  if (hpRatio < 0.35) wantDefense = Math.min(defenses.length, wantDefense + 1);
  const cheapFirst = energyRatio < 0.35 || style === 'defensive';

  const picked: string[] = defenses.slice(0, wantDefense);
  const order = cheapFirst ? byCost : [...byCost].reverse();
  for (const id of order) {
    if (picked.length >= ABILITY_SLOTS_MAX) break;
    picked.push(id);
  }

  // 同样局面下给一点变化，避免每回合一模一样
  if (rng.chance(0.3) && picked.length > 1) {
    const a = rng.int(picked.length);
    const b = rng.int(picked.length);
    const tmp = picked[a]!;
    picked[a] = picked[b]!;
    picked[b] = tmp;
  }

  void foe;
  return normalize(picked, pool);
}

/** 界面用：检查配置是否合法 */
export function validateLoadout(loadout: Loadout, pool: readonly string[]): string | null {
  if (loadout.length !== ABILITY_SLOTS_MAX) {
    return `必须正好装备 ${ABILITY_SLOTS_MAX} 个技能（一代不允许留空）`;
  }
  for (const id of loadout) {
    if (!pool.includes(id)) return `技能 ${id} 不在可用池里`;
  }
  if (!loadout.some((id) => getMove(id).kind === 'attack')) {
    return '至少要带一个攻击技能，否则打不出任何伤害';
  }
  return null;
}
