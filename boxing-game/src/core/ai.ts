import { getMove, tryGetMove } from './moves';
import type { Rng } from './rng';
import { DEFENSE_SLOTS, OFFENSE_SLOTS } from './stats';
import type { AiStyle, FighterState, Loadout, Slot } from './types';

/** 每种风格的基础套路 */
const TEMPLATES: Record<Exclude<AiStyle, 'manual'>, Loadout> = {
  // 压迫型：塞满重拳，赌前两回合把人打倒
  pressure: {
    offense: ['overhand', 'hook', 'haymaker', 'cross'],
    defense: ['guard', 'guard', 'clinch'],
  },
  // 游斗型：低耗刺拳磨血，防守全压闪避
  outboxer: {
    offense: ['jab', 'jab', 'cross', 'breathe'],
    defense: ['slip', 'slip', 'parry'],
  },
  // 铁壁型：出手节制，靠减伤和血量拖到读分
  wall: {
    offense: ['cross', 'hook', 'breathe', null],
    defense: ['guard', 'clinch', 'parry'],
  },
};

function fit(slots: Slot[], len: number): Slot[] {
  const out = slots.slice(0, len);
  while (out.length < len) out.push(null);
  return out;
}

/** 只保留招式池里有的招，池子里没有的槽位清空 */
function filterByPool(slots: Slot[], pool: readonly string[]): Slot[] {
  return slots.map((s) => (s && pool.includes(s) ? s : null));
}

function energyCostOf(id: Slot): number {
  const m = tryGetMove(id);
  return m ? m.energyCost : 0;
}

function damageOf(id: Slot): number {
  const m = tryGetMove(id);
  return m?.baseDamage ?? 0;
}

/**
 * 对手在回合之间重新配槽。规则很短，但足以让三种风格打出不同的比赛节奏：
 * 体力见底就空槽/休整，血量告急就加防，对手快倒了就上重拳。
 */
export function aiReconfigure(self: FighterState, foe: FighterState, rng: Rng): Loadout {
  const style = self.def.style;
  if (style === 'manual') return self.loadout;

  const pool = self.def.pool;
  const template = TEMPLATES[style];
  const offense = fit(filterByPool([...template.offense], pool), OFFENSE_SLOTS);
  const defense = fit(filterByPool([...template.defense], pool), DEFENSE_SLOTS);

  const energyRatio = self.energy / self.derived.maxEnergy;
  const hpRatio = self.hp / self.derived.maxHp;
  const foeHpRatio = foe.hp / foe.derived.maxHp;

  const replaceCostliest = (value: Slot): void => {
    let idx = -1;
    let cost = -1;
    for (let i = 0; i < offense.length; i++) {
      const c = energyCostOf(offense[i] ?? null);
      if (c > cost) {
        cost = c;
        idx = i;
      }
    }
    if (idx >= 0 && cost > 0) offense[idx] = value;
  };

  // 体力吃紧 → 先换休整，再直接空槽
  if (energyRatio < 0.3) {
    replaceCostliest(pool.includes('breathe') ? 'breathe' : null);
  }
  if (energyRatio < 0.15) {
    replaceCostliest(null);
  }

  // 血量告急 → 少出手、补满防守槽
  if (hpRatio < 0.35) {
    replaceCostliest(null);
    const guard = pool.find((id) => getMove(id).kind === 'defense' && getMove(id).energyCost <= 3);
    for (let i = 0; i < defense.length; i++) {
      if (!defense[i] && guard) defense[i] = guard;
    }
  }

  // 对手只剩一口气 → 收起休整，全换成池子里最重的拳
  if (foeHpRatio < 0.25 && energyRatio > 0.2) {
    const heaviest = [...pool]
      .filter((id) => getMove(id).kind === 'attack')
      .sort((a, b) => damageOf(b) - damageOf(a))[0];
    if (heaviest) {
      for (let i = 0; i < offense.length; i++) {
        const m = tryGetMove(offense[i] ?? null);
        if (!m || m.kind === 'rest') offense[i] = heaviest;
      }
    }
  }

  // 同样的局面下给一点点变化，避免三回合一模一样
  if (rng.chance(0.25)) {
    const a = rng.int(offense.length);
    const b = rng.int(offense.length);
    const tmp = offense[a] ?? null;
    offense[a] = offense[b] ?? null;
    offense[b] = tmp;
  }

  return { offense, defense };
}

/** 界面用：检查配槽是否合法（长度对、招式存在、种类放对槽） */
export function validateLoadout(loadout: Loadout): string | null {
  if (loadout.offense.length !== OFFENSE_SLOTS) return `进攻槽必须是 ${OFFENSE_SLOTS} 个`;
  if (loadout.defense.length !== DEFENSE_SLOTS) return `防守槽必须是 ${DEFENSE_SLOTS} 个`;
  for (const id of loadout.offense) {
    if (!id) continue;
    const kind = getMove(id).kind;
    if (kind !== 'attack' && kind !== 'rest') return `进攻槽不能放「${getMove(id).name}」`;
  }
  for (const id of loadout.defense) {
    if (!id) continue;
    if (getMove(id).kind !== 'defense') return `防守槽只能放防守招式`;
  }
  return null;
}
