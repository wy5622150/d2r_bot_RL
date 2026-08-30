import { getMove } from './moves';
import type { Rng } from './rng';
import { EXHAUST_BONUS_DAMAGE, evalAccuracy, evalFormula } from './stats';
import type { FighterState, Move, RoundEvent, Side } from './types';
import { UNK } from './unknowns';

type DistributiveOmit<T, K extends PropertyKey> = T extends unknown ? Omit<T, K> : never;

/** 事件里的 hp/energy 快照与 tick 由 engine 统一补齐 */
export type EventInput = DistributiveOmit<RoundEvent, 'hp' | 'energy' | 'tick'>;
export type Emit = (e: EventInput) => void;

export interface ResolveCtx {
  attacker: FighterState;
  defender: FighterState;
  attackerSide: Side;
  defenderSide: Side;
  rng: Rng;
  emit: Emit;
}

/** 技能对该选手的实际体力消耗（原版：base + 系数 × STR） */
export function costOf(move: Move, f: FighterState): number {
  return evalFormula(move.energyCost, f.def.stats.str);
}

function equipped(f: FighterState, kind: Move['kind']): Move[] {
  return f.loadout.map(getMove).filter((m) => m.kind === kind);
}

/**
 * 攻击阶段抽一个攻击技能。
 * 一代是一组共享技能槽、随机出招，开发者明确说槽位顺序无意义 —— 所以这里是等概率抽取，
 * 没有游标、没有顺序。装了几个同名技能就等比例提高它出现的概率。
 */
export function pickAttack(f: FighterState, rng: Rng): Move | null {
  const affordable = equipped(f, 'attack').filter((m) => f.energy >= costOf(m, f));
  if (affordable.length === 0) return null;
  return rng.pick(affordable);
}

/** 防守阶段抽一个防守技能；抽中但体力不足则防守失败 */
export function pickDefense(f: FighterState, rng: Rng): Move | null {
  const defenses = equipped(f, 'defense');
  if (defenses.length === 0) return null;
  const picked = rng.pick(defenses);
  if (UNK.defenseFailsWhenBroke && f.energy < costOf(picked, f)) return null;
  return picked;
}

/** 结算一次攻击。直接修改双方状态，并通过 emit 吐出事件。 */
export function resolveAttack(ctx: ResolveCtx, moveId: string): void {
  const { attacker, defender, attackerSide, defenderSide, rng, emit } = ctx;
  const move = getMove(moveId);

  attacker.energy = Math.max(0, attacker.energy - costOf(move, attacker));

  const defMove = pickDefense(defender, rng);
  if (defMove) {
    defender.energy = Math.max(0, defender.energy - costOf(defMove, defender));
  }

  emit({
    type: 'attack',
    side: attackerSide,
    move: move.id,
    defenseMove: defMove?.id ?? null,
    text: `${attacker.def.name} 打出 ${move.nameEn}${
      defMove ? `，${defender.def.name} 用 ${defMove.nameEn} 应对` : ''
    }`,
  });

  // 受击瞬间体力见底 → 额外 +10 伤害并被击倒（原版已确证）
  const exhausted = defender.energy <= 0;

  // 1) 闪避：成功则完全免伤（原版已确证；闪避率公式未公开，取自 unknowns.ts）
  if (defMove?.defenseKind === 'dodge') {
    if (rng.chance(UNK.dodgeChance(defender.derived.acc))) {
      emit({
        type: 'dodge',
        side: defenderSide,
        move: defMove.id,
        text: `${defender.def.name} 闪开了 ${move.nameEn}`,
      });
      return;
    }
  }

  // 2) 命中：accuracy = base + perAcc × ACC（原版已确证）
  const hitChance = move.accuracy ? evalAccuracy(move.accuracy, attacker.derived.acc) : 1;
  if (!rng.chance(hitChance)) {
    emit({
      type: 'miss',
      side: attackerSide,
      move: move.id,
      text: `${attacker.def.name} 的 ${move.nameEn} 打空了`,
    });
    return;
  }

  // 3) 伤害：base + 系数 × STR，四舍五入（原版已确证）
  const raw = Math.round(evalFormula(move.damage ?? { base: 0, perStr: 0 }, attacker.def.stats.str));

  // 格挡减伤（减伤量与成功率未公开，取自 unknowns.ts）
  const blocking = defMove?.defenseKind === 'block' && rng.chance(UNK.blockSuccessChance);
  const afterBlock = blocking ? raw * (1 - UNK.blockReduction) : raw;

  // 护甲 ARM = STM×1.3。相减还是按比例，wiki 没有确认 —— 见 unknowns.ts 的 armorMode
  const armed =
    UNK.armorMode === 'percent'
      ? afterBlock * (1 - defender.derived.arm / 100)
      : UNK.armorMode === 'capped'
        ? afterBlock - Math.min(defender.derived.arm, afterBlock * UNK.armorCapRatio)
        : afterBlock - defender.derived.arm;
  const landed = Math.max(UNK.minDamage, Math.round(armed));
  const damage = landed + (exhausted ? EXHAUST_BONUS_DAMAGE : 0);
  const blocked = raw - landed;

  defender.hp = Math.max(0, defender.hp - damage);

  emit({
    type: 'hit',
    side: attackerSide,
    target: defenderSide,
    move: move.id,
    damage,
    blocked,
    exhaustBonus: exhausted,
    text: `${move.nameEn} 命中，${damage} 点伤害${
      exhausted ? '——对手体力见底，额外挨了 10 点' : blocked > 0 ? `（挡下 ${blocked}）` : ''
    }`,
  });

  // 4) 体力归零 → 被击倒（KO 判定统一由 engine 在阶段结束后做）
  if (exhausted && defender.hp > 0) {
    defender.lostPhases += UNK.knockdownLostPhases;
    defender.energy = Math.min(
      defender.derived.maxEnergy,
      defender.energy + UNK.knockdownGetUpEnergy,
    );
    emit({
      type: 'knockdown',
      side: defenderSide,
      text: `${defender.def.name} 体力透支，被打倒在地！`,
    });
  }
}
