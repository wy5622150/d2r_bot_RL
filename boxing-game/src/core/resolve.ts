import { getMove, tryGetMove } from './moves';
import type { Rng } from './rng';
import { CRIT_MULT, EXHAUST_BONUS_DAMAGE, MAX_DODGE, MIN_DAMAGE } from './stats';
import type { FighterState, Move, RoundEvent, Side } from './types';

type DistributiveOmit<T, K extends PropertyKey> = T extends unknown ? Omit<T, K> : never;

/** 事件里的 hp/energy 快照与 tick 由 engine 统一补齐，调用方只写业务字段 */
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

/**
 * 从防守槽里抽一个防守招式。
 * 空槽也参与抽取 —— 防守槽留空 = 有一定概率完全不设防，这是原作的取舍。
 * 抽中但能量不够 → 防守失败，同样算不设防。
 */
export function pickDefense(defender: FighterState, rng: Rng): Move | null {
  const slots = defender.loadout.defense;
  if (slots.length === 0) return null;
  const picked = slots[rng.int(slots.length)];
  const move = tryGetMove(picked ?? null);
  if (!move) return null;
  if (defender.energy < move.energyCost) return null;
  return move;
}

/** 结算一次攻击。会直接修改 attacker / defender 的状态，并通过 emit 吐出事件。 */
export function resolveAttack(ctx: ResolveCtx, moveId: string): void {
  const { attacker, defender, attackerSide, defenderSide, rng, emit } = ctx;
  const move = getMove(moveId);

  attacker.energy = Math.max(0, attacker.energy - move.energyCost);

  const defMove = pickDefense(defender, rng);
  if (defMove) {
    defender.energy = Math.max(0, defender.energy - defMove.energyCost);
    if (defMove.energyRestore) {
      defender.energy = Math.min(defender.derived.maxEnergy, defender.energy + defMove.energyRestore);
    }
  }

  emit({
    type: 'attack',
    side: attackerSide,
    move: move.id,
    defenseMove: defMove?.id ?? null,
    text: `${attacker.def.name} 打出${move.name}${defMove ? `，${defender.def.name} 用${defMove.name}应对` : ''}`,
  });

  // 受击瞬间体力见底 → 额外惩罚伤害并被击倒（原作机制）
  const exhausted = defender.energy <= 0;

  // 1) 闪避
  if (defMove?.dodge) {
    // 留 15% 的兜底命中，再灵活的选手也不该完全无法被击中
    const p = Math.min(MAX_DODGE, defMove.dodge + defender.derived.dodgeBonus);
    if (rng.chance(p)) {
      emit({
        type: 'dodge',
        side: defenderSide,
        move: defMove.id,
        text: `${defender.def.name} 一个侧身，${move.name}擦着头皮过去了`,
      });
      return;
    }
  }

  // 2) 命中
  if (!rng.chance(move.accuracy ?? 1)) {
    emit({
      type: 'miss',
      side: attackerSide,
      move: move.id,
      text: `${attacker.def.name} 的${move.name}打空了`,
    });
    return;
  }

  // 3) 伤害
  const crit = rng.chance(attacker.derived.critChance);
  const raw = Math.round((move.baseDamage ?? 0) * attacker.derived.damageMult * (crit ? CRIT_MULT : 1));
  // 先按百分比减伤，再扣固定护甲：百分比让大小拳同比例受影响，护甲专门惩罚小拳
  const afterBlock = Math.max(
    MIN_DAMAGE,
    Math.round(raw * (1 - (defMove?.blockPct ?? 0))) - defender.derived.armor,
  );
  const blocked = raw - afterBlock;
  const damage = afterBlock + (exhausted ? EXHAUST_BONUS_DAMAGE : 0);

  defender.hp = Math.max(0, defender.hp - damage);

  emit({
    type: 'hit',
    side: attackerSide,
    target: defenderSide,
    move: move.id,
    damage,
    crit,
    blocked,
    exhaustBonus: exhausted,
    text: `${move.name}命中，${damage} 点伤害${crit ? '（暴击！）' : ''}${
      exhausted ? '——对手体力见底，这一拳格外沉' : blocked > 0 ? `（挡下 ${blocked}）` : ''
    }`,
  });

  // 4) 反击
  if (defMove?.counter && blocked > 0 && defender.hp > 0) {
    const cdmg = Math.max(MIN_DAMAGE, Math.round(blocked * defMove.counter));
    attacker.hp = Math.max(0, attacker.hp - cdmg);
    emit({
      type: 'counter',
      side: defenderSide,
      target: attackerSide,
      damage: cdmg,
      text: `${defender.def.name} 顺势反击，回敬 ${cdmg} 点`,
    });
  }

  // 5) 抽体力
  if (move.energyDrain) {
    defender.energy = Math.max(0, defender.energy - move.energyDrain);
  }

  // 6) 震慑
  if (move.stunChance && defender.hp > 0 && rng.chance(move.stunChance)) {
    defender.stunned = true;
    emit({
      type: 'stun',
      side: defenderSide,
      text: `${defender.def.name} 被打懵了，下一拍出不了手`,
    });
  }

  // 7) 体力归零 → 倒地（KO 判定统一由 engine 在动作结束后做）
  if (exhausted && defender.hp > 0) {
    defender.knockedDown = true;
    emit({
      type: 'knockdown',
      side: defenderSide,
      text: `${defender.def.name} 体力透支，被打倒在地！`,
    });
  }
}
