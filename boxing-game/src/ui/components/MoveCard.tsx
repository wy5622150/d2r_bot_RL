import { accuracyText, evalAccuracy, evalFormula, formulaText } from '../../core/stats';
import type { Derived, Move, Stats } from '../../core/types';

const KIND_LABEL: Record<Move['kind'], string> = {
  attack: '攻击',
  defense: '防守',
};

/** 原版写法：`1+0.7(str)` / `70+20(Hit%)` —— 直接显示公式本身，不藏起来 */
export function rawFormulas(move: Move): string[] {
  const out: string[] = [`体力 ${formulaText(move.energyCost)}`];
  if (move.damage) out.push(`伤害 ${formulaText(move.damage)}`);
  if (move.accuracy) out.push(`命中 ${accuracyText(move.accuracy)}`);
  return out;
}

/** 代入这名选手的属性后的实际数字 */
export function actualNumbers(move: Move, stats: Stats, derived: Derived): string[] {
  const out: string[] = [`体力 ${round1(evalFormula(move.energyCost, stats.str))}`];
  if (move.damage) out.push(`伤害 ${Math.round(evalFormula(move.damage, stats.str))}`);
  if (move.accuracy) out.push(`命中 ${Math.round(evalAccuracy(move.accuracy, derived.acc) * 100)}%`);
  if (move.defenseKind === 'dodge') out.push('成功则完全免伤');
  if (move.defenseKind === 'block') out.push('成功则减伤');
  return out;
}

function round1(n: number): number {
  return Math.round(n * 10) / 10;
}

interface Props {
  move: Move;
  stats: Stats;
  derived: Derived;
  selected?: boolean;
  disabled?: boolean;
  onClick?: () => void;
}

export function MoveCard({ move, stats, derived, selected, disabled, onClick }: Props) {
  return (
    <button
      type="button"
      className={`movecard movecard--${move.kind}${selected ? ' is-selected' : ''}`}
      disabled={disabled}
      onClick={onClick}
    >
      <div className="movecard__top">
        <span className="movecard__name">{move.nameEn}</span>
        <span className="movecard__kind">
          {move.school} · {KIND_LABEL[move.kind]}
        </span>
      </div>
      <div className="movecard__stats">
        {actualNumbers(move, stats, derived).map((s) => (
          <span key={s}>{s}</span>
        ))}
      </div>
      <p className="movecard__desc">{move.effects.join(' ') || move.nameZhUnofficial}</p>
      <p className="movecard__raw">原版公式：{rawFormulas(move).join('　')}</p>
    </button>
  );
}
