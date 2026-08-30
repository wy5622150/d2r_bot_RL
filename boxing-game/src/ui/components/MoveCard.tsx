import type { Move } from '../../core/types';

const KIND_LABEL: Record<Move['kind'], string> = {
  attack: '进攻',
  defense: '防守',
  rest: '休整',
};

/** 把一个招式的关键数字摊开成人话 */
export function moveStats(move: Move): string[] {
  const out: string[] = [`体力 ${move.energyCost}`];
  if (move.baseDamage) out.push(`伤害 ${move.baseDamage}`);
  if (move.accuracy !== undefined) out.push(`命中 ${Math.round(move.accuracy * 100)}%`);
  if (move.blockPct) out.push(`减伤 ${Math.round(move.blockPct * 100)}%`);
  if (move.dodge) out.push(`闪避 ${Math.round(move.dodge * 100)}%`);
  if (move.counter) out.push(`反击 ${Math.round(move.counter * 100)}%`);
  if (move.energyDrain) out.push(`抽体力 ${move.energyDrain}`);
  if (move.energyRestore) out.push(`回体力 ${move.energyRestore}`);
  if (move.stunChance) out.push(`震慑 ${Math.round(move.stunChance * 100)}%`);
  return out;
}

interface Props {
  move: Move;
  selected?: boolean;
  disabled?: boolean;
  onClick?: () => void;
}

export function MoveCard({ move, selected, disabled, onClick }: Props) {
  return (
    <button
      type="button"
      className={`movecard movecard--${move.kind}${selected ? ' is-selected' : ''}`}
      disabled={disabled}
      onClick={onClick}
    >
      <div className="movecard__top">
        <span className="movecard__name">{move.name}</span>
        <span className="movecard__kind">{KIND_LABEL[move.kind]}</span>
      </div>
      <div className="movecard__stats">
        {moveStats(move).map((s) => (
          <span key={s}>{s}</span>
        ))}
      </div>
      <p className="movecard__desc">{move.desc}</p>
    </button>
  );
}
