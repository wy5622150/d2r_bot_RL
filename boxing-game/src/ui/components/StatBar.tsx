interface Props {
  label: string;
  value: number;
  max: number;
  tone: 'hp' | 'energy';
  align?: 'left' | 'right';
}

export function StatBar({ label, value, max, tone, align = 'left' }: Props) {
  const pct = max > 0 ? Math.max(0, Math.min(1, value / max)) : 0;
  return (
    <div className={`statbar statbar--${align}`}>
      <div className="statbar__head">
        <span>{label}</span>
        <span className="statbar__num">
          {Math.round(value)} / {max}
        </span>
      </div>
      <div className="statbar__track">
        <div className={`statbar__fill statbar__fill--${tone}`} style={{ width: `${pct * 100}%` }} />
      </div>
    </div>
  );
}
