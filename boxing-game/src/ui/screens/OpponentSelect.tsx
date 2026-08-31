import { OPPONENTS, PLAYER } from '../../core/fighters';
import { derive } from '../../core/stats';
import type { FighterDef } from '../../core/types';
import { useGame } from '../store';

function StatLine({ fighter }: { fighter: FighterDef }) {
  const d = derive(fighter.stats, fighter.health);
  const cells: [string, string | number][] = [
    ['力量 STR', fighter.stats.str],
    ['敏捷 AGI', fighter.stats.agi],
    ['耐力 STM', fighter.stats.stm],
    ['血量 HP', Math.round(d.maxHp)],
    ['命中 ACC', d.acc.toFixed(2)],
    ['护甲 ARM', d.arm.toFixed(1)],
  ];
  return (
    <dl className="statline">
      {cells.map(([k, v]) => (
        <div key={k}>
          <dt>{k}</dt>
          <dd>{v}</dd>
        </div>
      ))}
    </dl>
  );
}

function Provenance({ fighter }: { fighter: FighterDef }) {
  if (!fighter.sourceNote) return null;
  return (
    <p className="provenance">
      {fighter.sourceUrl ? (
        <a href={fighter.sourceUrl} target="_blank" rel="noreferrer">
          数据来源
        </a>
      ) : (
        <span>数据来源</span>
      )}
      ：{fighter.sourceNote}
    </p>
  );
}

export function OpponentSelect() {
  const chooseOpponent = useGame((s) => s.chooseOpponent);

  return (
    <div className="screen screen--select">
      <div className="panel">
        <h2>你的选手</h2>
        <StatLine fighter={PLAYER} />
        <p className="muted">{PLAYER.tagline}</p>
        <Provenance fighter={PLAYER} />
      </div>

      <h2 className="screen__title">挑一个对手</h2>
      <p className="muted screen__hint">
        只有两个对手：一代公开资料里能查到完整属性的只有这两位。其余对手的属性没有可靠出处，
        与其编一个不如不做。
      </p>
      <div className="cards">
        {OPPONENTS.map((o) => (
          <button
            key={o.id}
            type="button"
            className="card"
            style={{ borderTopColor: `#${o.color.toString(16).padStart(6, '0')}` }}
            onClick={() => chooseOpponent(o.id)}
          >
            <h3>{o.name}</h3>
            <p className="muted">{o.tagline}</p>
            <StatLine fighter={o} />
            <Provenance fighter={o} />
          </button>
        ))}
      </div>
    </div>
  );
}
