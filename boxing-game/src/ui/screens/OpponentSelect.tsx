import { OPPONENTS, PLAYER } from '../../core/fighters';
import { derive } from '../../core/stats';
import type { FighterDef } from '../../core/types';
import { useGame } from '../store';

function StatLine({ fighter }: { fighter: FighterDef }) {
  const d = derive(fighter.stats);
  return (
    <dl className="statline">
      <div>
        <dt>力量</dt>
        <dd>{fighter.stats.str}</dd>
      </div>
      <div>
        <dt>敏捷</dt>
        <dd>{fighter.stats.agi}</dd>
      </div>
      <div>
        <dt>耐力</dt>
        <dd>{fighter.stats.sta}</dd>
      </div>
      <div>
        <dt>血量</dt>
        <dd>{d.maxHp}</dd>
      </div>
      <div>
        <dt>体力</dt>
        <dd>{d.maxEnergy}</dd>
      </div>
      <div>
        <dt>先手</dt>
        <dd>{d.initiative}</dd>
      </div>
    </dl>
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
      </div>

      <h2 className="screen__title">挑一个对手</h2>
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
          </button>
        ))}
      </div>
    </div>
  );
}
