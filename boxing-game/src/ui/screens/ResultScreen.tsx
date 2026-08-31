import { PLAYER } from '../../core/fighters';
import { useGame } from '../store';

export function ResultScreen() {
  const fight = useGame((s) => s.fight);
  const rematch = useGame((s) => s.rematch);
  const backToSelect = useGame((s) => s.backToSelect);
  if (!fight?.result) return null;

  const { winner, method, round } = fight.result;
  const title = winner === null ? '平局' : winner === 'player' ? '你赢了' : '你输了';
  const how =
    method === 'ko'
      ? `第 ${round} 回合 KO`
      : `打满 ${round} 个回合，按剩余血量百分比读分`;

  const pct = (hp: number, max: number) => `${Math.round((hp / max) * 100)}%`;

  return (
    <div className="overlay">
      <div className="overlay__panel overlay__panel--result">
        <h2 className={winner === 'player' ? 'win' : winner === null ? '' : 'lose'}>{title}</h2>
        <p className="muted">{how}</p>
        <div className="result__score">
          <div>
            <span>{PLAYER.name}</span>
            <strong>{pct(fight.player.hp, fight.player.derived.maxHp)}</strong>
          </div>
          <div>
            <span>{fight.opponent.def.name}</span>
            <strong>{pct(fight.opponent.hp, fight.opponent.derived.maxHp)}</strong>
          </div>
        </div>
        <div className="overlay__actions">
          <button type="button" className="btn btn--primary" onClick={rematch}>
            改配槽再打一场
          </button>
          <button type="button" className="btn btn--ghost" onClick={backToSelect}>
            换个对手
          </button>
        </div>
      </div>
    </div>
  );
}
