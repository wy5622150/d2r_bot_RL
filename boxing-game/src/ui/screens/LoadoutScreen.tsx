import { getOpponent } from '../../core/fighters';
import { MAX_ROUNDS } from '../../core/stats';
import { LoadoutEditor } from '../components/LoadoutEditor';
import { useGame } from '../store';

export function LoadoutScreen() {
  const opponentId = useGame((s) => s.opponentId);
  const startFight = useGame((s) => s.startFight);
  const backToSelect = useGame((s) => s.backToSelect);
  const opponent = getOpponent(opponentId);

  return (
    <div className="screen">
      <div className="screen__bar">
        <button type="button" className="btn btn--ghost" onClick={backToSelect}>
          ← 换个对手
        </button>
        <h2>
          赛前选技能 · 对手：<strong>{opponent.name}</strong>
        </h2>
        <button type="button" className="btn btn--primary" onClick={startFight}>
          开始比赛
        </button>
      </div>

      <p className="muted screen__hint">
        最多打 {MAX_ROUNDS} 个回合，全程自动进行 —— 你唯一的操作就是决定带哪 5 个技能。
        每个回合结束后还能重新选一次，针对刚才打出来的情况调整。
      </p>
      <p className="muted screen__hint">{opponent.tagline}</p>

      <LoadoutEditor />
    </div>
  );
}
