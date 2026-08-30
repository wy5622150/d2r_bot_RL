import { FightScreen } from './ui/screens/FightScreen';
import { LoadoutScreen } from './ui/screens/LoadoutScreen';
import { OpponentSelect } from './ui/screens/OpponentSelect';
import { ResultScreen } from './ui/screens/ResultScreen';
import { useGame } from './ui/store';

export default function App() {
  const screen = useGame((s) => s.screen);
  const inFight = screen === 'fight' || screen === 'result';

  return (
    <div className="app">
      <header className="app__head">
        <h1>拳击俱乐部 · 战斗</h1>
        <span className="muted">配槽 → 自动对打 → 回合间再配槽</span>
      </header>

      {screen === 'select' && <OpponentSelect />}
      {screen === 'loadout' && <LoadoutScreen />}
      {inFight && <FightScreen />}
      {screen === 'result' && <ResultScreen />}
    </div>
  );
}
