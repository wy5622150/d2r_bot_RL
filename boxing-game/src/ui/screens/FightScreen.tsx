import { useEffect, useRef } from 'react';
import { PLAYER } from '../../core/fighters';
import { ROUNDS } from '../../core/stats';
import type { RoundEvent } from '../../core/types';
import { busState, EventBus } from '../../game/EventBus';
import { PhaserGame } from '../../game/PhaserGame';
import { LoadoutEditor } from '../components/LoadoutEditor';
import { StatBar } from '../components/StatBar';
import { useGame } from '../store';

const SPEEDS = [1, 2, 4];

export function FightScreen() {
  const fight = useGame((s) => s.fight);
  const hud = useGame((s) => s.hud);
  const log = useGame((s) => s.log);
  const phase = useGame((s) => s.phase);
  const speed = useGame((s) => s.speed);
  const setSpeed = useGame((s) => s.setSpeed);
  const playNextRound = useGame((s) => s.playNextRound);
  const logRef = useRef<HTMLDivElement>(null);

  // Phaser → React：血条/体力条和解说都跟着"画面播到哪一条"走，不会提前剧透
  useEffect(() => {
    const onShown = (e: RoundEvent) => useGame.getState().onEventShown(e);
    const onDone = () => useGame.getState().onRoundDone();
    EventBus.on('fight:event-shown', onShown);
    EventBus.on('fight:round-done', onDone);
    return () => {
      EventBus.off('fight:event-shown', onShown);
      EventBus.off('fight:round-done', onDone);
    };
  }, []);

  // 场景就绪后再开打：React 挂载和 Phaser boot 是两条时间线
  useEffect(() => {
    if (!fight || phase !== 'ready') return;
    const start = () => {
      EventBus.emit('fight:setup', {
        player: { name: PLAYER.name, color: PLAYER.color, side: 'player' },
        opponent: {
          name: fight.opponent.def.name,
          color: fight.opponent.def.color,
          side: 'opponent',
        },
        hp: { player: fight.player.hp, opponent: fight.opponent.hp },
        maxHp: { player: fight.player.derived.maxHp, opponent: fight.opponent.derived.maxHp },
      });
      EventBus.emit('fight:set-speed', useGame.getState().speed);
      useGame.getState().playNextRound();
    };
    if (busState.sceneReady) {
      start();
      return;
    }
    EventBus.on('scene:ready', start);
    return () => {
      EventBus.off('scene:ready', start);
    };
  }, [fight, phase]);

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight });
  }, [log]);

  if (!fight) return null;
  const round = Math.max(1, fight.round);

  return (
    <div className="fight">
      <div className="fight__hud">
        <StatBar label={PLAYER.name} value={hud.hp.player} max={fight.player.derived.maxHp} tone="hp" />
        <div className="fight__round">
          第 {round} / {ROUNDS} 回合
        </div>
        <StatBar
          label={fight.opponent.def.name}
          value={hud.hp.opponent}
          max={fight.opponent.derived.maxHp}
          tone="hp"
          align="right"
        />
        <StatBar
          label="体力"
          value={hud.energy.player}
          max={fight.player.derived.maxEnergy}
          tone="energy"
        />
        <div className="fight__speed">
          {SPEEDS.map((s) => (
            <button
              key={s}
              type="button"
              className={`btn btn--chip${speed === s ? ' is-on' : ''}`}
              onClick={() => setSpeed(s)}
            >
              {s}x
            </button>
          ))}
          <button
            type="button"
            className="btn btn--chip"
            disabled={phase !== 'playing'}
            onClick={() => EventBus.emit('fight:skip')}
          >
            跳过本回合
          </button>
        </div>
        <StatBar
          label="体力"
          value={hud.energy.opponent}
          max={fight.opponent.derived.maxEnergy}
          tone="energy"
          align="right"
        />
      </div>

      <div className="fight__stage">
        <PhaserGame />
        {phase === 'between' && (
          <div className="overlay">
            <div className="overlay__panel">
              <h2>第 {round} 回合结束 —— 重新配槽</h2>
              <p className="muted">
                对手也会在回合之间改自己的配置。看看刚才是被什么打疼的，再决定下一回合怎么打。
              </p>
              <LoadoutEditor compact />
              <button type="button" className="btn btn--primary btn--wide" onClick={playNextRound}>
                开始第 {round + 1} 回合
              </button>
            </div>
          </div>
        )}
      </div>

      <div className="fight__log" ref={logRef}>
        {log.map((line, i) => (
          <p key={i}>{line}</p>
        ))}
      </div>
    </div>
  );
}
