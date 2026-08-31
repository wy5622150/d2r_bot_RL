import { useState } from 'react';
import { validateLoadout } from '../../core/ai';
import { PLAYER } from '../../core/fighters';
import { MOVES, tryGetMove } from '../../core/moves';
import { ABILITY_SLOTS_MAX, derive } from '../../core/stats';
import type { Move } from '../../core/types';
import { useGame } from '../store';
import { actualNumbers } from './MoveCard';

const POOL = MOVES.filter((m) => PLAYER.pool.includes(m.id));
const STATS = PLAYER.stats;
const DERIVED = derive(PLAYER.stats, PLAYER.health);

function Slot({
  move,
  index,
  selected,
  onClick,
}: {
  move: Move | null;
  index: number;
  selected: boolean;
  onClick: () => void;
}) {
  return (
    <button type="button" className={`slot${selected ? ' is-selected' : ''}`} onClick={onClick}>
      <span className="slot__idx">{index + 1}</span>
      {move ? (
        <>
          <span className="slot__name">{move.nameEn}</span>
          <span className={`slot__kind slot__kind--${move.kind}`}>
            {move.kind === 'attack' ? '攻击' : '防守'}
          </span>
        </>
      ) : (
        <span className="slot__name slot__name--empty">未装备</span>
      )}
    </button>
  );
}

/**
 * 技能配置面板。战前和每个回合之间都用同一个组件 —— 这是一代里玩家唯一的操作。
 *
 * 与二代不同：一代是**一组共享槽位**（最多 5 个），攻击与防守技能混装，
 * 且开发者明确说过槽位顺序没有意义 —— 所以这里不做任何"顺序即连招"的暗示。
 */
export function LoadoutEditor({ compact = false }: { compact?: boolean }) {
  const loadout = useGame((s) => s.loadout);
  const setSlot = useGame((s) => s.setSlot);
  const resetLoadout = useGame((s) => s.resetLoadout);
  const [sel, setSel] = useState(0);

  const error = validateLoadout(loadout, PLAYER.pool);
  const counts = loadout.reduce<Record<string, number>>((acc, id) => {
    acc[id] = (acc[id] ?? 0) + 1;
    return acc;
  }, {});

  const pick = (move: Move) => {
    setSlot(sel, move.id);
    setSel((i) => (i + 1) % ABILITY_SLOTS_MAX);
  };

  return (
    <div className={`loadout${compact ? ' loadout--compact' : ''}`}>
      <section>
        <h3>
          技能槽 <small>共 {ABILITY_SLOTS_MAX} 个，攻守混装。出招是随机抽取，槽位顺序没有意义</small>
        </h3>
        <div className="slotrow">
          {Array.from({ length: ABILITY_SLOTS_MAX }, (_, i) => (
            <Slot
              key={i}
              index={i}
              move={tryGetMove(loadout[i] ?? null)}
              selected={sel === i}
              onClick={() => setSel(i)}
            />
          ))}
        </div>
        {error ? (
          <p className="warn">⚠ {error}</p>
        ) : (
          <p className="muted">
            同一个技能装多份会等比例提高它被抽中的概率
            {Object.entries(counts).some(([, n]) => n > 1)
              ? ` —— 当前：${Object.entries(counts)
                  .filter(([, n]) => n > 1)
                  .map(([id, n]) => `${tryGetMove(id)?.nameEn} ×${n}`)
                  .join('、')}`
              : ''}
          </p>
        )}
      </section>

      <div className="loadout__palette">
        <div className="palette__head">
          <span>正在编辑：第 {sel + 1} 个技能槽</span>
          <button type="button" className="btn btn--ghost" onClick={resetLoadout}>
            恢复默认
          </button>
        </div>
        <div className="palette__grid">
          {POOL.map((move) => (
            <button
              key={move.id}
              type="button"
              className={`palette__item palette__item--${move.kind}`}
              title={move.effects.join(' ')}
              onClick={() => pick(move)}
            >
              <span className="palette__name">
                {move.nameEn}
                <em>{move.school}</em>
                {move.conflict && <span className="palette__flag" title={move.conflict}>⚑</span>}
              </span>
              <span className="palette__stats">
                {actualNumbers(move, STATS, DERIVED).join(' · ')}
              </span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
