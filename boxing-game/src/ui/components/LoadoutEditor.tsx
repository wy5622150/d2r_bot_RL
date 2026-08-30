import { useState } from 'react';
import { PLAYER } from '../../core/fighters';
import { MOVES, tryGetMove } from '../../core/moves';
import { DEFENSE_SLOTS, OFFENSE_SLOTS } from '../../core/stats';
import type { Loadout, Move } from '../../core/types';
import { useGame } from '../store';
import { moveStats } from './MoveCard';

type SlotKind = 'offense' | 'defense';
interface Selection {
  kind: SlotKind;
  index: number;
}

const POOL = MOVES.filter((m) => PLAYER.pool.includes(m.id));

function allowedIn(kind: SlotKind, move: Move): boolean {
  return kind === 'offense' ? move.kind !== 'defense' : move.kind === 'defense';
}

function Slot({
  move,
  selected,
  onClick,
}: {
  move: Move | null;
  selected: boolean;
  onClick: () => void;
}) {
  return (
    <button type="button" className={`slot${selected ? ' is-selected' : ''}`} onClick={onClick}>
      {move ? (
        <>
          <span className="slot__name">{move.name}</span>
          <span className="slot__cost">体力 {move.energyCost}</span>
        </>
      ) : (
        <>
          <span className="slot__name slot__name--empty">空槽</span>
          <span className="slot__cost">回体力</span>
        </>
      )}
    </button>
  );
}

/**
 * 配槽面板。战前和每个回合之间都用同一个组件 —— 这正是原作里玩家唯一的操作。
 * 交互：先点一个槽位，再点下面的招式填进去；重复放同一招 = 出现频率更高。
 */
export function LoadoutEditor({ compact = false }: { compact?: boolean }) {
  const loadout = useGame((s) => s.loadout);
  const setSlot = useGame((s) => s.setSlot);
  const resetLoadout = useGame((s) => s.resetLoadout);
  const [sel, setSel] = useState<Selection>({ kind: 'offense', index: 0 });

  const pick = (move: Move | null) => {
    setSlot(sel.kind, sel.index, move?.id ?? null);
    const len = sel.kind === 'offense' ? OFFENSE_SLOTS : DEFENSE_SLOTS;
    setSel({ kind: sel.kind, index: (sel.index + 1) % len });
  };

  const render = (kind: SlotKind, slots: Loadout['offense'], count: number) =>
    Array.from({ length: count }, (_, i) => (
      <Slot
        key={i}
        move={tryGetMove(slots[i] ?? null)}
        selected={sel.kind === kind && sel.index === i}
        onClick={() => setSel({ kind, index: i })}
      />
    ));

  return (
    <div className={`loadout${compact ? ' loadout--compact' : ''}`}>
      <div className="loadout__slots">
        <section>
          <h3>
            进攻槽 <small>按顺序循环出手，空槽 = 喘口气回体力</small>
          </h3>
          <div className="slotrow">{render('offense', loadout.offense, OFFENSE_SLOTS)}</div>
        </section>
        <section>
          <h3>
            防守槽 <small>挨打时随机抽一个应对，留空 = 有概率完全不设防</small>
          </h3>
          <div className="slotrow">{render('defense', loadout.defense, DEFENSE_SLOTS)}</div>
        </section>
      </div>

      <div className="loadout__palette">
        <div className="palette__head">
          <span>
            正在编辑：{sel.kind === 'offense' ? '进攻' : '防守'}槽 {sel.index + 1}
          </span>
          <div>
            <button type="button" className="btn btn--ghost" onClick={() => pick(null)}>
              清空该槽
            </button>
            <button type="button" className="btn btn--ghost" onClick={resetLoadout}>
              恢复默认
            </button>
          </div>
        </div>
        <div className="palette__grid">
          {POOL.map((move) => {
            const ok = allowedIn(sel.kind, move);
            return (
              <button
                key={move.id}
                type="button"
                className={`palette__item palette__item--${move.kind}`}
                disabled={!ok}
                title={move.desc}
                onClick={() => pick(move)}
              >
                <span className="palette__name">{move.name}</span>
                <span className="palette__stats">{moveStats(move).join(' · ')}</span>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
