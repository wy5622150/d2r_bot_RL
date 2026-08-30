import Phaser from 'phaser';
import type { RoundEvent, Side } from '../core/types';

export interface FighterView {
  name: string;
  color: number;
  side: Side;
}

export interface FightSetup {
  player: FighterView;
  opponent: FighterView;
  hp: Record<Side, number>;
  maxHp: Record<Side, number>;
}

/** React ↔ Phaser 的唯一通道。两边都不直接持有对方的对象。 */
export interface BusEvents {
  // React → Phaser
  'fight:setup': (setup: FightSetup) => void;
  'fight:play-round': (events: RoundEvent[]) => void;
  'fight:set-speed': (speed: number) => void;
  'fight:skip': () => void;
  // Phaser → React
  'scene:ready': () => void;
  /** 画面播到了哪一条事件 —— HUD 据此更新，保证数字和画面同步 */
  'fight:event-shown': (event: RoundEvent) => void;
  'fight:round-done': (round: number) => void;
}

class TypedBus extends Phaser.Events.EventEmitter {
  override emit<K extends keyof BusEvents>(
    event: K,
    ...args: Parameters<BusEvents[K]>
  ): boolean {
    return super.emit(event as string, ...args);
  }

  override on<K extends keyof BusEvents>(event: K, fn: BusEvents[K], context?: unknown): this {
    return super.on(event as string, fn as (...a: unknown[]) => void, context);
  }

  override off<K extends keyof BusEvents>(event: K, fn?: BusEvents[K]): this {
    return super.off(event as string, fn as ((...a: unknown[]) => void) | undefined);
  }
}

export const EventBus = new TypedBus();

/**
 * 场景是否已经 create 完毕。
 * React 侧挂载监听和 Phaser boot 是两条时间线，用这个标记兜住"监听注册晚了"的情况。
 */
export const busState = { sceneReady: false };
