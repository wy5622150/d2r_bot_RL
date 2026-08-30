import Phaser from 'phaser';
import { useEffect, useRef } from 'react';
import { ARENA_HEIGHT, ARENA_WIDTH, ArenaScene } from './scenes/ArenaScene';

/**
 * 把 Phaser.Game 挂到一个 div 上。React 只负责生命周期，不碰场景内部。
 * StrictMode 下 effect 会跑两次，所以销毁必须干净。
 */
export function PhaserGame() {
  const hostRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;

    const game = new Phaser.Game({
      type: Phaser.AUTO,
      parent: host,
      backgroundColor: '#0f1218',
      scale: {
        mode: Phaser.Scale.FIT,
        autoCenter: Phaser.Scale.CENTER_BOTH,
        width: ARENA_WIDTH,
        height: ARENA_HEIGHT,
      },
      scene: [ArenaScene],
    });

    return () => game.destroy(true);
  }, []);

  return <div ref={hostRef} className="arena" />;
}
