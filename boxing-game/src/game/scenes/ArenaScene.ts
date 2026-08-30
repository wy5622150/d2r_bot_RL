import Phaser from 'phaser';
import { getMove, tryGetMove } from '../../core/moves';
import type { RoundEvent, Side } from '../../core/types';
import { busState, EventBus, type FightSetup } from '../EventBus';
import { BoxerView } from '../view/BoxerView';

export const ARENA_WIDTH = 960;
export const ARENA_HEIGHT = 540;
const FLOOR_Y = 462;
const PLAYER_X = 402;
const OPPONENT_X = 558;

const FONT = 'ui-sans-serif, system-ui, "PingFang SC", "Microsoft YaHei", sans-serif';

/**
 * 只做一件事：把引擎算好的 RoundEvent[] 按时间线演出来。
 * 这里没有任何战斗规则 —— 血量、命中、胜负全部已经在 core/ 里定好了。
 */
export class ArenaScene extends Phaser.Scene {
  private boxers: Record<Side, BoxerView> | null = null;
  private labels: Phaser.GameObjects.Text[] = [];
  private banner!: Phaser.GameObjects.Text;
  private queue: RoundEvent[] = [];
  private cursor = 0;
  private speed = 1;
  private pending?: Phaser.Time.TimerEvent;

  constructor() {
    super('Arena');
  }

  create(): void {
    this.drawArena();

    this.banner = this.add
      .text(ARENA_WIDTH / 2, 150, '', {
        fontFamily: FONT,
        fontSize: '46px',
        color: '#ffffff',
        fontStyle: 'bold',
      })
      .setOrigin(0.5)
      .setAlpha(0);

    EventBus.on('fight:setup', this.onSetup, this);
    EventBus.on('fight:play-round', this.playRound, this);
    EventBus.on('fight:set-speed', this.onSpeed, this);
    EventBus.on('fight:skip', this.skip, this);

    this.events.once(Phaser.Scenes.Events.SHUTDOWN, () => {
      busState.sceneReady = false;
      EventBus.off('fight:setup', this.onSetup);
      EventBus.off('fight:play-round', this.playRound);
      EventBus.off('fight:set-speed', this.onSpeed);
      EventBus.off('fight:skip', this.skip);
    });

    busState.sceneReady = true;
    EventBus.emit('scene:ready');
  }

  // ---------------------------------------------------------------- 场景

  private drawArena(): void {
    this.add.rectangle(ARENA_WIDTH / 2, ARENA_HEIGHT / 2, ARENA_WIDTH, ARENA_HEIGHT, 0x0f1218);

    // 看台上的人影
    const crowd = this.add.container(0, 0);
    for (let i = 0; i < 90; i++) {
      const x = Phaser.Math.Between(20, ARENA_WIDTH - 20);
      const y = Phaser.Math.Between(60, 235);
      const shade = Phaser.Display.Color.GetColor(
        26 + Phaser.Math.Between(0, 26),
        28 + Phaser.Math.Between(0, 26),
        38 + Phaser.Math.Between(0, 30),
      );
      crowd.add(this.add.circle(x, y, Phaser.Math.Between(5, 9), shade));
    }

    // 顶灯
    this.add.ellipse(ARENA_WIDTH / 2, 250, 900, 320, 0x2a3446, 0.35);

    // 擂台
    this.add.rectangle(ARENA_WIDTH / 2, 300, 880, 130, 0x171c26);
    this.add.rectangle(ARENA_WIDTH / 2, 492, 940, 130, 0x323b4d);
    this.add.rectangle(ARENA_WIDTH / 2, 462, 900, 14, 0x4a566e);

    // 远端围绳
    for (const [y, color] of [
      [286, 0xc7473f],
      [318, 0xe8e8e8],
      [350, 0x3f6bc7],
    ] as const) {
      this.add.rectangle(ARENA_WIDTH / 2, y, 880, 5, color, 0.85);
    }
    // 角柱
    this.add.rectangle(46, 320, 16, 200, 0x5b6478);
    this.add.rectangle(ARENA_WIDTH - 46, 320, 16, 200, 0x5b6478);
  }

  private onSetup(setup: FightSetup): void {
    this.clearFight();
    this.boxers = {
      player: new BoxerView(this, PLAYER_X, FLOOR_Y, setup.player.color, 1),
      opponent: new BoxerView(this, OPPONENT_X, FLOOR_Y, setup.opponent.color, -1),
    };
    this.labels = (
      [
        [PLAYER_X, setup.player.name, setup.player.color],
        [OPPONENT_X, setup.opponent.name, setup.opponent.color],
      ] as const
    ).map(([x, name, color]) =>
      this.add
        .text(x, FLOOR_Y + 26, name, {
          fontFamily: FONT,
          fontSize: '16px',
          color: '#' + color.toString(16).padStart(6, '0'),
        })
        .setOrigin(0.5),
    );
  }

  private clearFight(): void {
    this.pending?.remove();
    this.pending = undefined;
    this.queue = [];
    this.cursor = 0;
    this.boxers?.player.root.destroy();
    this.boxers?.opponent.root.destroy();
    this.boxers = null;
    this.labels.forEach((l) => l.destroy());
    this.labels = [];
  }

  private onSpeed(speed: number): void {
    this.speed = speed;
  }

  // ---------------------------------------------------------------- 回放

  private playRound(events: RoundEvent[]): void {
    this.pending?.remove();
    this.queue = events;
    this.cursor = 0;
    this.step();
  }

  /** 跳过剩下的表演，但仍然把事件全部抛给 HUD，保证界面数字停在正确的终点 */
  private skip(): void {
    this.pending?.remove();
    this.pending = undefined;
    while (this.cursor < this.queue.length) {
      const e = this.queue[this.cursor++]!;
      EventBus.emit('fight:event-shown', e);
      if (e.type === 'knockdown' || e.type === 'ko') this.boxers?.[e.side]?.fall(4);
    }
    this.finishRound();
  }

  private step(): void {
    if (this.cursor >= this.queue.length) {
      this.finishRound();
      return;
    }
    const event = this.queue[this.cursor++]!;
    EventBus.emit('fight:event-shown', event);
    const hold = this.render(event);
    this.pending = this.time.delayedCall(Math.max(40, hold / this.speed), () => this.step());
  }

  private finishRound(): void {
    const last = this.queue[this.queue.length - 1];
    const round = last && last.type === 'round_end' ? last.round : 0;
    this.queue = [];
    this.cursor = 0;
    EventBus.emit('fight:round-done', round);
  }

  /** 返回这条事件在 1x 速度下应该占用的毫秒数 */
  private render(e: RoundEvent): number {
    const b = this.boxers;
    if (!b) return 60;
    const spd = this.speed;

    switch (e.type) {
      case 'round_start':
        this.showBanner(`第 ${e.round} 回合`);
        return 900;

      case 'turn_switch':
        return 260;

      case 'attack': {
        const attacker = b[e.side];
        const defender = b[e.side === 'player' ? 'opponent' : 'player'];
        const defMove = tryGetMove(e.defenseMove);
        if (defMove) defender.defend(defMove.anim, spd);
        return attacker.punch(getMove(e.move).anim, spd, () => {});
      }

      case 'hit': {
        const target = b[e.target];
        target.hurt(e.damage, spd);
        this.floatText(
          e.target,
          `-${e.damage}`,
          e.crit ? '#ff5252' : '#ffffff',
          e.crit ? 32 : 24,
        );
        if (e.crit || e.exhaustBonus) this.cameras.main.shake(180 / spd, 0.009);
        if (e.blocked > 0 && !e.exhaustBonus) this.floatText(e.target, `挡下 ${e.blocked}`, '#8fd3ff', 15, 40);
        return 320;
      }

      case 'counter':
        b[e.target].hurt(e.damage, spd);
        this.floatText(e.target, `反击 -${e.damage}`, '#ffd166', 22);
        return 300;

      case 'dodge':
        this.floatText(e.side, '闪开', '#8fd3ff', 20);
        return 260;

      case 'miss':
        b[e.side].stumble(spd);
        this.floatText(e.side, '打空', '#9aa4b2', 20);
        return 260;

      case 'rest':
      case 'empty_slot':
        b[e.side].breathe(spd);
        if (e.energyGain > 0) this.floatText(e.side, `+${e.energyGain} 体力`, '#ffd166', 18);
        return 320;

      case 'exhausted':
        b[e.side].stumble(spd);
        this.floatText(e.side, '力竭', '#ff9f43', 20);
        return 320;

      case 'stun':
        this.floatText(e.side, '被打懵', '#ff8fab', 20);
        return 280;

      case 'knockdown':
        b[e.side].fall(spd);
        this.floatText(e.side, '倒地！', '#ff5252', 26);
        this.cameras.main.shake(260 / spd, 0.012);
        return 700;

      case 'skip':
        if (e.cause === 'knockdown') b[e.side].standUp(spd);
        this.floatText(e.side, e.cause === 'knockdown' ? '爬起来' : '还没缓过来', '#9aa4b2', 18);
        return 380;

      case 'ko':
        b[e.side].fall(spd);
        this.showBanner('K.O.');
        this.cameras.main.shake(420 / spd, 0.016);
        return 1400;

      case 'round_end':
        this.showBanner(`第 ${e.round} 回合结束`, 28);
        return 700;
    }
  }

  // ---------------------------------------------------------------- 小部件

  private showBanner(text: string, size = 46): void {
    this.banner.setText(text).setFontSize(size).setAlpha(0).setScale(0.85);
    this.tweens.add({
      targets: this.banner,
      alpha: 1,
      scale: 1,
      duration: 200 / this.speed,
      yoyo: true,
      hold: 500 / this.speed,
      ease: 'Quad.easeOut',
    });
  }

  private floatText(side: Side, text: string, color: string, size: number, offsetY = 0): void {
    const x = side === 'player' ? PLAYER_X : OPPONENT_X;
    const label = this.add
      .text(x + Phaser.Math.Between(-14, 14), FLOOR_Y - 210 + offsetY, text, {
        fontFamily: FONT,
        fontSize: `${size}px`,
        color,
        fontStyle: 'bold',
      })
      .setOrigin(0.5);
    this.tweens.add({
      targets: label,
      y: label.y - 46,
      alpha: 0,
      duration: 700 / this.speed,
      ease: 'Quad.easeOut',
      onComplete: () => label.destroy(),
    });
  }
}
