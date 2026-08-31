import Phaser from 'phaser';
import type { AnimKey } from '../../core/types';

/** 出拳动作的表现参数：幅度越大的拳，前摇越长、身体前压越多 */
const PUNCH_SHAPE: Record<string, { reach: number; lunge: number; windup: number; arc: number }> = {
  punch: { reach: 58, lunge: 10, windup: 60, arc: 0 },
  high_punch: { reach: 64, lunge: 16, windup: 95, arc: -12 },
  uppercut: { reach: 52, lunge: 18, windup: 120, arc: -26 },
  crosspunch: { reach: 68, lunge: 20, windup: 110, arc: -6 },
  backhand: { reach: 66, lunge: 18, windup: 130, arc: -18 },
  kick: { reach: 62, lunge: 14, windup: 90, arc: 22 },
  high_kick: { reach: 70, lunge: 20, windup: 140, arc: -16 },
  low_kick: { reach: 58, lunge: 12, windup: 80, arc: 34 },
  knee: { reach: 46, lunge: 16, windup: 90, arc: 12 },
  chop: { reach: 60, lunge: 16, windup: 100, arc: -8 },
};

function shade(color: number, factor: number): number {
  const c = Phaser.Display.Color.IntegerToColor(color);
  return Phaser.Display.Color.GetColor(
    Phaser.Math.Clamp(Math.round(c.red * factor), 0, 255),
    Phaser.Math.Clamp(Math.round(c.green * factor), 0, 255),
    Phaser.Math.Clamp(Math.round(c.blue * factor), 0, 255),
  );
}

/**
 * 程序化生成的拳手：全部用 Phaser 的 Shape 画出来，不依赖任何外部素材。
 * 换真素材时，把这个类换成 Sprite + 序列帧即可，接口（punch/defend/hurt/...）保持不变。
 */
export class BoxerView {
  readonly root: Phaser.GameObjects.Container;

  private readonly scene: Phaser.Scene;
  private readonly facing: 1 | -1;
  private readonly baseX: number;
  private readonly color: number;

  private readonly body: Phaser.GameObjects.Container;
  private readonly torso: Phaser.GameObjects.Rectangle;
  private readonly head: Phaser.GameObjects.Arc;
  private readonly frontGlove: Phaser.GameObjects.Arc;
  private readonly backGlove: Phaser.GameObjects.Arc;
  private readonly legFront: Phaser.GameObjects.Rectangle;
  private readonly legBack: Phaser.GameObjects.Rectangle;
  private readonly shadow: Phaser.GameObjects.Ellipse;

  private readonly gloveHome: { x: number; y: number };
  private readonly backGloveHome: { x: number; y: number };
  private bob?: Phaser.Tweens.Tween;
  private down = false;

  constructor(scene: Phaser.Scene, x: number, y: number, color: number, facing: 1 | -1) {
    this.scene = scene;
    this.facing = facing;
    this.baseX = x;
    this.color = color;

    const dark = shade(color, 0.62);
    const skin = 0xe8c9a0;

    this.shadow = scene.add.ellipse(0, 4, 74, 16, 0x000000, 0.28);
    this.body = scene.add.container(0, 0);

    this.legBack = scene.add.rectangle(-facing * 12, -24, 13, 50, dark).setOrigin(0.5, 0.5);
    this.legFront = scene.add.rectangle(facing * 12, -24, 13, 50, shade(color, 0.8));
    this.torso = scene.add.rectangle(0, -76, 42, 62, color).setStrokeStyle(2, dark);
    this.head = scene.add.circle(facing * 4, -118, 16, skin).setStrokeStyle(2, dark);
    this.backGlove = scene.add.circle(-facing * 16, -92, 11, dark);
    this.frontGlove = scene.add.circle(facing * 20, -96, 12, shade(color, 1.25));

    this.gloveHome = { x: this.frontGlove.x, y: this.frontGlove.y };
    this.backGloveHome = { x: this.backGlove.x, y: this.backGlove.y };

    this.body.add([
      this.legBack,
      this.legFront,
      this.torso,
      this.backGlove,
      this.head,
      this.frontGlove,
    ]);
    this.root = scene.add.container(x, y, [this.shadow, this.body]);
    this.root.setScale(1.25);
    this.startBob();
  }

  private startBob(): void {
    this.bob?.remove();
    this.bob = this.scene.tweens.add({
      targets: this.body,
      y: -5,
      duration: 900,
      yoyo: true,
      repeat: -1,
      ease: 'Sine.easeInOut',
    });
  }

  /** 出拳。onImpact 在拳头到位的那一刻触发，用来同步受击表现。 */
  punch(anim: AnimKey, speed: number, onImpact: () => void): number {
    const s = PUNCH_SHAPE[anim] ?? PUNCH_SHAPE['punch']!;
    const windup = s.windup / speed;
    const strike = 90 / speed;

    // 前摇：收拳、身体后压
    this.scene.tweens.add({
      targets: this.frontGlove,
      x: this.gloveHome.x - this.facing * 14,
      y: this.gloveHome.y + s.arc * 0.4,
      duration: windup,
      ease: 'Sine.easeOut',
    });
    this.scene.tweens.add({
      targets: this.root,
      x: this.baseX - this.facing * 6,
      duration: windup,
      ease: 'Sine.easeOut',
    });

    // 击出
    this.scene.time.delayedCall(windup, () => {
      this.scene.tweens.add({
        targets: this.frontGlove,
        x: this.gloveHome.x + this.facing * s.reach,
        y: this.gloveHome.y + s.arc,
        duration: strike,
        ease: 'Quad.easeOut',
        onComplete: () => {
          onImpact();
          this.scene.tweens.add({
            targets: this.frontGlove,
            x: this.gloveHome.x,
            y: this.gloveHome.y,
            duration: 150 / speed,
            ease: 'Quad.easeIn',
          });
        },
      });
      this.scene.tweens.add({
        targets: this.root,
        x: this.baseX + this.facing * s.lunge,
        duration: strike,
        ease: 'Quad.easeOut',
        yoyo: true,
        hold: 40 / speed,
      });
    });

    return windup + strike + 150 / speed;
  }

  /** 摆出防守姿态 */
  defend(anim: AnimKey, speed: number): void {
    if (this.down) return;
    const d = 140 / speed;
    if (anim === 'dodge') {
      this.scene.tweens.add({
        targets: this.body,
        x: -this.facing * 16,
        angle: -this.facing * 10,
        duration: d,
        yoyo: true,
        ease: 'Sine.easeInOut',
      });
      return;
    }
    // Block：双手护到面门前
    const up = 14;
    this.scene.tweens.add({
      targets: [this.frontGlove, this.backGlove],
      x: this.facing * 16,
      y: `-=${up}`,
      duration: d,
      yoyo: true,
      hold: 80 / speed,
      ease: 'Sine.easeInOut',
      onComplete: () => this.resetGloves(),
    });
  }

  private resetGloves(): void {
    this.frontGlove.setPosition(this.gloveHome.x, this.gloveHome.y);
    this.backGlove.setPosition(this.backGloveHome.x, this.backGloveHome.y);
  }

  /** 受击：后仰 + 闪红 */
  hurt(power: number, speed: number): void {
    if (this.down) return;
    const push = Phaser.Math.Clamp(6 + power * 0.5, 6, 26);
    this.scene.tweens.add({
      targets: this.root,
      x: this.baseX - this.facing * push,
      duration: 90 / speed,
      yoyo: true,
      ease: 'Quad.easeOut',
    });
    this.scene.tweens.add({
      targets: this.body,
      angle: -this.facing * Phaser.Math.Clamp(power * 0.35, 3, 14),
      duration: 90 / speed,
      yoyo: true,
      ease: 'Quad.easeOut',
    });
    this.torso.setFillStyle(0xffffff);
    this.head.setFillStyle(0xffdede);
    this.scene.time.delayedCall(110 / speed, () => {
      this.torso.setFillStyle(this.color);
      this.head.setFillStyle(0xe8c9a0);
    });
  }

  /** 打空后的失衡 */
  stumble(speed: number): void {
    this.scene.tweens.add({
      targets: this.body,
      angle: this.facing * 8,
      duration: 160 / speed,
      yoyo: true,
      ease: 'Sine.easeInOut',
    });
  }

  /** 喘气回体力 */
  breathe(speed: number): void {
    this.scene.tweens.add({
      targets: this.torso,
      scaleY: 1.12,
      duration: 220 / speed,
      yoyo: true,
      repeat: 1,
      ease: 'Sine.easeInOut',
    });
  }

  fall(speed: number): void {
    if (this.down) return;
    this.down = true;
    this.bob?.remove();
    this.scene.tweens.add({
      targets: this.body,
      angle: -this.facing * 82,
      y: 30,
      x: -this.facing * 26,
      duration: 320 / speed,
      ease: 'Quad.easeIn',
    });
  }

  standUp(speed: number): void {
    if (!this.down) return;
    this.down = false;
    this.scene.tweens.add({
      targets: this.body,
      angle: 0,
      y: 0,
      x: 0,
      duration: 380 / speed,
      ease: 'Back.easeOut',
      onComplete: () => this.startBob(),
    });
  }

  get isDown(): boolean {
    return this.down;
  }
}
