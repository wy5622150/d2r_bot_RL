/**
 * mulberry32：32 位状态的确定性伪随机数发生器。
 * 状态是单个 number，可直接存进 FightState 并序列化 —— 这是"同 seed 同战报"的基础。
 */

export interface Rng {
  /** 返回 [0, 1) */
  next(): number;
  /** 返回 [0, n) 的整数 */
  int(n: number): number;
  /** p 概率为 true */
  chance(p: number): boolean;
  /** 从数组里等概率取一个 */
  pick<T>(arr: readonly T[]): T;
  /** 取出当前状态，写回 FightState */
  state(): number;
}

export function createRng(seed: number): Rng {
  let s = seed >>> 0;

  const next = (): number => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };

  return {
    next,
    int: (n) => Math.floor(next() * n),
    chance: (p) => next() < p,
    pick: <T,>(arr: readonly T[]): T => {
      const v = arr[Math.floor(next() * arr.length)];
      if (v === undefined) throw new Error('pick() 收到空数组');
      return v;
    },
    state: () => s,
  };
}

/** 把字符串转成种子，方便用对手名字之类的东西当 seed */
export function hashSeed(text: string): number {
  let h = 2166136261;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}
