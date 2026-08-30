/**
 * ⚠️ 本文件是整个战斗系统里**唯一**允许出现「非原版数值」的地方。
 *
 * 《拳击俱乐部》一代的战斗结算只有一部分被公开逆向出来（见 data/punch-club-source.md）。
 * 闪避率、格挡减伤、防守技能的选取算法、回合计时器时长、20 回合后的判胜公式等，
 * 官方与 wiki 都没有给出。这些缺口如果散落在各处，几个月后就分不清哪个数字来自原版、
 * 哪个是我们填的 —— 所以全部集中在这里，每条都注明来源状态。
 *
 * 规则：
 * 1. `src/core/` 的其它文件只允许使用**已确证**的原版公式，任何未知量必须从这里取。
 * 2. 每一项都要写清楚「原版怎么说的」和「我们为什么这么填」。
 * 3. 将来若逆向出真实公式，改这里一个文件即可。
 */

export interface Unknown<T> {
  value: T;
  /** 原版/资料对这一项的确切说法 */
  source: string;
  /** 我们填这个值的理由 */
  why: string;
}

const u = <T,>(value: T, source: string, why: string): Unknown<T> => ({ value, source, why });

export const UNKNOWNS = {
  // ---------------------------------------------------------------- 回合结构

  phasesPerRound: u(
    12,
    '资料：「回合内有一个倒计时器，双方交替进行攻防阶段」，但计时器时长与固定动作数均未找到。',
    '取一个能让 20 回合打出完整比赛节奏的整数；它只影响一场比赛的长度，不影响任何单次攻防的结算。',
  ),

  /**
   * 战斗中恢复的体力：在**自己每个阶段开始时**恢复一次。
   * 回复的时机（每回合一次 / 每阶段 / 按秒）同样没有公开，见下面的 why。
   */
  inFightRegen: u(
    (reg: number) => reg * 0.25,
    'REG = 5 + STM×1.5 是角色属性页显示值；wiki 明确说「战斗内实际浮动出来的回复数字更小」，且真实算法仍待测试。',
    'wiki 用「不断浮出的数字」描述战斗内回复，说明它是持续发生的小额回复而非每回合一次的大额回复 —— 所以放在每个阶段开始时，取 REG 的四分之一。倍率与时机都没有来源。',
  ),

  // ---------------------------------------------------------------- 攻防结算

  dodgeChance: u(
    (acc: number) => Math.min(0.15 + acc * 0.45, 0.75),
    '资料：闪避公式「明确未知」。只知道闪避成功则完全免伤，且 Technician / Fast Reaction 等被动会提高闪避率并使其更依赖敏捷。',
    '既然只知道「依赖敏捷」，就直接用已确证的 ACC（= 3·AGI/(STR+AGI+STM)）线性映射到一个有上下限的概率。这是占位，不是原版公式。',
  ),

  blockReduction: u(
    0.5,
    '资料：「格挡成功会减少受到的伤害」，但成功率与减伤量的公式都没找到。Iron Curtain「消耗更多体力、挡下更多伤害」，Rain of Strikes 使「对手格挡效果降低 25%」—— 说明减伤是个可被百分比削弱的量。',
    '取 50% 减伤，因为「可被削弱 25%/50%」的表述暗示它是一个比例而非固定值。数值本身是占位。',
  ),

  blockSuccessChance: u(
    1,
    '资料：格挡是否需要判定成功率未找到；只知道 Fine Block 会「提高格挡成功率」，说明存在成功率这个量。',
    '暂定抽中格挡即生效（成功率 100%），把不确定性收敛到 blockReduction 一处，避免两个未知量相乘后无法解释结果。',
  ),

  /** 抽中的防守技能能量不足时会怎样 */
  defenseFailsWhenBroke: u(
    true,
    '资料未说明体力不足以支付防守技能时的行为。',
    '与攻击一致处理：付不起就防不了。这是最保守的推断。',
  ),

  attackerDrawsAttacksOnly: u(
    true,
    '一代是**一组共享技能槽**（最多 5 个），开发者明确说槽位顺序无意义、招式自动/随机选取。但「攻击阶段抽到防守技能会怎样」没有说明。',
    '攻击阶段只从已装备的攻击技能里抽，防守阶段只从已装备的防守技能里抽。否则装了防守技能反而会浪费攻击阶段，与「防守技能有用」这一常识冲突。',
  ),

  /** 攻击阶段体力不足以打出任何技能时 */
  exhaustedPhaseRegen: u(
    5,
    '资料未说明打不出任何技能时的行为。Skip Attacks 技能「提高跳过攻击阶段并回复体力的概率」，说明存在「跳过攻击阶段 + 回体力」这一状态。',
    '借用 Skip Attacks 描述里的状态：打不出技能就视为跳过该阶段并回一点体力。数值是占位。',
  ),

  // ---------------------------------------------------------------- 击倒

  knockdownLostPhases: u(
    1,
    '资料：体力归零挨打会「额外 +10 伤害并被击倒」，但击倒持续多久、起身回多少体力都没找到。（Diehard 是独立被动：起身时有概率回 30% 体力。）',
    '取跳过 1 个阶段。这是能体现「被击倒有代价」的最小值。',
  ),

  knockdownGetUpEnergy: u(
    10,
    '同上：普通起身恢复的体力量未找到。',
    '给一个小额恢复，否则被击倒的一方会锁死在「体力 0 → 再次被击倒」的死循环里。',
  ),

  // ---------------------------------------------------------------- 判定

  decisionRule: u(
    'hpPercent' as const,
    '资料：「若 20 回合内双方都没有被击倒，由系统判定胜者」，但具体计分/比较公式没找到。',
    '按剩余血量百分比比较，相等判平局。这是最直观的读分方式，但没有来源支持。',
  ),

  /**
   * ARM 是「直接相减」还是「按比例减伤」。
   * 这是目前对战斗手感影响最大的一个未知项 —— 见下面 why 里的实测数据。
   */
  armorMode: u(
    'subtract' as 'subtract' | 'percent',
    'ARM = STM×1.3 是已确证的属性公式，但 wiki 对它「直接相减」的行为**明确标注为初步判断**（tentative），并未确认。',
    '默认取 wiki 字面所说的直接相减 —— 在拿到确凿证据前不擅自改默认值。但两种读法都不像真实游戏，实测（各 600 场）：\n' +
      '  subtract：KO 率 0%，每一场都拖到 20 回合读分。原因是 5/5/5 时 Punch 原始伤害 5、对手 ARM 6.5，相减后触发下限只剩 1 点，而血量有 203。\n' +
      '  percent（按 1 − ARM/100 折算）：KO 率 100%，平均 11.6~15.6 回合结束。\n' +
      '真实的一代两种结局都常见，说明真正的公式在这两者之间 —— 可能护甲有上限、或伤害另有加成、或血量口径不同。这是目前最值得优先查证的一项。',
  ),

  minDamage: u(
    1,
    '资料未说明护甲高于伤害时的下限。ARM = STM×1.3，wiki 说「直接相减」的行为本身也只是初步判断。',
    '命中至少造成 1 点，避免高耐力选手完全免疫。',
  ),

  damageRounding: u(
    'round' as const,
    '资料：伤害「base + 系数×STR」，示例显示常规四舍五入（4.5→5、3.1→3）。护甲相减后如何取整未说明。',
    '相减后同样用四舍五入，与伤害本身的取整口径保持一致。',
  ),
} as const;

/** 取值的简写，读起来比 UNKNOWNS.x.value 顺 */
export const UNK = {
  phasesPerRound: UNKNOWNS.phasesPerRound.value,
  inFightRegen: UNKNOWNS.inFightRegen.value,
  dodgeChance: UNKNOWNS.dodgeChance.value,
  blockReduction: UNKNOWNS.blockReduction.value,
  blockSuccessChance: UNKNOWNS.blockSuccessChance.value,
  defenseFailsWhenBroke: UNKNOWNS.defenseFailsWhenBroke.value,
  attackerDrawsAttacksOnly: UNKNOWNS.attackerDrawsAttacksOnly.value,
  exhaustedPhaseRegen: UNKNOWNS.exhaustedPhaseRegen.value,
  knockdownLostPhases: UNKNOWNS.knockdownLostPhases.value,
  knockdownGetUpEnergy: UNKNOWNS.knockdownGetUpEnergy.value,
  armorMode: UNKNOWNS.armorMode.value,
  minDamage: UNKNOWNS.minDamage.value,
};
