#!/usr/bin/env node
/**
 * 把 phaser 官方 npm 包自带的 Claude Code skills 同步到 boxing-game/.claude/skills/。
 *
 * 这些 skill 由 Phaser Studio 随 phaser 包一起发布（node_modules/phaser/skills/），
 * 是官方文档而非第三方整理。升级 phaser 版本后重跑本脚本即可保持一致：
 *
 *   node scripts/sync-phaser-skills.mjs
 */
import { cp, mkdir, readFile, rm, writeFile, readdir } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const src = join(root, 'node_modules', 'phaser', 'skills');
const dest = join(root, '.claude', 'skills');

if (!existsSync(src)) {
  console.error(`找不到 ${src}，先运行 npm install`);
  process.exit(1);
}

const pkg = JSON.parse(await readFile(join(root, 'node_modules', 'phaser', 'package.json'), 'utf8'));

await rm(dest, { recursive: true, force: true });
await mkdir(dest, { recursive: true });
await cp(src, dest, { recursive: true });

const names = (await readdir(dest, { withFileTypes: true }))
  .filter((e) => e.isDirectory())
  .map((e) => e.name)
  .sort();

await writeFile(
  join(dest, 'SOURCE.md'),
  `# 来源说明

本目录下的 skill 全部原样拷贝自官方 \`phaser\` npm 包的 \`skills/\` 目录，
由 Phaser Studio Inc. 随引擎一起发布，MIT 协议。

- 来源包：\`phaser@${pkg.version}\`（release "${pkg.release ?? '-'}"）
- 同步方式：\`node scripts/sync-phaser-skills.mjs\`
- 请勿手工修改这里的文件，升级 phaser 后重跑上面的脚本即可。

共 ${names.length} 个 skill：

${names.map((n) => `- \`${n}\``).join('\n')}
`,
  'utf8',
);

console.log(`已同步 ${names.length} 个官方 Phaser skill（phaser@${pkg.version}）到 .claude/skills/`);
