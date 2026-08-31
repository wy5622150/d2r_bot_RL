import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    // core/ 是纯 TS，不需要浏览器环境
    environment: 'node',
    include: ['src/core/**/*.test.ts'],
  },
});
