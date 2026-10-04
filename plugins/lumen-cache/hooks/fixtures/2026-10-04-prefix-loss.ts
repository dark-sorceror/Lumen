import type { Fixture } from '../record'

/**
 * Live capture: five main-thread requests, copied unmodified from the store
 * (session 664b793a), all four counts as the engine reported them. Every
 * request carries `uncached: 2` — input neither read from nor written to the
 * cache — and it stays separate from `written`, because the cache never held it.
 *
 * Request 5 (a new turn, the message count falling 41 -> 32) was served 27,221
 * tokens of the 51,548 the cache held after request 4: 24,327 lost. No
 * compaction time was recorded for this session, so the cause is not claimed.
 */
export const fixture: Fixture = {
  name: '2026-10-04-prefix-loss',
  capturedAt: '2026-10-04',
  note: 'Live session 664b793a, all 5 requests verbatim; no compaction recorded, so the one collapse is unattributed.',
  compactions: [],
  samples: [
    { turnId: '57b87eaa-9e48-44ea-98bd-8b3b977cd13b', index: 0, model: 'claude-opus-5-5', reused: 25_391, written: 20_708, uncached: 2, output: 76, at: 1791088415539, messageCount: 21, stopReason: 'tool_use', toolUseCount: 1, effort: 'medium' },
    { turnId: '57b87eaa-9e48-44ea-98bd-8b3b977cd13b', index: 1, model: 'claude-opus-5-5', reused: 46_099, written: 2_459, uncached: 2, output: 1410, at: 1791088430495, messageCount: 27, stopReason: 'end_turn', toolUseCount: 0, effort: 'medium' },
    { turnId: '73b5b6fc-6dc4-405f-b617-ae068f112f7a', index: 0, model: 'claude-opus-5-5', reused: 49_968, written: 56, uncached: 2, output: 787, at: 1791088452963, messageCount: 36, stopReason: 'tool_use', toolUseCount: 1, effort: 'medium' },
    { turnId: '73b5b6fc-6dc4-405f-b617-ae068f112f7a', index: 1, model: 'claude-opus-5-5', reused: 50_024, written: 1_524, uncached: 2, output: 993, at: 1791088464905, messageCount: 41, stopReason: 'end_turn', toolUseCount: 0, effort: 'medium' },
    { turnId: '2318ce9a-a6b4-41f3-96d9-f26e6bb52bf1', index: 0, model: 'claude-opus-5-5', reused: 27_221, written: 14_475, uncached: 2, output: 831, at: 1791088553347, messageCount: 32, stopReason: 'end_turn', toolUseCount: 0, effort: 'medium' },
  ],
  expected: {
    requests: 5,
    reused: 198_703,
    reprocessed: 39_232,
    collapses: [{ lost: 24_327, cause: 'unattributed' }],
  },
}
