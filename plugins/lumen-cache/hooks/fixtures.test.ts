import { test, expect } from 'claude-code/testing'

import { type Fixture, replay } from './record'
import { fixture as prefixLoss } from './fixtures/2026-10-04-prefix-loss'

/** Every captured session. An export prints the line to add here. */
const FIXTURES: Fixture[] = [prefixLoss]

for (const f of FIXTURES) {
  test(`${f.name}: replays to the verdict it was captured with`, () => {
    expect(replay(f)).toEqual(f.expected)
  })

  test(`${f.name}: every capture keeps its four counts as whole, non-negative numbers`, () => {
    for (const s of f.samples) {
      for (const n of [s.reused, s.written, s.uncached, s.output]) {
        expect(Number.isInteger(n) && n >= 0).toBe(true)
      }
    }
  })
}

test('the capture\u2019s one collapse is measured against what was cached, not the window', () => {
  const v = replay(prefixLoss)
  expect(v.collapses).toHaveLength(1)
  // Request 4 cached 50,024 + 1,524; request 5 was served 27,221. The 2
  // uncached tokens beside request 4 were never cached, so they are not lost.
  const prev = prefixLoss.samples[3]
  expect(prev.uncached).toBeGreaterThan(0)
  expect(v.collapses[0].lost).toBe(24_327)
  expect(v.collapses[0].cause).toBe('unattributed')
})
