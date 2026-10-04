import { test, expect } from 'claude-code/testing'

import {
  type Sample,
  LOST_SHARE,
  MIN_PRIOR_WINDOW,
  SHOW_COLLAPSES,
  allCollapses,
  attribute,
  causeText,
  bar,
  compactionSpend,
  engineRemoved,
  sizesAgree,
  spendText,
  summarizerReprocessed,
  summarizerText,
  effortLabel,
  fresh,
  frontier,
  gapMs,
  gapText,
  isCollapse,
  isDescribed,
  isMainThread,
  lostGround,
  lostShare,
  pct,
  rebuilt,
  reprocessed,
  retain,
  reuseRate,
  shape,
  shapeText,
  sessionLines,
  statusText,
  stopLabel,
  tallyText,
  tok,
  toolUses,
  totals,
  view,
  windowTokens,
} from './cost'

const sample = (over: Partial<Sample> = {}): Sample => ({
  turnId: 't1',
  index: 0,
  model: 'claude-opus-5',
  reused: 0,
  written: 0,
  uncached: 0,
  output: 0,
  at: 1_000,
  ...over,
})

test('the window is the input side only, output excluded', () => {
  const s = sample({ reused: 100, written: 20, uncached: 5, output: 9_999 })
  expect(windowTokens(s)).toBe(125)
})

test('frontier is the cached prefix and reprocessed is everything else', () => {
  const s = sample({ reused: 8_000, written: 1_500, uncached: 500 })
  expect(frontier(s)).toBe(8_000)
  expect(reprocessed(s)).toBe(2_000)
})

test('reuse rate is a over T', () => {
  expect(reuseRate(sample({ reused: 750, written: 250 }))).toBe(0.75)
})

test('an empty window reads as zero reuse, never NaN', () => {
  expect(reuseRate(sample())).toBe(0)
  expect(Number.isNaN(reuseRate(sample()))).toBe(false)
})

// The finding, as a test: same number of tokens changed, different position,
// order-of-magnitude different cost.
test('position sets the cost, not the size of the edit', () => {
  const T = 20_000

  // Two 50-token edits in a window of T. Near the front the cache holds only
  // what preceded the edit; at the tail it holds everything up to the edit.
  const early = sample({ reused: 200, written: T - 200, uncached: 0 })
  const late = sample({ reused: T - 200, written: 200, uncached: 0 })

  expect(windowTokens(early)).toBe(windowTokens(late))
  expect(frontier(late)).toBeGreaterThan(frontier(early))
  expect(reprocessed(early)).toBe(19_800)
  expect(reprocessed(late)).toBe(200)
  expect(reprocessed(early) / reprocessed(late)).toBe(99)
})

test('totals sum both sides and report the session rate', () => {
  const t = totals([
    sample({ reused: 1_000, written: 100, uncached: 0 }),
    sample({ reused: 3_000, written: 0, uncached: 900 }),
  ])
  expect(t.requests).toBe(2)
  expect(t.reused).toBe(4_000)
  expect(t.reprocessed).toBe(1_000)
  expect(t.rate).toBe(0.8)
})

test('totals of nothing are zero, not NaN', () => {
  const t = totals([])
  expect(t.rate).toBe(0)
  expect(t.rebuiltShare).toBe(0)
  expect(t.rebuilt).toBe(0)
  expect(t.fresh).toBe(0)
  expect(t.requests).toBe(0)
})

test('a subagent step is not a main-thread sample', () => {
  expect(isMainThread(sample())).toBe(true)
  expect(isMainThread(sample({ agentId: 'a1' }))).toBe(false)
})

test('lost ground is what the previous window held and this request did not get', () => {
  const prev = sample({ reused: 30_000, written: 500 })
  const now = sample({ reused: 900, written: 24_000 })

  expect(lostGround(now, prev)).toBe(29_600)
  expect(lostShare(now, prev)).toBe(29_600 / 30_500)
})

test('a request that only grew the context lost no ground', () => {
  const prev = sample({ reused: 11_000, written: 1_000 })
  const now = sample({ reused: 12_000, written: 400 })

  expect(lostGround(now, prev)).toBe(0)
  expect(isCollapse(now, prev)).toBe(false)
})

test('tokens that were never cached are not counted as lost', () => {
  // prev: the cache held 15k (10k served + 5k written). The other 20k was
  // uncached — by the engine's definition neither read from nor written to
  // the cache, so it was never there to lose.
  const prev = sample({ reused: 10_000, written: 5_000, uncached: 20_000, at: 1_000 })
  // now: the cache served all 15k it held. Nothing was lost.
  const now = sample({ reused: 15_000, written: 1_000, uncached: 0, at: 2_000 })

  expect(lostGround(now, prev)).toBe(0)
  expect(isCollapse(now, prev)).toBe(false)
})

test('lost share is of what the cache held, not of the whole window', () => {
  // The cache held 15k; 20k more went uncached. Half the cached prefix is lost.
  const prev = sample({ reused: 10_000, written: 5_000, uncached: 20_000, at: 1_000 })
  const now = sample({ reused: 7_500, written: 9_000, uncached: 0, at: 2_000 })

  expect(lostGround(now, prev)).toBe(7_500)
  expect(lostShare(now, prev)).toBe(0.5)
})

test('the floor counts what was cached, so a mostly-uncached window is not enough', () => {
  // A 14k window, but only 4k of it was ever cached: below the floor.
  const prev = sample({ reused: 3_000, written: 1_000, uncached: 10_000, at: 1_000 })
  const now = sample({ reused: 0, written: 4_000, uncached: 0, at: 2_000 })

  expect(windowTokens(prev)).toBeGreaterThan(MIN_PRIOR_WINDOW)
  expect(lostShare(now, prev)).toBe(1)
  expect(isCollapse(now, prev)).toBe(false)
})

// The regression this metric exists for. A big tool result appended to a small
// fully-cached context reuses a minority of its window and loses nothing; the
// reuse rate alone cannot tell it apart from a rewritten prefix.
test('a large append is not a collapse, however low the reuse rate', () => {
  const prev = sample({ reused: 11_000, written: 1_000 })
  const now = sample({ reused: 12_000, written: 20_000 })

  expect(reuseRate(now)).toBeLessThan(0.4)
  expect(lostGround(now, prev)).toBe(0)
  expect(isCollapse(now, prev)).toBe(false)
})

test('a rewritten prefix is a collapse even at a healthy-looking reuse rate', () => {
  // Captured live: the request after a /compact held 27.3k of a 54.3k context.
  const prev = sample({ reused: 53_300, written: 971, at: 1_000 })
  const now = sample({ reused: 27_300, written: 12_300, at: 2_000 })

  expect(reuseRate(now)).toBeGreaterThan(0.6)
  expect(lostGround(now, prev)).toBe(26_971)
  expect(isCollapse(now, prev)).toBe(true)
})

test('a collapse needs a previous request and a prior context worth losing', () => {
  const prev = sample({ reused: 30_000, at: 1_000 })
  const bad = sample({ reused: 1_000, written: 29_000, at: 2_000 })

  expect(isCollapse(bad, prev)).toBe(true)
  expect(isCollapse(bad, undefined)).toBe(false)
  expect(isCollapse({ ...bad, agentId: 'a1' }, prev)).toBe(false)
})

// The thresholds, pinned from both sides. Every case above loses either nothing
// or nearly everything, so on their own they hold for any threshold in between.
// These sit a hair either side of the literal values, so moving one fails.

test('losing a quarter of the cache or less is not a collapse, more is', () => {
  const prev = sample({ reused: 40_000, written: 0, at: 1_000 })
  const lose = (lost: number): Sample => sample({ reused: 40_000 - lost, written: lost, at: 2_000 })

  // 24% of the cache lost: under the line.
  expect(lostShare(lose(9_600), prev)).toBe(0.24)
  expect(isCollapse(lose(9_600), prev)).toBe(false)
  // Exactly 25%: the line itself is not over it.
  expect(isCollapse(lose(10_000), prev)).toBe(false)
  // 26% lost: over it.
  expect(lostShare(lose(10_400), prev)).toBe(0.26)
  expect(isCollapse(lose(10_400), prev)).toBe(true)
})

test('a prior cache of 5,000 tokens is the least that can be called collapsed', () => {
  const lostAll = (cachedBefore: number): [Sample, Sample] => [
    sample({ reused: cachedBefore, at: 1_000 }),
    sample({ reused: 0, written: cachedBefore, at: 2_000 }),
  ]

  const [smallPrev, smallNow] = lostAll(4_999)
  expect(lostShare(smallNow, smallPrev)).toBe(1)
  expect(isCollapse(smallNow, smallPrev)).toBe(false)

  const [bigPrev, bigNow] = lostAll(5_000)
  expect(isCollapse(bigNow, bigPrev)).toBe(true)
})

test('a prior context under the floor is never called a collapse', () => {
  const tiny = sample({ reused: MIN_PRIOR_WINDOW - 1, at: 1_000 })
  const now = sample({ reused: 0, written: 400, at: 2_000 })

  expect(lostShare(now, tiny)).toBeGreaterThan(LOST_SHARE)
  expect(isCollapse(now, tiny)).toBe(false)
})

test('a compaction between the two requests is a known cause', () => {
  const prev = sample({ at: 1_000 })
  const now = sample({ at: 3_000 })
  expect(attribute(now, prev, [2_000])).toBe('compaction')
})

test('a compaction outside the interval is not this request’s cause', () => {
  const prev = sample({ at: 1_000 })
  const now = sample({ at: 3_000 })
  expect(attribute(now, prev, [500])).toBe('unattributed')
  expect(attribute(now, prev, [4_000])).toBe('unattributed')
})

test('an edit and a lapsed cache entry are not distinguishable, and say so', () => {
  const prev = sample({ at: 1_000 })
  const now = sample({ at: 2_000 })
  expect(attribute(now, prev, [])).toBe('unattributed')
})

test('the bar is exactly the width asked for', () => {
  expect(bar(0.5, 10)).toHaveLength(10)
  expect(bar(0, 10)).toHaveLength(10)
  expect(bar(1, 10)).toHaveLength(10)
})

test('the bar clamps instead of overrunning its box', () => {
  expect(bar(5, 4)).toHaveLength(4)
  expect(bar(-5, 4)).toHaveLength(4)
  expect(bar(Number.NaN, 4)).toHaveLength(4)
  expect(bar(0.5, 0)).toHaveLength(1)
})

test('token counts shorten only once they are long', () => {
  expect(tok(42)).toBe('42')
  expect(tok(9_999)).toBe('9999')
  expect(tok(12_480)).toBe('12.5k')
  expect(tok(2_400_000)).toBe('2.40M')
})

test('percentages are whole and clamped', () => {
  expect(pct(0.923)).toBe(92)
  expect(pct(2)).toBe(100)
  expect(pct(Number.NaN)).toBe(0)
})

// --- view: every decision the pane makes, made here ---

test('an unmeasured session has nothing to show', () => {
  const v = view([], [])
  expect(v.isEmpty).toBe(true)
  expect(v.last).toBe(null)
  expect(v.forks).toBe(null)
  expect(v.collapses).toHaveLength(0)
})

test('a session of only subagent steps has no main-thread figure', () => {
  const v = view([sample({ agentId: 'a1', written: 9_000 })], [])
  expect(v.isEmpty).toBe(true)
  expect(v.last).toBe(null)
  expect(v.forks?.requests).toBe(1)
  expect(v.forks?.reprocessed).toBe(9_000)
})

test('the session row counts main-thread requests only', () => {
  const v = view(
    [
      sample({ index: 0, reused: 0, written: 20_000, at: 1_000 }),
      sample({ index: 1, reused: 20_000, written: 1_000, at: 2_000 }),
      sample({ index: 2, reused: 1_000, written: 25_000, at: 3_000 }),
      sample({ index: 3, agentId: 'a1', reused: 0, written: 8_000, at: 4_000 }),
    ],
    [],
  )

  expect(v.isEmpty).toBe(false)
  expect(v.session.requests).toBe(3)
  expect(v.session.reused).toBe(21_000)
  expect(v.session.reprocessed).toBe(46_000)
  // The fork is reported apart, never folded in.
  expect(v.forks?.requests).toBe(1)
  expect(v.forks?.reprocessed).toBe(8_000)
})

test('the last request is the last MAIN-THREAD request, not the last of all', () => {
  const v = view(
    [
      sample({ index: 0, reused: 9_000, written: 1_000, at: 1_000 }),
      sample({ index: 1, agentId: 'a1', reused: 0, written: 8_000, at: 2_000 }),
    ],
    [],
  )
  expect(v.last?.held).toBe(9_000)
  expect(v.last?.reprocessed).toBe(1_000)
})

test('a collapse with no known cause says so rather than guessing', () => {
  const v = view(
    [
      sample({ index: 0, reused: 30_000, written: 500, at: 1_000 }),
      sample({ index: 1, reused: 900, written: 24_000, at: 3_000 }),
    ],
    [],
  )
  expect(v.collapses).toHaveLength(1)
  expect(v.collapses[0].cause).toBe('unattributed')
  expect(v.collapses[0].reprocessed).toBe(24_000)
  expect(v.collapses[0].lost).toBe(29_600)
})

test('a compaction between the two requests is named', () => {
  const v = view(
    [
      sample({ index: 0, reused: 30_000, written: 500, at: 1_000 }),
      sample({ index: 1, reused: 900, written: 24_000, at: 3_000 }),
    ],
    [2_000],
  )
  expect(v.collapses[0].cause).toBe('compaction')
})

test('a healthy session flags nothing', () => {
  const v = view(
    [
      sample({ index: 0, reused: 0, written: 20_000, at: 1_000 }),
      sample({ index: 1, reused: 20_000, written: 400, at: 2_000 }),
      sample({ index: 2, reused: 20_400, written: 300, at: 3_000 }),
    ],
    [],
  )
  expect(v.collapses).toHaveLength(0)
})

test('only the most recent collapses are shown, newest first', () => {
  // Each request holds the same 1k prefix while the context grows, so every
  // one loses more ground than the one before it.
  const many: Sample[] = []
  for (let i = 0; i < 10; i += 1) {
    many.push(sample({ index: i, reused: 1_000, written: 30_000 + 1_000 * i, at: 1_000 * (i + 1) }))
  }
  const v = view(many, [])

  expect(v.collapses).toHaveLength(SHOW_COLLAPSES)
  expect(v.collapses[0].lost).toBeGreaterThan(v.collapses[1].lost)
  expect(v.collapses[1].lost).toBeGreaterThan(v.collapses[2].lost)
})

test('a second compaction does not relabel the first one', () => {
  const v = view(
    [
      sample({ index: 0, reused: 30_000, written: 500, at: 1_000 }),
      // compaction at 2_000
      sample({ index: 1, reused: 900, written: 24_000, at: 3_000 }),
      sample({ index: 2, reused: 24_000, written: 500, at: 4_000 }),
      // compaction at 5_000
      sample({ index: 3, reused: 800, written: 20_000, at: 6_000 }),
    ],
    [2_000, 5_000],
  )

  expect(v.collapses).toHaveLength(2)
  // Both are compactions. With only the latest timestamp kept, the older one
  // silently became 'unattributed' on every render after the second compaction.
  expect(v.collapses[0].cause).toBe('compaction')
  expect(v.collapses[1].cause).toBe('compaction')
})

// --- the request's shape, and the records written before it ---
//
// Every test here turns on one distinction: a record that does not carry a
// field is not a record whose field is zero. The oldest records in the store
// predate all four fields, and `sample()` above builds exactly that shape.

test('an old record carries no shape, and says so rather than zero', () => {
  const old = sample({ reused: 10_000 })

  expect(isDescribed(old)).toBe(false)
  expect(toolUses(old)).toBe(null)
  expect(stopLabel(old)).toBe(null)
  expect(effortLabel(old)).toBe(null)
})

test('a step that called no tools is not a step that never said', () => {
  expect(toolUses(sample({ toolUseCount: 0 }))).toBe(0)
  expect(toolUses(sample())).toBe(null)
})

test('a null stop reason is a recorded answer, not a missing field', () => {
  expect(stopLabel(sample({ stopReason: null }))).toBe('none')
  expect(stopLabel(sample({ stopReason: 'tool_use' }))).toBe('tool_use')
  expect(stopLabel(sample())).toBe(null)
})

test('effort is tallied by its label, a number as readily as a name', () => {
  expect(effortLabel(sample({ effort: 'high' }))).toBe('high')
  expect(effortLabel(sample({ effort: 4_096 }))).toBe('4096')
  expect(effortLabel(sample())).toBe(null)
})

test('a junk message count is not a measurement', () => {
  expect(isDescribed(sample({ messageCount: Number.NaN }))).toBe(false)
  expect(toolUses(sample({ toolUseCount: Number.NaN }))).toBe(null)
  expect(isDescribed(sample({ messageCount: 0 }))).toBe(true)
})

test('the shape of a mixed run reports its own coverage', () => {
  const s = shape([
    sample({ messageCount: 10, stopReason: 'tool_use', toolUseCount: 3, effort: 'high' }),
    sample({ messageCount: 30, stopReason: 'end_turn', toolUseCount: 0, effort: 'high' }),
    // Written before the fields existed: counted as a request, described by none.
    sample({ reused: 1_000 }),
  ])

  expect(s.requests).toBe(3)
  expect(s.described).toBe(2)
  expect(s.meanMessages).toBe(20)
  expect(s.peakMessages).toBe(30)
  expect(s.withToolUses).toBe(2)
  expect(s.toolUses).toBe(3)
  expect(s.stops).toEqual({ tool_use: 1, end_turn: 1 })
  expect(s.efforts).toEqual({ high: 2 })
})

test('a run of only old records has a shape of zeroes, never NaN', () => {
  const s = shape([sample({ reused: 10_000 }), sample({ reused: 20_000 })])

  expect(s.requests).toBe(2)
  expect(s.described).toBe(0)
  expect(s.meanMessages).toBe(0)
  expect(Number.isNaN(s.meanMessages)).toBe(false)
  expect(s.peakMessages).toBe(0)
  expect(s.withToolUses).toBe(0)
  expect(s.toolUses).toBe(0)
  expect(s.stops).toEqual({})
  expect(s.efforts).toEqual({})
})

test('the shape of nothing is zeroes, never NaN', () => {
  const s = shape([])
  expect(s.described).toBe(0)
  expect(s.meanMessages).toBe(0)
  expect(Number.isNaN(s.meanMessages)).toBe(false)
})

test('a tally reads commonest first, ties by name', () => {
  expect(tallyText({ end_turn: 2, tool_use: 9 })).toBe('tool_use 9 · end_turn 2')
  expect(tallyText({ refusal: 1, end_turn: 1 })).toBe('end_turn 1 · refusal 1')
  expect(tallyText({})).toBe('')
})

// --- every collapse, not only the few the pane shows ---

test('allCollapses counts past what the pane has room for', () => {
  const many: Sample[] = []
  for (let i = 0; i < 10; i += 1) {
    many.push(sample({ index: i, reused: 1_000, written: 30_000 + 1_000 * i, at: 1_000 * (i + 1) }))
  }

  expect(allCollapses(many, [])).toHaveLength(9)
  expect(view(many, []).collapses).toHaveLength(SHOW_COLLAPSES)
})

test('allCollapses skips forks and keeps the oldest first', () => {
  const all = allCollapses(
    [
      sample({ index: 0, reused: 30_000, written: 500, at: 1_000 }),
      sample({ index: 1, reused: 900, written: 24_000, at: 2_000 }),
      sample({ index: 2, agentId: 'a1', reused: 0, written: 40_000, at: 3_000 }),
      sample({ index: 3, reused: 1_000, written: 40_000, at: 4_000 }),
    ],
    [],
  )

  expect(all).toHaveLength(2)
  expect(all[0].lost).toBe(29_600)
  expect(all[1].lost).toBe(23_900)
})

// --- the shape on the pane’s own line ---

test('the shape line names only what the record carries', () => {
  expect(shapeText({ messageCount: 12, toolUses: 2 })).toBe('12 messages · 2 tool calls')
  expect(shapeText({ messageCount: 12, toolUses: null })).toBe('12 messages')
  expect(shapeText({ messageCount: null, toolUses: 0 })).toBe('0 tool calls')
  expect(shapeText({ messageCount: null, toolUses: null })).toBe(null)
})

test('the view carries the last request\u2019s shape beside its cost', () => {
  const v = view(
    [
      sample({ index: 0, reused: 9_000, written: 1_000, at: 1_000, messageCount: 8 }),
      sample({
        index: 1,
        reused: 10_000,
        written: 500,
        at: 2_000,
        messageCount: 11,
        toolUseCount: 2,
      }),
    ],
    [],
  )

  expect(v.last?.messageCount).toBe(11)
  expect(v.last?.toolUses).toBe(2)
})

test('the view of an old record reports no shape rather than a zero', () => {
  const v = view([sample({ reused: 9_000, written: 1_000 })], [])

  expect(v.last?.messageCount).toBe(null)
  expect(v.last?.toolUses).toBe(null)
  expect(v.last?.held).toBe(9_000)
})

// --- rebuilt and fresh: what the reprocessed tokens were ---

test('the first request rebuilt nothing: all of it is fresh', () => {
  const first = sample({ reused: 0, written: 18_400 })
  expect(rebuilt(first, undefined)).toBe(0)
  expect(fresh(first, undefined)).toBe(18_400)
})

test('appended content is fresh, a lost prefix is rebuilt', () => {
  const prev = sample({ reused: 30_000, written: 500 })
  const appended = sample({ reused: 30_500, written: 2_000 })
  expect(rebuilt(appended, prev)).toBe(0)
  expect(fresh(appended, prev)).toBe(2_000)

  // 30.5k was held, 900 served, 24k written: 29.6k lost, only 24k re-paid.
  const lost = sample({ reused: 900, written: 24_000 })
  expect(rebuilt(lost, prev)).toBe(24_000)
  expect(fresh(lost, prev)).toBe(0)

  // Lost 10k of the prefix and added 3k new on top.
  const mixed = sample({ reused: 20_500, written: 13_000 })
  expect(rebuilt(mixed, prev)).toBe(10_000)
  expect(fresh(mixed, prev)).toBe(3_000)
})

test('a shrinking context loses more than it re-pays, so rebuilt is capped', () => {
  const prev = sample({ reused: 100_000 })
  const shrunk = sample({ reused: 0, written: 50_000 })
  // The ground lost (100k) exceeds what was reprocessed (50k).
  expect(lostGround(shrunk, prev)).toBe(100_000)
  expect(rebuilt(shrunk, prev)).toBeLessThanOrEqual(reprocessed(shrunk))
  expect(rebuilt(shrunk, prev)).toBe(50_000)
  expect(rebuilt(shrunk, prev) + fresh(shrunk, prev)).toBe(reprocessed(shrunk))
  expect(fresh(shrunk, prev)).toBe(0)
})

test('rebuilt and fresh always partition reprocessed', () => {
  const prev = sample({ reused: 40_000, written: 2_000, uncached: 100 })
  for (const reused of [0, 1_000, 41_000, 42_100, 60_000]) {
    for (const written of [0, 500, 20_000]) {
      const s = sample({ reused, written, uncached: 7 })
      expect(rebuilt(s, prev) + fresh(s, prev)).toBe(reprocessed(s))
      expect(fresh(s, prev)).toBeGreaterThanOrEqual(0)
    }
  }
})

test('totals read each sample against the one before it', () => {
  const t = totals([
    sample({ reused: 0, written: 20_000, at: 1_000 }),
    sample({ reused: 20_000, written: 1_000, at: 2_000 }),
    sample({ reused: 1_000, written: 25_000, at: 3_000 }),
  ])
  expect(t.reprocessed).toBe(46_000)
  // Request 3 held 21,000, was served 1,000: 20,000 lost, all of it re-paid.
  // The 20,000 first write and the 1,000 append are new; so is request 3's 5,000.
  expect(t.rebuilt).toBe(20_000)
  expect(t.fresh).toBe(26_000)
  expect(t.rebuilt + t.fresh).toBe(t.reprocessed)
  expect(t.rebuiltShare).toBe(20_000 / 46_000)
})

test('rebuiltShare does not drift with length where the cumulative rate does', () => {
  // Context grows 40k -> 100k by appending; nothing is ever lost.
  const grow = (n: number): Sample[] => {
    const out: Sample[] = []
    let window = 40_000
    for (let i = 0; i < n; i += 1) {
      const add = i === 0 ? 40_000 : 1_000
      out.push(sample({ index: i, reused: i === 0 ? 0 : window, written: add, at: 1_000 * (i + 1) }))
      window = (i === 0 ? 0 : window) + add
    }
    return out
  }

  for (const n of [10, 60]) {
    const t = totals(grow(n))
    expect(t.rebuilt).toBe(0)
    expect(t.rebuiltShare).toBe(0)
  }
  // The cumulative rate, by contrast, climbs on its own.
  expect(totals(grow(60)).rate).toBeGreaterThan(totals(grow(10)).rate)
})

// --- the aggregate labels ---

test('the session lines say what was reprocessed and how much of it was rebuilt', () => {
  const t = totals([
    sample({ reused: 0, written: 20_000, at: 1_000 }),
    sample({ reused: 20_000, written: 1_000, at: 2_000 }),
    sample({ reused: 1_000, written: 25_000, at: 3_000 }),
  ])
  const lines = sessionLines(t)

  expect(lines[0]).toBe('3 requests · 46.0k reprocessed')
  expect(lines[1]).toBe('43% of that was rebuilt — work the cache had held and lost')
  expect(lines[2]).toBe('26.0k was new content, never cached before')
})

test('the session lines carry no cumulative reuse rate and no sum called held', () => {
  // 60 requests growing 40k -> 100k, nothing ever lost.
  const grown: Sample[] = [sample({ reused: 0, written: 40_000, at: 1_000 })]
  for (let i = 1; i < 60; i += 1) {
    grown.push(sample({ index: i, reused: 40_000 + 1_000 * (i - 1), written: 1_000, at: 1_000 * (i + 1) }))
  }
  const text = sessionLines(totals(grown)).join('\n') + '\n' + statusText(totals(grown))

  // "held" as a per-request size is fine; a count of it summed is not.
  expect(text).not.toMatch(/\d held/)
  expect(text).not.toContain('reused')
  expect(text).not.toContain('re-sent')
  expect(sessionLines(totals(grown))[1]).toContain('0% of that was rebuilt')
})

test('a session that reprocessed nothing says so instead of 0% of nothing', () => {
  const lines = sessionLines(totals([]))
  expect(lines[1]).toBe('nothing reprocessed')
  expect(lines[2]).toBe('')
})

test('the status line leads with the share that compares sessions', () => {
  const t = totals([
    sample({ reused: 0, written: 20_000, at: 1_000 }),
    sample({ reused: 20_000, written: 1_000, at: 2_000 }),
    sample({ reused: 1_000, written: 25_000, at: 3_000 }),
  ])
  expect(statusText(t)).toBe('cache 43% rebuilt · 46.0k reprocessed')
  expect(statusText(totals([]))).toBe('cache 0% rebuilt · 0 reprocessed')
})

const main = (i: number): Sample => sample({ turnId: `m${i}`, index: i })
const fork = (i: number): Sample => sample({ turnId: `f${i}`, index: i, agentId: 'agent-1' })

test('a burst of fork samples leaves every main-thread sample intact', () => {
  const mains = Array.from({ length: 10 }, (_, i) => main(i))
  const forks = Array.from({ length: 500 }, (_, i) => fork(i))
  const kept = retain([...mains, ...forks], 10, 5)

  expect(kept.filter(isMainThread)).toEqual(mains)
  expect(kept.filter(s => !isMainThread(s))).toEqual(forks.slice(-5))
})

test('each kind keeps its own newest, in the original order', () => {
  const all = [main(0), fork(0), main(1), fork(1), main(2), fork(2)]

  expect(retain(all, 2, 1).map(s => s.turnId)).toEqual(['m1', 'm2', 'f2'])
})

test('under both caps nothing is dropped', () => {
  const all = [main(0), fork(0), main(1)]

  expect(retain(all, 5, 5)).toEqual(all)
  expect(retain([], 5, 5)).toEqual([])
})

test('a cap of zero drops that kind entirely and keeps the other', () => {
  const all = [main(0), fork(0), main(1)]

  expect(retain(all, 0, 5).map(s => s.turnId)).toEqual(['f0'])
  expect(retain(all, 5, 0).map(s => s.turnId)).toEqual(['m0', 'm1'])
})

// --- a model switch is provable from the two records ---

const SWITCHED = [
  sample({ index: 0, model: 'claude-opus-5-5', reused: 90_000, written: 6_000, at: 1_000 }),
  sample({ index: 1, model: 'claude-haiku-4-5', reused: 0, written: 96_000, at: 3_000 }),
]

test('a different model on the next request is named as the cause', () => {
  const [prev, now] = SWITCHED
  expect(attribute(now, prev, [])).toBe('model-change')
  expect(attribute(now, { ...prev, model: now.model }, [])).toBe('unattributed')
  // Same-length names still differ.
  expect(attribute({ ...now, model: 'claude-opus-5-6' }, { ...prev, model: 'claude-opus-5-5' }, [])).toBe(
    'model-change',
  )
})

test('a model change stays a collapse, and names both models', () => {
  const v = view(SWITCHED, [])
  expect(v.collapses).toHaveLength(1)
  expect(v.collapses[0].lost).toBe(96_000)
  expect(v.collapses[0].cause).toBe('model-change')
  expect(v.collapses[0].why).toBe('the model changed (claude-opus-5-5 → claude-haiku-4-5)')
})

test('a model change outranks a compaction in the same window, and says so', () => {
  const v = view(SWITCHED, [2_000])
  expect(v.collapses[0].cause).toBe('model-change')
  expect(v.collapses[0].notes).toContain('a compaction also ran in this window')
  // Without the compaction there is nothing to add.
  expect(view(SWITCHED, []).collapses[0].notes).not.toContain('a compaction also ran in this window')
})

test('a compaction with the same model on both sides is still a compaction', () => {
  const [prev, now] = SWITCHED
  const same = { ...now, model: prev.model }
  expect(attribute(same, prev, [2_000])).toBe('compaction')
  expect(causeText('compaction', same, prev)).toBe('compaction rewrote the prefix')
})

// --- a hold of exactly zero rules out a partial edit ---

const pair = (reused: number) => [
  sample({ index: 0, reused: 30_000, written: 500, at: 1_000 }),
  sample({ index: 1, reused, written: 30_500 - reused, at: 3_000 }),
]
const HELD_NOTE = 'nothing was held — not a partial edit'

test('a collapse that held nothing says it was not a partial edit, and names no cause', () => {
  const [one] = view(pair(0), []).collapses
  expect(one.notes).toContain(HELD_NOTE)
  expect(one.cause).toBe('unattributed')
  // It narrows; it must not claim which whole-prefix event it was.
  expect(one.notes.join(' ')).not.toMatch(/lapse|expired|cleared|\/clear|model/)
})

test('a collapse that still held something makes no such claim', () => {
  expect(view(pair(1), []).collapses[0].notes).not.toContain(HELD_NOTE)
  expect(view(pair(900), []).collapses[0].notes).not.toContain(HELD_NOTE)
})

test('a named cause is not narrowed by the zero-hold note', () => {
  const [prev, now] = pair(0)
  expect(view([prev, { ...now, model: 'other' }], []).collapses[0].notes).not.toContain(HELD_NOTE)
  expect(view(pair(0), [2_000]).collapses[0].notes).not.toContain(HELD_NOTE)
})

// --- the gap is an observation, measured start to start ---

const H = 3_600_000
const gapPair = (prevStart?: number, nowStart?: number) => [
  sample({ index: 0, reused: 30_000, written: 500, at: 1_000, startedAt: prevStart }),
  sample({ index: 1, reused: 900, written: 24_000, at: 9 * H, startedAt: nowStart }),
]

test('the gap runs from one start to the next, not from end to end', () => {
  // at - prev.at would be ~9h; the starts say 3.3h.
  const [prev, now] = gapPair(0, 3.3 * H)
  expect(gapMs(now, prev)).toBe(3.3 * H)
})

test('a gap that cannot be computed is unknown, never zero', () => {
  expect(gapMs(...(gapPair(0, undefined).reverse() as [Sample, Sample]))).toBeNull()
  expect(gapMs(...(gapPair(undefined, H).reverse() as [Sample, Sample]))).toBeNull()
  expect(gapMs(...(gapPair().reverse() as [Sample, Sample]))).toBeNull()
  const [prev, now] = gapPair(0, H)
  expect(gapMs({ ...now, startedAt: NaN }, prev)).toBeNull()
  expect(gapMs({ ...now, startedAt: '5' as unknown as number }, prev)).toBeNull()
  expect(gapMs({ ...now, startedAt: Infinity }, prev)).toBeNull()
  expect(gapMs(now, { ...prev, startedAt: -Infinity })).toBeNull()
  // Backwards clocks are not a negative interval.
  expect(gapMs(...(gapPair(H, 0).reverse() as [Sample, Sample]))).toBeNull()
  // Genuinely simultaneous starts are a known zero, which is a different fact.
  expect(gapMs(...(gapPair(5, 5).reverse() as [Sample, Sample]))).toBe(0)
})

test('durations read in the unit that fits', () => {
  expect(gapText(45_000)).toBe('45s')
  expect(gapText(59_999)).toBe('59s')
  expect(gapText(60_000)).toBe('1m')
  expect(gapText(12 * 60_000)).toBe('12m')
  expect(gapText(H - 1)).toBe('59m')
  expect(gapText(3.3 * H)).toBe('3.3h')
  expect(gapText(18.7 * H)).toBe('18.7h')
})

test('a collapse shows the gap beside it and attributes nothing to it', () => {
  const [one] = view(gapPair(0, 3.3 * H), []).collapses
  expect(one.notes).toContain('3.3h since the previous request')
  expect(one.cause).toBe('unattributed')
})

test('a short gap is shown too, since the reader decides what it means', () => {
  expect(view(gapPair(0, 12 * 60_000), []).collapses[0].notes).toContain('12m since the previous request')
  expect(view(gapPair(0, 5_000), []).collapses[0].notes).toContain('5s since the previous request')
})

test('a gap of any length is never a cause', () => {
  for (const hours of [0.01, 0.5, 2, 18.7, 400]) {
    const [one] = view(gapPair(0, hours * H), []).collapses
    expect(one.cause).toBe('unattributed')
    expect(one.why).toBe('cause not attributable from token counts')
    // Past any lifetime, nothing is held-claimed beyond the one-count fact.
    expect(one.notes.join(' ')).not.toMatch(/lapse|expire|ttl|lifetime/i)
  }
})

test('a collapse from records without start times shows no gap, not 0s', () => {
  const [one] = view(gapPair(), []).collapses
  expect(one.notes.join(' ')).not.toContain('since the previous request')
  expect(one.notes.join(' ')).not.toContain('0s')
})

test('the gap is shown beside a named cause too', () => {
  const [prev, now] = gapPair(0, 3.3 * H)
  const [one] = view([prev, { ...now, model: 'other' }], []).collapses
  expect(one.cause).toBe('model-change')
  expect(one.notes).toContain('3.3h since the previous request')
})

// --- what a compaction's own request cost ---
//
// Four distinct counts, so no test can pass by reading one field as another.

const SUMMARY = {
  input_tokens: 3_000,
  output_tokens: 700,
  cache_read_input_tokens: 40_000,
  cache_creation_input_tokens: 9_000,
}

test('the summary request reprocessed its uncached and written input, not what it read', () => {
  expect(summarizerReprocessed(SUMMARY)).toBe(12_000)
})

test('the summary request’s sentence says it is outside the reprocessed total', () => {
  const text = summarizerText(SUMMARY)

  expect(text).toContain('12.0k reprocessed')
  expect(text).toContain('40.0k served from cache')
  expect(text).toContain('700 out')
  expect(text).toContain('not in the reprocessed total')
})

test('the spend sums only the compactions that reported usage', () => {
  const spend = compactionSpend([
    { at: 1, usage: SUMMARY },
    1_500,
    { at: 2, trigger: 'auto' },
    { at: 3, usage: { ...SUMMARY, input_tokens: 1, output_tokens: 50, cache_creation_input_tokens: 2 } },
  ])

  expect(spend).toEqual({ compactions: 4, withUsage: 2, reprocessed: 12_000 + 3, output: 750 })
})

test('no compaction is no spend, and unknown usage is not a zero spend', () => {
  expect(spendText(compactionSpend([]))).toBe(null)
  expect(spendText(compactionSpend([1_000, { at: 2 }]))).toBe('2 compactions · summary request usage not recorded')
  expect(spendText(compactionSpend([{ at: 1, usage: SUMMARY }]))).toBe(
    '1 compaction · 12.0k reprocessed by the summary requests (usage known for 1), not counted above',
  )
})

const compactedPair = (usage?: typeof SUMMARY) =>
  view(
    [
      sample({ index: 0, reused: 30_000, written: 500, at: 1_000 }),
      sample({ index: 1, reused: 900, written: 24_000, at: 3_000 }),
    ],
    [{ at: 2_000, trigger: 'auto', ...(usage === undefined ? {} : { usage }) }],
  )

test('a collapse row carries the summary request that caused it', () => {
  expect(compactedPair(SUMMARY).collapses[0].notes).toContain(summarizerText(SUMMARY))
  expect(compactedPair().collapses[0].notes.some(n => n.includes('summary request'))).toBe(false)
})

test('the summary request is beside the session figures, never inside them', () => {
  const withIt = compactedPair(SUMMARY)
  const without = compactedPair()

  expect(withIt.compaction.reprocessed).toBe(12_000)
  expect(withIt.session).toEqual(without.session)
  expect(without.compaction).toEqual({ compactions: 1, withUsage: 0, reprocessed: 0, output: 0 })
})

test('a precompute’s summary request is a cost even though it caused nothing', () => {
  const v = view(
    [
      sample({ index: 0, reused: 30_000, written: 500, at: 1_000 }),
      sample({ index: 1, reused: 900, written: 24_000, at: 3_000 }),
    ],
    [{ at: 2_000, trigger: 'precompute', usage: SUMMARY }],
  )

  expect(v.collapses[0].cause).toBe('unattributed')
  expect(v.collapses[0].notes.some(n => n.includes('summary request'))).toBe(false)
  expect(v.compaction.reprocessed).toBe(12_000)
})

test('a model change that took the cause leaves the summary request to the spend line', () => {
  const v = view(
    [
      sample({ index: 0, reused: 30_000, written: 500, at: 1_000 }),
      sample({ index: 1, model: 'other-model', reused: 900, written: 24_000, at: 3_000 }),
    ],
    [{ at: 2_000, trigger: 'auto', usage: SUMMARY }],
  )

  expect(v.collapses[0].cause).toBe('model-change')
  expect(v.collapses[0].notes.some(n => n.includes('summary request'))).toBe(false)
  expect(v.compaction.reprocessed).toBe(12_000)
})

// --- the oracle: the engine's sizes against the measured loss ---
//
// COMPACTED: the cache held 30.5k, the next request served 6.2k, so 24.3k lost.

const COMPACTED = [
  sample({ index: 0, reused: 30_000, written: 500, at: 1_000 }),
  sample({ index: 1, reused: 6_200, written: 24_000, at: 3_000 }),
]
const LOST = 24_300

const compactedWith = (c: Record<string, unknown>) =>
  view(COMPACTED, [{ at: 2_000, trigger: 'manual', ...c }]).collapses[0]

test('the fixture loses what the tests say it loses', () => {
  expect(lostGround(COMPACTED[1], COMPACTED[0])).toBe(LOST)
})

test('the engine’s shrink needs both sizes, and is their difference', () => {
  expect(engineRemoved({ at: 1, tokensBefore: 50_000, tokensAfter: 12_000 })).toBe(38_000)
  expect(engineRemoved({ at: 1, tokensBefore: 50_000 })).toBe(null)
  expect(engineRemoved({ at: 1, tokensAfter: 12_000 })).toBe(null)
  expect(engineRemoved({ at: 1 })).toBe(null)
  expect(engineRemoved({ at: 1, tokensBefore: 0, tokensAfter: 0 })).toBe(0)
})

test('sizes agree within the larger of the floor and a share of the larger size', () => {
  // floor binds below 13.3k: 2k apart agrees, 2.001k apart does not
  expect(sizesAgree(5_000, 7_000)).toBe(true)
  expect(sizesAgree(5_000, 7_001)).toBe(false)
  // share binds above it: 15% of 40k is 6k
  expect(sizesAgree(34_000, 40_000)).toBe(true)
  expect(sizesAgree(33_999, 40_000)).toBe(false)
  // symmetric
  expect(sizesAgree(40_000, 34_000)).toBe(true)
  expect(sizesAgree(40_000, 33_999)).toBe(false)
  // the share is of the larger: 15% of the smaller (30k) would be 4.5k, and 4.9k apart still agrees
  expect(sizesAgree(30_000, 34_900)).toBe(true)
})

test('a collapse the engine’s sizes match says so, with both numbers', () => {
  const one = compactedWith({ tokensBefore: 40_000, tokensAfter: 16_000 })

  expect(one.engine?.compared).toBe(true)
  expect(one.engine?.disagrees).toBe(false)
  expect(one.engine?.text).toBe(
    'engine reported 24.0k removed; measured 24.3k lost — agree within tolerance (engine sized the conversation 40.0k → 16.0k)',
  )
})

test('a collapse the engine’s sizes contradict says so and prefers neither', () => {
  const one = compactedWith({ tokensBefore: 40_000, tokensAfter: 30_000 })

  expect(one.engine?.disagrees).toBe(true)
  expect(one.engine?.text).toBe(
    'engine reported 10.0k removed; measured 24.3k lost — DISAGREE by 14.3k, not reconciled (engine sized the conversation 40.0k → 30.0k)',
  )
  // the measured loss is untouched by the engine's figure
  expect(one.lost).toBe(LOST)
})

test('an engine report of growth is shown as growth, not clamped to zero', () => {
  const one = compactedWith({ tokensBefore: 10_000, tokensAfter: 16_000 })

  expect(one.engine?.text).toContain('engine reported the conversation grew by 6000;')
  expect(one.engine?.disagrees).toBe(true)
})

test('an old bare-number compaction has nothing to check against, and still attributes', () => {
  const one = view(COMPACTED, [2_000]).collapses[0]

  expect(one.cause).toBe('compaction')
  expect(one.engine).toEqual({
    text: 'engine did not record both sizes — nothing to check against',
    compared: false,
    disagrees: false,
  })
})

test('one engine size alone is not enough to check', () => {
  expect(compactedWith({ tokensBefore: 40_000 }).engine?.compared).toBe(false)
  expect(compactedWith({ tokensAfter: 16_000 }).engine?.compared).toBe(false)
})

test('two compactions in one window are not paired with one loss', () => {
  const one = view(COMPACTED, [
    { at: 1_500, trigger: 'auto', tokensBefore: 40_000, tokensAfter: 16_000 },
    { at: 2_000, trigger: 'manual', tokensBefore: 40_000, tokensAfter: 16_000 },
  ]).collapses[0]

  expect(one.cause).toBe('compaction')
  expect(one.engine).toEqual({
    text: '2 compactions ran in this window; engine sizes not compared',
    compared: false,
    disagrees: false,
  })
})

test('a precompute in the window is not a second compaction', () => {
  const one = view(COMPACTED, [
    { at: 1_500, trigger: 'precompute', tokensBefore: 99_000, tokensAfter: 1_000 },
    { at: 2_000, trigger: 'manual', tokensBefore: 40_000, tokensAfter: 16_000 },
  ]).collapses[0]

  expect(one.engine?.compared).toBe(true)
  expect(one.engine?.disagrees).toBe(false)
})

test('a collapse no compaction caused carries no engine row', () => {
  const sw = [COMPACTED[0], { ...COMPACTED[1], model: 'other-model' }]

  expect(view(sw, [{ at: 2_000, trigger: 'auto', tokensBefore: 40_000, tokensAfter: 16_000 }]).collapses[0].engine).toBe(null)
  expect(view(COMPACTED, []).collapses[0].engine).toBe(null)
})
