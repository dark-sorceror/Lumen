import { test, expect } from 'claude-code/testing'

import type { Sample } from './cost'
import { pct, view } from './cost'
import {
  type StoredSession,
  acrossSessions,
  causeNote,
  collapseText,
  compactionRecord,
  compactionsKey,
  isCompactionsKey,
  readCompaction,
  readCompactions,
  staleKeys,
  isSample,
  isSessionKey,
  readSamples,
  sampleOf,
  sessionIdFromKey,
  sessionRow,
  sessionRows,
  sessionsReport,
  sessionsShape,
  shouldTell,
  storeKey,
} from './record'

/** A record of the shape the FIRST version of this mod wrote: no shape fields. */
const old = (over: Partial<Sample> = {}): Sample => ({
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

const usage = (over: Partial<Parameters<typeof sampleOf>[1]> = {}) => ({
  turnId: 'turn-1',
  index: 2,
  stopReason: 'tool_use' as const,
  toolUses: [{ name: 'Read', input: {} }],
  usage: {
    model: 'claude-opus-5',
    input_tokens: 120,
    output_tokens: 400,
    cache_read_input_tokens: 48_000,
    cache_creation_input_tokens: 1_200,
  },
  ...over,
})

// --- what one measured request records ---

test('a record carries the request’s shape and its cost together', () => {
  const s = sampleOf({ messageCount: 34, effort: 'high' }, usage(), 7_000)

  expect(s).toEqual({
    turnId: 'turn-1',
    index: 2,
    model: 'claude-opus-5',
    reused: 48_000,
    written: 1_200,
    uncached: 120,
    output: 400,
    at: 7_000,
    messageCount: 34,
    stopReason: 'tool_use',
    toolUseCount: 1,
    effort: 'high',
  })
})

test('a step the API reported nothing for records nothing', () => {
  expect(sampleOf({ messageCount: 3 }, usage({ usage: null }), 1)).toBe(null)
})

test('a field the engine did not give is left off, not written as zero', () => {
  // A model without an effort setting, and the main thread, which has no
  // agentId: neither absence may read back as a measurement.
  const s = sampleOf({ messageCount: 9 }, usage(), 2_000)

  expect('effort' in (s ?? {})).toBe(false)
  expect('agentId' in (s ?? {})).toBe(false)
  expect(s?.messageCount).toBe(9)
})

test('a subagent’s step records the loop it was made in', () => {
  const s = sampleOf({ messageCount: 4, agentId: 'a1' }, usage(), 2_000)
  expect(s?.agentId).toBe('a1')
})

test('a text-only step records zero tool calls, which is a measurement', () => {
  const s = sampleOf(
    { messageCount: 5 },
    usage({ stopReason: 'end_turn', toolUses: [] }),
    3_000,
  )

  expect(s?.toolUseCount).toBe(0)
  expect(s?.stopReason).toBe('end_turn')
})

// --- reading the store back ---

test('the key a session is stored under round-trips', () => {
  expect(storeKey('abc-123')).toBe('session:abc-123')
  expect(sessionIdFromKey(storeKey('abc-123'))).toBe('abc-123')
  expect(isSessionKey('session:abc-123')).toBe(true)
  expect(isSessionKey('something-else')).toBe(false)
})

test('a record missing a count the arithmetic reads is not a record', () => {
  expect(isSample(old())).toBe(true)
  // The enriched fields are not required: an old record is old, not broken.
  expect(isSample({ reused: 1, written: 2, uncached: 3, at: 4 })).toBe(true)
  expect(isSample({ reused: 1, written: 2, uncached: 3 })).toBe(false)
  expect(isSample({ written: 2, uncached: 3, at: 4 })).toBe(false)
  expect(isSample({ reused: '1', written: 2, uncached: 3, at: 4 })).toBe(false)
  expect(isSample({ reused: Number.NaN, written: 2, uncached: 3, at: 4 })).toBe(false)
  expect(isSample(null)).toBe(false)
  expect(isSample('session:1')).toBe(false)
})

test('a stored value that is not a list of records reads as none', () => {
  expect(readSamples(undefined)).toEqual([])
  expect(readSamples(null)).toEqual([])
  expect(readSamples({ samples: [] })).toEqual([])
  expect(readSamples([old(), null, 7, { reused: 1 }])).toEqual([old()])
})

// --- the cross-session review ---

test('a session recorded before the shape existed replays without NaN', () => {
  // Exactly what the store holds from an earlier build of this mod.
  const row = sessionRow('old-session', [
    old({ index: 0, reused: 0, written: 20_000, at: 1_000 }),
    old({ index: 1, reused: 20_000, written: 1_000, at: 2_000 }),
    old({ index: 2, reused: 1_000, written: 25_000, at: 3_000 }),
  ])

  expect(row.requests).toBe(3)
  expect(row.served).toBe(21_000)
  expect(row.reprocessed).toBe(46_000)
  expect(row.rebuilt).toBe(20_000)
  expect(row.rebuiltShare).toBe(20_000 / 46_000)
  expect(Number.isNaN(row.rebuiltShare)).toBe(false)
  expect(row.collapses).toBe(1)
  expect(row.lastSeen).toBe(3_000)
})

test('a session row counts main-thread requests and the forks apart', () => {
  const row = sessionRow('s1', [
    old({ index: 0, reused: 10_000, written: 500, at: 1_000 }),
    old({ index: 1, agentId: 'a1', reused: 0, written: 30_000, at: 2_000 }),
  ])

  expect(row.requests).toBe(1)
  expect(row.served).toBe(10_000)
  expect(row.reprocessed).toBe(500)
  // The fork's own 30k is not this session's reprocessing to answer for, but
  // its timestamp is still the latest thing the session recorded.
  expect(row.lastSeen).toBe(2_000)
})

test('a session with nothing measured is zeroes, never NaN', () => {
  const row = sessionRow('empty', [])

  expect(row.requests).toBe(0)
  expect(row.rebuiltShare).toBe(0)
  expect(row.collapses).toBe(0)
  expect(row.lastSeen).toBe(0)
})

test('a session’s collapses are counted whole, not capped at the pane’s few', () => {
  // The pane shows three; a review of the session has to say how many there
  // were. Each request here holds the same small prefix while the context
  // grows, so every one after the first loses ground.
  const samples: Sample[] = []
  for (let i = 0; i < 8; i += 1) {
    samples.push(
      old({ index: i, reused: 1_000, written: 30_000 + 1_000 * i, at: 1_000 * (i + 1) }),
    )
  }

  expect(sessionRow('busy', samples).collapses).toBe(7)
})

test('a fork’s step is no part of the shape the review reports', () => {
  const s = sessionsShape([
    {
      key: 'session:s1',
      value: [
        { ...old({ index: 0, at: 1_000 }), messageCount: 10, toolUseCount: 1 },
        { ...old({ index: 1, agentId: 'a1', at: 2_000 }), messageCount: 90, toolUseCount: 9 },
      ],
    },
  ])

  expect(s.requests).toBe(1)
  expect(s.described).toBe(1)
  expect(s.meanMessages).toBe(10)
  expect(s.peakMessages).toBe(10)
  expect(s.toolUses).toBe(1)
})

const ENTRIES: StoredSession[] = [
  {
    key: 'session:older',
    value: [
      old({ index: 0, reused: 0, written: 10_000, at: 1_000 }),
      old({ index: 1, reused: 10_000, written: 200, at: 2_000 }),
    ],
  },
  {
    key: 'session:newer',
    value: [
      {
        ...old({ index: 0, reused: 0, written: 30_000, at: 9_000 }),
        messageCount: 12,
        stopReason: 'tool_use' as const,
        toolUseCount: 2,
        effort: 'high' as const,
      },
      {
        ...old({ index: 1, reused: 1_000, written: 29_000, at: 10_000 }),
        messageCount: 20,
        stopReason: 'end_turn' as const,
        toolUseCount: 0,
        effort: 'high' as const,
      },
    ],
  },
  // Not a session: the store is the whole plugin's, not this review's.
  { key: 'something-else', value: { note: 'not a session' } },
]

test('every session key is read, and nothing else is', () => {
  const rows = sessionRows(ENTRIES)

  expect(rows).toHaveLength(2)
  expect(rows.map(r => r.sessionId)).toEqual(['newer', 'older'])
})

test('the rows run newest first, by the latest record in each', () => {
  const rows = sessionRows(ENTRIES)

  expect(rows[0].lastSeen).toBe(10_000)
  expect(rows[1].lastSeen).toBe(2_000)
})

test('each row reports what that session reprocessed and how much of it was rebuilt', () => {
  const [newer, older] = sessionRows(ENTRIES)

  expect(older.requests).toBe(2)
  expect(older.served).toBe(10_000)
  expect(older.reprocessed).toBe(10_200)
  expect(newer.served).toBe(1_000)
  expect(newer.reprocessed).toBe(59_000)
  expect(newer.collapses).toBe(1)
})

test('the sessions sum by token, so a long session is not a short one', () => {
  const [newer, older] = sessionRows(ENTRIES)
  const all = acrossSessions([newer, older])

  expect(all.sessions).toBe(2)
  expect(all.requests).toBe(4)
  expect(all.served).toBe(11_000)
  expect(all.reprocessed).toBe(69_200)
  expect(all.rebuilt).toBe(older.rebuilt + newer.rebuilt)
  expect(all.rebuiltShare).toBe(all.rebuilt / 69_200)
  expect(all.collapses).toBe(1)
  // Averaging the two session shares would have given something else entirely.
  expect(all.rebuiltShare).not.toBe((older.rebuiltShare + newer.rebuiltShare) / 2)
})

test('summing no sessions is zeroes, never NaN', () => {
  const all = acrossSessions([])

  expect(all.sessions).toBe(0)
  expect(all.rebuiltShare).toBe(0)
  expect(Number.isNaN(all.rebuiltShare)).toBe(false)
})

test('the shape across sessions counts old records as undescribed', () => {
  const s = sessionsShape(ENTRIES)

  expect(s.requests).toBe(4)
  expect(s.described).toBe(2)
  expect(s.meanMessages).toBe(16)
  expect(s.peakMessages).toBe(20)
  expect(s.toolUses).toBe(2)
  expect(s.stops).toEqual({ tool_use: 1, end_turn: 1 })
})

test('the report names every session and what it cost', () => {
  const text = sessionsReport(ENTRIES)
  const all = acrossSessions(sessionRows(ENTRIES))

  expect(text).toContain('2 recorded sessions')
  expect(text).toContain('newer')
  expect(text).toContain('older')
  expect(text).toContain('1 collapse')
  expect(text).toContain('4 requests')
  expect(text).toContain('69.2k reprocessed')
  expect(text).toContain(`${pct(all.rebuiltShare)}% rebuilt`)
  // The two defects: a sum of reads called "held", and a rate that climbs with length.
  expect(text).not.toMatch(/\d held/)
  expect(text).not.toContain('reused')
})

test('the report says how much of itself the shape rests on', () => {
  const text = sessionsReport(ENTRIES)

  expect(text).toContain('recorded for 2 of 4 requests')
  expect(text).toContain('16 mean')
  expect(text).toContain('20 peak')
  expect(text).toContain('stops: end_turn 1 · tool_use 1')
  expect(text).toContain('effort: high 2')
})

test('a report of only old records says so instead of inventing zeroes', () => {
  const text = sessionsReport([ENTRIES[0]])

  expect(text).toContain('1 recorded session')
  expect(text).toContain('recorded for 0 of 2 requests')
  expect(text).toContain('stops: none recorded')
  expect(text).toContain('effort: none recorded')
  expect(text).not.toContain('NaN')
})

test('the report refuses to name a cause it cannot prove', () => {
  const text = sessionsReport(ENTRIES)

  // The one collapse across these sessions is reported as a count and left
  // unattributed: the store keeps no compaction times to attribute it to.
  expect(text).toContain('1 collapse')
  expect(text).toContain('no cause is named here')
  expect(text).not.toContain('compaction rewrote')
  expect(text).not.toContain('unattributed')
})

test('the running session is marked where it is already recorded', () => {
  expect(sessionsReport(ENTRIES, 'newer')).toContain('(this session)')
  expect(sessionsReport(ENTRIES, 'not-recorded-yet')).not.toContain('(this session)')
  expect(sessionsReport(ENTRIES)).not.toContain('(this session)')
})

test('nothing recorded reads as nothing, not as an empty table', () => {
  expect(sessionsReport([])).toBe('No session recorded yet — nothing to review.')
  expect(sessionsReport([{ key: 'something-else', value: 1 }])).toBe(
    'No session recorded yet — nothing to review.',
  )
})

test('a junk record in the store does not poison the session it is in', () => {
  const rows = sessionRows([
    { key: 'session:s1', value: [old({ reused: 5_000, at: 1_000 }), { reused: 'lots' }, null] },
  ])

  expect(rows[0].requests).toBe(1)
  expect(rows[0].served).toBe(5_000)
  expect(Number.isNaN(rows[0].rebuiltShare)).toBe(false)
  expect(sessionsReport([{ key: 'session:s1', value: ['junk'] }])).toContain('0 req')
})

// --- compaction times kept in the store ---

/** Two requests with a collapse between them, at 2_000 and 3_000. */
const COLLAPSED: Sample[] = [
  old({ index: 0, reused: 20_000, written: 1_000, at: 2_000 }),
  old({ index: 1, reused: 1_000, written: 25_000, at: 3_000 }),
]

test('a session’s compaction times are keyed beside its samples, not on them', () => {
  expect(compactionsKey('abc')).toBe('compactions:abc')
  expect(compactionsKey('abc')).not.toBe(storeKey('abc'))
  expect(isCompactionsKey(compactionsKey('abc'))).toBe(true)
  expect(isCompactionsKey(storeKey('abc'))).toBe(false)
  expect(isSessionKey(compactionsKey('abc'))).toBe(false)
})

test('stored compaction times read back as numbers, or as nothing recorded', () => {
  expect(readCompactions([1_000, 2_000])).toEqual([1_000, 2_000])
  expect(readCompactions([])).toEqual([])
  expect(readCompactions([1_000, 'x', null, NaN, Infinity])).toEqual([1_000])
  expect(readCompactions(undefined)).toBeNull()
  expect(readCompactions({ at: 1 })).toBeNull()
})

test('a collapse between two stored samples is named when a compaction fell in it', () => {
  const row = sessionRow('s', COLLAPSED, [2_500])

  expect(row.collapses).toBe(1)
  expect(row.causes).toEqual({ 'model-change': 0, compaction: 1, unattributed: 0 })
  expect(row.compactionsRecorded).toBe(true)
  expect(collapseText(row)).toBe('1 collapse (1 compaction)')
})

test('the review and the pane agree on the same collapse', () => {
  const row = sessionRow('s', COLLAPSED, [2_500])
  const pane = view(COLLAPSED, [2_500])

  expect(pane.collapses.map(c => c.cause)).toEqual(['compaction'])
  expect(row.causes.compaction).toBe(1)
})

test('a compaction outside the collapse’s window is not its cause', () => {
  // At the previous sample's timestamp is outside (the window is open there);
  // at the collapsing sample's own timestamp is inside.
  expect(sessionRow('s', COLLAPSED, [2_000]).causes.compaction).toBe(0)
  expect(sessionRow('s', COLLAPSED, [3_000]).causes.compaction).toBe(1)
  expect(sessionRow('s', COLLAPSED, [3_001]).causes.compaction).toBe(0)
})

test('a recorded session with no fitting compaction says unattributed, not unrecorded', () => {
  const row = sessionRow('s', COLLAPSED, [])

  expect(row.compactionsRecorded).toBe(true)
  expect(row.causes).toEqual({ 'model-change': 0, compaction: 0, unattributed: 1 })
  expect(collapseText(row)).toBe('1 collapse (1 unattributed)')
})

test('a session with no compaction record says so instead of implying none ran', () => {
  const row = sessionRow('s', COLLAPSED)

  expect(row.compactionsRecorded).toBe(false)
  expect(row.causes.compaction).toBe(0)
  expect(collapseText(row)).toBe('1 collapse (no compaction times recorded)')
})

test('a row with no collapses carries no cause text either way', () => {
  expect(collapseText(sessionRow('s', [old()]))).toBe('0 collapses')
  expect(collapseText(sessionRow('s', [old()], []))).toBe('0 collapses')
})

test('the report reads each session’s compaction times from its own key', () => {
  const entries: StoredSession[] = [
    { key: storeKey('with'), value: COLLAPSED },
    { key: compactionsKey('with'), value: [2_500] },
    { key: storeKey('without'), value: COLLAPSED },
    { key: compactionsKey('other'), value: [2_500] },
  ]
  const rows = sessionRows(entries)
  const byId = Object.fromEntries(rows.map(r => [r.sessionId, r]))

  expect(rows).toHaveLength(2)
  expect(byId.with.causes.compaction).toBe(1)
  expect(byId.without.compactionsRecorded).toBe(false)
  expect(byId.without.causes.compaction).toBe(0)

  const text = sessionsReport(entries)
  expect(text).toContain('1 collapse (1 compaction)')
  expect(text).toContain('1 collapse (no compaction times recorded)')
  expect(text).toContain('no cause is named here')
})

test('the report’s cause note says only what the records support', () => {
  const none = sessionRow('a', COLLAPSED)
  const proven = sessionRow('b', COLLAPSED, [2_500])
  const unproven = sessionRow('c', COLLAPSED, [])
  const quiet = sessionRow('d', [old()])

  expect(causeNote([none]).join(' ')).toContain('no cause is named here')
  expect(causeNote([proven]).join(' ')).not.toContain('no cause is named here')
  expect(causeNote([proven]).join(' ')).not.toContain('none fell')
  expect(causeNote([unproven]).join(' ')).toContain('none fell')
  expect(causeNote([unproven]).join(' ')).not.toContain('no cause is named here')
  // A session without collapses has nothing to be blind about.
  expect(causeNote([quiet]).join(' ')).not.toContain('no cause is named here')
})

test('pruning drops a stale session together with its compaction times', () => {
  const keys = [
    storeKey('a'), storeKey('b'), storeKey('c'),
    compactionsKey('a'), compactionsKey('c'), 'something-else',
  ]

  expect(staleKeys(keys, 2).sort()).toEqual([compactionsKey('a'), storeKey('a')].sort())
  expect(staleKeys(keys, 3)).toEqual([])
})

test('pruning drops compaction times whose session recorded no samples', () => {
  const keys = [storeKey('a'), compactionsKey('a'), compactionsKey('ghost')]

  expect(staleKeys(keys, 5)).toEqual([compactionsKey('ghost')])
  expect(staleKeys([], 5)).toEqual([])
})

test('a session’s causes are counted apart when its collapses differ', () => {
  const samples = [
    ...COLLAPSED,
    old({ index: 2, reused: 26_000, written: 500, at: 4_000 }),
    old({ index: 3, reused: 1_000, written: 30_000, at: 5_000 }),
  ]
  const row = sessionRow('s', samples, [2_500])

  expect(row.collapses).toBe(2)
  expect(row.causes).toEqual({ 'model-change': 0, compaction: 1, unattributed: 1 })
  expect(collapseText(row)).toBe('2 collapses (1 compaction, 1 unattributed)')
})

test('a session row reads the rebuilt share into its line', () => {
  const text = sessionsReport([
    {
      key: 'session:s1',
      value: [
        old({ index: 0, reused: 0, written: 20_000, at: 1_000 }),
        old({ index: 1, reused: 20_000, written: 1_000, at: 2_000 }),
        old({ index: 2, reused: 1_000, written: 25_000, at: 3_000 }),
      ],
    },
  ])

  // 46,000 reprocessed, 20,000 of it rebuilt.
  expect(text).toMatch(/3 req\s+46\.0k reprocessed\s+43% rebuilt/)
  expect(text).toContain('All sessions: 3 requests · 46.0k reprocessed · 43% rebuilt')
})

test('a failed store write is announced once per session, not once per request', () => {
  const told = new Set<string>()
  expect(shouldTell(told, 'a')).toBe(true)
  expect(shouldTell(told, 'a')).toBe(false)
  expect(shouldTell(told, 'a')).toBe(false)
})

test('a new session after a /clear is told again', () => {
  const told = new Set<string>()
  expect(shouldTell(told, 'a')).toBe(true)
  expect(shouldTell(told, 'b')).toBe(true)
  expect(shouldTell(told, 'b')).toBe(false)
})

test('the review counts a model change as proven, with or without compaction times', () => {
  const switched: Sample[] = [
    { ...COLLAPSED[0], model: 'a' },
    { ...COLLAPSED[1], model: 'b' },
  ]
  const blind = sessionRow('s', switched)
  const recorded = sessionRow('s', switched, [])

  expect(blind.causes['model-change']).toBe(1)
  expect(collapseText(blind)).toBe('1 collapse (1 model-change, no compaction times recorded)')
  expect(collapseText(recorded)).toBe('1 collapse (1 model-change)')
  expect(causeNote([blind]).join(' ')).not.toContain('no cause is named here')
  expect(view(switched, []).collapses[0].cause).toBe('model-change')
})

test('a record keeps when the step began beside when it ended', () => {
  const s = sampleOf({ messageCount: 3 }, usage(), 9_000, 4_000)

  expect(s?.at).toBe(9_000)
  expect(s?.startedAt).toBe(4_000)
})

test('a record made without a start time carries no startedAt, not a zero', () => {
  const s = sampleOf({ messageCount: 3 }, usage(), 9_000)

  expect(s).not.toBeNull()
  expect(s !== null && 'startedAt' in s).toBe(false)
})

test('the review and the pane show the same gap for the same collapse', () => {
  const stamped: Sample[] = [
    { ...COLLAPSED[0], startedAt: 0 },
    { ...COLLAPSED[1], startedAt: 7_200_000 },
  ]
  expect(view(stamped, []).collapses[0].notes).toContain('2.0h since the previous request')
  expect(sessionRow('s', stamped, []).causes.unattributed).toBe(1)
})

// --- a compaction keeps what the engine said about it ---
//
// Every figure but the time is optional, and an absent one is unknown. Each
// field below gets its own distinct number so a test cannot pass by reading
// one field as another.

const SUMMARIZER = {
  input_tokens: 11,
  output_tokens: 22,
  cache_read_input_tokens: 33,
  cache_creation_input_tokens: 44,
}

test('a compaction record keeps each figure the event supplied', () => {
  expect(
    compactionRecord(5_000, 'auto', { tokensBefore: 90_000, tokensAfter: 12_000, usage: SUMMARIZER }),
  ).toEqual({ at: 5_000, trigger: 'auto', tokensBefore: 90_000, tokensAfter: 12_000, usage: SUMMARIZER })
})

test('a figure the event did not supply is absent from the record, not zero', () => {
  const r = compactionRecord(5_000, 'precompute', { tokensBefore: 90_000 })

  expect(r).toEqual({ at: 5_000, trigger: 'precompute', tokensBefore: 90_000 })
  expect('tokensAfter' in r).toBe(false)
  expect('usage' in r).toBe(false)
  expect('tokensBefore' in compactionRecord(1, 'manual', {})).toBe(false)
})

test('a figure the engine reported as zero is kept as zero', () => {
  const r = compactionRecord(5_000, 'manual', { tokensBefore: 0, tokensAfter: 0 })

  expect(r.tokensBefore).toBe(0)
  expect(r.tokensAfter).toBe(0)
})

test('a usage missing any one count is dropped whole rather than kept low', () => {
  for (const missing of Object.keys(SUMMARIZER)) {
    const partial = { ...SUMMARIZER, [missing]: undefined }
    expect('usage' in compactionRecord(1, 'auto', { usage: partial })).toBe(false)
  }
  expect('usage' in compactionRecord(1, 'auto', { usage: null })).toBe(false)
  expect('usage' in compactionRecord(1, 'auto', { tokensBefore: NaN, tokensAfter: Infinity })).toBe(false)
})

test('a compaction record survives the store’s JSON round trip', () => {
  const r = compactionRecord(5_000, 'manual', { tokensBefore: 90_000, tokensAfter: 12_000, usage: SUMMARIZER })

  expect(readCompaction(JSON.parse(JSON.stringify(r)))).toEqual(r)
})

test('an old bare-number compaction reads back as the bare number', () => {
  expect(readCompaction(2_500)).toBe(2_500)
  expect(readCompactions([2_500, compactionRecord(3_000, 'auto', { tokensBefore: 7 })])).toEqual([
    2_500,
    { at: 3_000, trigger: 'auto', tokensBefore: 7 },
  ])
})

test('a stored compaction keeps only the fields that are intact', () => {
  expect(readCompaction({ at: 1, trigger: 'sideways', tokensBefore: 'x', tokensAfter: 5, usage: { input_tokens: 1 } })).toEqual({
    at: 1,
    tokensAfter: 5,
  })
  expect(readCompaction({ tokensBefore: 5 })).toBeNull()
  expect(readCompaction({ at: NaN })).toBeNull()
  expect(readCompaction(null)).toBeNull()
  expect(readCompaction('1')).toBeNull()
})

test('a record attributes a collapse exactly as the bare time did', () => {
  const bare = view(COLLAPSED, [2_500])
  const rich = view(COLLAPSED, [compactionRecord(2_500, 'manual', { tokensBefore: 1, tokensAfter: 1 })])

  expect(bare.collapses[0].cause).toBe('compaction')
  expect(rich.collapses[0].cause).toBe('compaction')
  expect(sessionRow('s', COLLAPSED, [compactionRecord(3_001, 'auto', {})]).causes.compaction).toBe(0)
  expect(sessionRow('s', COLLAPSED, [compactionRecord(2_500, 'auto', {})]).causes.compaction).toBe(1)
})

test('a precompute is not a cause: it installs nothing', () => {
  const pre = compactionRecord(2_500, 'precompute', { tokensBefore: 90_000, usage: SUMMARIZER })

  expect(view(COLLAPSED, [pre]).collapses[0].cause).toBe('unattributed')
  expect(sessionRow('s', COLLAPSED, [pre]).causes.compaction).toBe(0)
  // the compaction that comes still counts
  expect(view(COLLAPSED, [pre, compactionRecord(2_600, 'auto', {})]).collapses[0].cause).toBe('compaction')
})

test('the review prints the summary requests apart from its totals, and nothing when none ran', () => {
  const rich = [compactionRecord(2_500, 'auto', { usage: SUMMARIZER })]
  const entries = (value: unknown[]): StoredSession[] => [
    { key: storeKey('s'), value: COLLAPSED },
    { key: compactionsKey('s'), value: value },
  ]

  expect(sessionRow('s', COLLAPSED, rich).compaction.reprocessed).toBe(11 + 44)
  expect(sessionsReport(entries(rich))).toContain(
    'Compactions: 1 compaction · 55 reprocessed by the summary requests (usage known for 1), not counted above',
  )
  expect(sessionsReport(entries([2_500]))).toContain('Compactions: 1 compaction · summary request usage not recorded')
  expect(sessionsReport(entries([]))).not.toContain('Compactions:')
  expect(sessionRow('s', COLLAPSED, rich).reprocessed).toBe(sessionRow('s', COLLAPSED, [2_500]).reprocessed)
})

test('the review sums the summary requests of every session, not just the first', () => {
  const entries: StoredSession[] = [
    { key: storeKey('a'), value: COLLAPSED },
    { key: compactionsKey('a'), value: [compactionRecord(2_500, 'auto', { usage: SUMMARIZER })] },
    { key: storeKey('b'), value: COLLAPSED },
    { key: compactionsKey('b'), value: [compactionRecord(2_500, 'auto', { usage: { ...SUMMARIZER, input_tokens: 1000 } })] },
  ]

  expect(sessionsReport(entries)).toContain('2 compactions · 1099 reprocessed by the summary requests (usage known for 2)')
})

test('the review says where the engine’s sizes disagreed, and is silent when they agreed', () => {
  // the cache held 21k and the next request served 1k: 20k lost
  const entries = (c: unknown): StoredSession[] => [
    { key: storeKey('s'), value: COLLAPSED },
    { key: compactionsKey('s'), value: [c] },
  ]
  const agree = compactionRecord(2_500, 'manual', { tokensBefore: 30_000, tokensAfter: 10_000 })
  const clash = compactionRecord(2_500, 'manual', { tokensBefore: 30_000, tokensAfter: 28_000 })

  expect(sessionRow('s', COLLAPSED, [agree]).engine).toEqual({ checked: 1, disagreed: 0 })
  expect(sessionRow('s', COLLAPSED, [clash]).engine).toEqual({ checked: 1, disagreed: 1 })
  expect(sessionRow('s', COLLAPSED, [2_500]).engine).toEqual({ checked: 0, disagreed: 0 })
  expect(sessionsReport(entries(clash))).toContain(
    'Engine sizes disagreed with the measured loss on 1 of 1 compaction collapse that could be checked.',
  )
  expect(sessionsReport(entries(agree))).not.toContain('Engine sizes disagreed')
  expect(sessionsReport(entries(2_500))).not.toContain('Engine sizes disagreed')
})

test('the review sums disagreements over sessions and counts only checkable collapses', () => {
  const clash = compactionRecord(2_500, 'manual', { tokensBefore: 30_000, tokensAfter: 28_000 })
  const agree = compactionRecord(2_500, 'manual', { tokensBefore: 30_000, tokensAfter: 10_000 })
  const entries: StoredSession[] = [
    { key: storeKey('a'), value: COLLAPSED },
    { key: compactionsKey('a'), value: [clash] },
    { key: storeKey('b'), value: COLLAPSED },
    { key: compactionsKey('b'), value: [agree] },
    { key: storeKey('c'), value: COLLAPSED },
    { key: compactionsKey('c'), value: [2_500] },
  ]

  expect(sessionsReport(entries)).toContain('disagreed with the measured loss on 1 of 2 compaction collapses')
})
