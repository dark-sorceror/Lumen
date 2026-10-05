/**
 * The harness: a real session's samples, kept across sessions and replayable
 * as a regression test.
 *
 * The pane answers "what is happening now". This answers "did that change" —
 * a captured run carries what it produced when it was taken, so a later edit
 * to the cost logic that alters a real session's verdicts fails a test instead
 * of quietly redrawing.
 */
import type { TurnStepInput, TurnStepResult } from 'claude-code'

import type { Cause, CompactionSpend, Compaction, CompactionUsage, CompactTrigger, Sample, Shape, StoredCompaction } from './cost'
import { CAUSE_TEXT, allCollapses, compactionSpend, spendText, isMainThread, pct, shape, tallyText, tok, totals, view } from './cost'

/**
 * The record one measured request leaves: the request's shape beside its cost.
 *
 * `null` when the step carried no usage — the API reported nothing, so there
 * is nothing measured to record. The fields are taken from both halves of the
 * step: `messageCount` and `effort` from the request the engine is about to
 * send, `stopReason` and the tool calls from the response it got back.
 *
 * `startedAt` is when the step began, taken by the caller before the response
 * streamed; `at` is when it ended. Left off when not given.
 *
 * A field the engine did not give is left off the record rather than written
 * as a zero, so a reader can tell silence from a measurement.
 */
export const sampleOf = (
  e: Pick<TurnStepInput, 'agentId' | 'effort' | 'messageCount'>,
  r: Pick<TurnStepResult, 'turnId' | 'index' | 'stopReason' | 'toolUses' | 'usage'>,
  at: number,
  startedAt?: number,
): Sample | null => {
  const usage = r.usage
  if (usage === null) return null

  return {
    turnId: r.turnId,
    index: r.index,
    ...(e.agentId === undefined ? {} : { agentId: e.agentId }),
    model: usage.model,
    reused: usage.cache_read_input_tokens,
    written: usage.cache_creation_input_tokens,
    uncached: usage.input_tokens,
    output: usage.output_tokens,
    at,
    ...(startedAt === undefined ? {} : { startedAt }),
    messageCount: e.messageCount,
    stopReason: r.stopReason,
    toolUseCount: r.toolUses.length,
    ...(e.effort === undefined ? {} : { effort: e.effort }),
  }
}

export type Verdict = {
  requests: number
  reused: number
  reprocessed: number
  collapses: { lost: number; cause: Cause }[]
}

export type Fixture = {
  name: string
  capturedAt: string
  /** How it was captured and anything the numbers do not say for themselves. */
  note: string
  compactions: StoredCompaction[]
  samples: Sample[]
  /** What this capture produced when taken. A diff here is a regression. */
  expected: Verdict
}

/** What a capture produces under the current logic. */
export const replay = (f: Pick<Fixture, 'samples' | 'compactions'>): Verdict => {
  const v = view(f.samples, f.compactions)

  return {
    requests: v.session.requests,
    reused: v.session.reused,
    reprocessed: v.session.reprocessed,
    collapses: v.collapses.map(c => ({ lost: c.lost, cause: c.cause })),
  }
}

/** The prefix every session's key carries in the cross-session store. */
export const SESSION_PREFIX = 'session:'

/** Where a session's samples live in the plugin's cross-session store. */
export const storeKey = (sessionId: string): string => `${SESSION_PREFIX}${sessionId}`

/** Is this store key one of the recorded sessions, rather than anything else? */
export const isSessionKey = (key: string): boolean => key.startsWith(SESSION_PREFIX)

/** The session a key names; the key itself when there is no prefix to strip. */
export const sessionIdFromKey = (key: string): string =>
  isSessionKey(key) ? key.slice(SESSION_PREFIX.length) : key

/** The prefix every session's compaction times carry in the store. */
export const COMPACTIONS_PREFIX = 'compactions:'

/** Where a session's compaction times live, beside its samples under `storeKey`. */
export const compactionsKey = (sessionId: string): string => `${COMPACTIONS_PREFIX}${sessionId}`

/** Is this store key one session's compaction times? */
export const isCompactionsKey = (key: string): boolean => key.startsWith(COMPACTIONS_PREFIX)

/**
 * Which keys to delete so the store keeps the newest `keep` sessions.
 *
 * Sessions are dropped oldest first, in the order the store lists them. A
 * session's compaction times go with it, and compaction times whose session
 * has no samples kept are dropped too — they attribute nothing.
 */
export const staleKeys = (keys: readonly string[], keep: number): string[] => {
  const sessions = keys.filter(isSessionKey)
  const stale = sessions.slice(0, Math.max(0, sessions.length - keep))
  const kept = new Set(sessions.slice(stale.length).map(sessionIdFromKey))
  const orphans = keys.filter(
    k => isCompactionsKey(k) && !kept.has(k.slice(COMPACTIONS_PREFIX.length)),
  )

  return [...stale, ...orphans]
}

/**
 * Should this failed store write be announced? Once per session: the first
 * failure says so and records the session in `told`; later ones stay quiet,
 * so a full store costs one toast rather than one per request.
 */
export const shouldTell = (told: Set<string>, sessionId: string): boolean => {
  if (told.has(sessionId)) return false
  told.add(sessionId)

  return true
}

/**
 * Is this stored value one of this mod's records, with the counts the
 * arithmetic reads intact?
 *
 * The store holds JSON written by every earlier build of this mod, and it is
 * a file on disk besides. The four numbers checked here are the ones every
 * version has written and every function in cost.ts reads; `output` is not
 * among them because nothing computes from it. The enriched fields are not
 * checked at all — a record without them is old, not broken.
 */
export const isSample = (value: unknown): value is Sample => {
  if (typeof value !== 'object' || value === null) return false
  const r = value as Record<string, unknown>

  return [r.reused, r.written, r.uncached, r.at].every(
    n => typeof n === 'number' && Number.isFinite(n),
  )
}

/**
 * A stored session's records, read back. Anything that is not a record with
 * its counts intact is dropped rather than carried into the arithmetic, where
 * one missing count would turn a whole session's figures into NaN.
 */
export const readSamples = (value: unknown): Sample[] =>
  Array.isArray(value) ? value.filter(isSample) : []

const finite = (n: unknown): n is number => typeof n === 'number' && Number.isFinite(n)

const TRIGGERS: readonly CompactTrigger[] = ['manual', 'auto', 'plugin', 'precompute']

/**
 * The record one compaction leaves: when it ran, and whatever of the engine's
 * own figures it supplied.
 *
 * A figure the event did not give is left off rather than written as zero. A
 * `usage` is kept only whole: with one of its four counts missing or not a
 * number, the request is unknown, and a partial one would price it low.
 */
export const compactionRecord = (
  at: number,
  trigger: CompactTrigger,
  r: { tokensBefore?: number; tokensAfter?: number; usage?: Partial<CompactionUsage> | null },
): Compaction => {
  const u = r.usage

  return {
    at,
    trigger,
    ...(finite(r.tokensBefore) ? { tokensBefore: r.tokensBefore } : {}),
    ...(finite(r.tokensAfter) ? { tokensAfter: r.tokensAfter } : {}),
    ...(usageOf(u) === null ? {} : { usage: usageOf(u) as CompactionUsage }),
  }
}

const usageOf = (u: unknown): CompactionUsage | null => {
  if (typeof u !== 'object' || u === null) return null
  const r = u as Record<string, unknown>
  if (
    !finite(r.input_tokens) ||
    !finite(r.output_tokens) ||
    !finite(r.cache_read_input_tokens) ||
    !finite(r.cache_creation_input_tokens)
  ) {
    return null
  }

  return {
    input_tokens: r.input_tokens,
    output_tokens: r.output_tokens,
    cache_read_input_tokens: r.cache_read_input_tokens,
    cache_creation_input_tokens: r.cache_creation_input_tokens,
  }
}

/**
 * One stored compaction, read back: a bare timestamp stays a bare timestamp
 * (it is an old record, not a broken one), an object keeps only the fields
 * that are intact, and anything without a time is dropped.
 */
export const readCompaction = (value: unknown): StoredCompaction | null => {
  if (finite(value)) return value
  if (typeof value !== 'object' || value === null) return null
  const r = value as Record<string, unknown>
  if (!finite(r.at)) return null

  return {
    at: r.at,
    ...(TRIGGERS.includes(r.trigger as CompactTrigger) ? { trigger: r.trigger as CompactTrigger } : {}),
    ...(finite(r.tokensBefore) ? { tokensBefore: r.tokensBefore } : {}),
    ...(finite(r.tokensAfter) ? { tokensAfter: r.tokensAfter } : {}),
    ...(usageOf(r.usage) === null ? {} : { usage: usageOf(r.usage) as CompactionUsage }),
  }
}

/**
 * A stored session's compactions, read back; null when none were ever
 * recorded for it. The two are different facts: no list means the session
 * predates the record, an empty list means it was recorded that none ran.
 */
export const readCompactions = (value: unknown): StoredCompaction[] | null =>
  Array.isArray(value)
    ? value.map(readCompaction).filter((c): c is StoredCompaction => c !== null)
    : null

/** One session as the review lists it. Main-thread requests only, as the pane's are. */
export type SessionRow = {
  sessionId: string
  requests: number
  /** Tokens the cache served, summed over requests: a billing figure, not a size. */
  served: number
  reprocessed: number
  /** Reprocessed tokens that had been cached and were lost. */
  rebuilt: number
  /** `rebuilt / reprocessed`; 0 for a session that reprocessed nothing. */
  rebuiltShare: number
  collapses: number
  /** How many of them each cause accounts for; only `model-change` and `compaction` are ever proven. */
  causes: Record<Cause, number>
  /** What its compactions' own requests cost, beyond `reprocessed`; zeros when none were recorded. */
  compaction: CompactionSpend
  /** Compaction collapses whose engine sizes could be compared with the measured loss, and how many disagreed. */
  engine: { checked: number; disagreed: number }
  /** Were this session's compaction times recorded? False for any session from before they were. */
  compactionsRecorded: boolean
  /** The latest record's timestamp, 0 when none; what the rows sort on. */
  lastSeen: number
}

/** A key and the value under it, as `$.store` hands them over. */
export type StoredSession = {
  key: string
  value: unknown
}

export const sessionRow = (
  sessionId: string,
  samples: readonly Sample[],
  compactions: readonly StoredCompaction[] | null = null,
): SessionRow => {
  const main = samples.filter(isMainThread)
  const t = totals(main)
  // The pane's own attribution, over the same samples: the two views cannot
  // disagree about an event. With no record, nothing can be proven.
  const flagged = allCollapses(samples, compactions ?? [])
  const causes: Record<Cause, number> = { 'model-change': 0, compaction: 0, unattributed: 0 }
  for (const c of flagged) causes[c.cause] += 1

  const engine = { checked: 0, disagreed: 0 }
  for (const c of flagged) {
    if (c.engine === null || !c.engine.compared) continue
    engine.checked += 1
    if (c.engine.disagrees) engine.disagreed += 1
  }

  let lastSeen = 0
  for (const s of samples) if (s.at > lastSeen) lastSeen = s.at

  return {
    sessionId,
    requests: t.requests,
    served: t.reused,
    reprocessed: t.reprocessed,
    rebuilt: t.rebuilt,
    rebuiltShare: t.rebuiltShare,
    collapses: flagged.length,
    causes,
    compaction: compactionSpend(compactions ?? []),
    engine,
    compactionsRecorded: compactions !== null,
    lastSeen,
  }
}

/** Every recorded session's figures, newest first; other keys are not sessions. */
export const sessionRows = (entries: readonly StoredSession[]): SessionRow[] =>
  entries
    .filter(entry => isSessionKey(entry.key))
    .map(entry => {
      const id = sessionIdFromKey(entry.key)
      const times = entries.find(other => other.key === compactionsKey(id))

      return sessionRow(
        id,
        readSamples(entry.value),
        times === undefined ? null : readCompactions(times.value),
      )
    })
    .sort((a, b) => b.lastSeen - a.lastSeen)

/** Every recorded session summed. Rates are recomputed, never averaged. */
export type AllSessions = {
  sessions: number
  requests: number
  served: number
  reprocessed: number
  rebuilt: number
  rebuiltShare: number
  collapses: number
}

export const acrossSessions = (rows: readonly SessionRow[]): AllSessions => {
  let requests = 0
  let served = 0
  let reprocessed = 0
  let rebuilt = 0
  let collapses = 0

  for (const row of rows) {
    requests += row.requests
    served += row.served
    reprocessed += row.reprocessed
    rebuilt += row.rebuilt
    collapses += row.collapses
  }

  return {
    sessions: rows.length,
    requests,
    served,
    reprocessed,
    rebuilt,
    // The session shares are over different numbers of tokens, so averaging
    // them would weight a short session like a long one.
    rebuiltShare: reprocessed === 0 ? 0 : rebuilt / reprocessed,
    collapses,
  }
}

/** The recorded request shape across every session, main thread only. */
export const sessionsShape = (entries: readonly StoredSession[]): Shape =>
  shape(
    entries
      .filter(entry => isSessionKey(entry.key))
      .flatMap(entry => readSamples(entry.value).filter(isMainThread)),
  )

/**
 * A row's collapses, with the causes only where they are proven. A session
 * with no compaction record says so rather than implying its collapses were
 * not compactions; one with a record names what the record accounts for.
 */
export const collapseText = (row: SessionRow): string => {
  const count = plural(row.collapses, 'collapse', 'collapses')
  if (row.collapses === 0) return count
  if (!row.compactionsRecorded) {
    // A model change is read off the records themselves, so it is named even
    // where no compaction times were kept.
    const known = row.causes['model-change']
    return known === 0
      ? `${count} (no compaction times recorded)`
      : `${count} (${known} model-change, no compaction times recorded)`
  }

  const named = (['model-change', 'compaction', 'unattributed'] as const)
    .filter(cause => row.causes[cause] > 0)
    .map(cause => `${row.causes[cause]} ${cause}`)

  return `${count} (${named.join(', ')})`
}

/**
 * What the compactions' own requests cost, summed over the rows, said apart
 * from the totals above it: `turn.step` never sees a summary request, so none
 * of it is in those figures. Nothing when no compaction was recorded.
 */
export const spendLines = (rows: readonly SessionRow[]): string[] => {
  const all = compactionSpend([])
  for (const row of rows) {
    all.compactions += row.compaction.compactions
    all.withUsage += row.compaction.withUsage
    all.reprocessed += row.compaction.reprocessed
    all.output += row.compaction.output
  }
  const text = spendText(all)

  return text === null ? [] : ['', `Compactions: ${text}`]
}

/**
 * Where the engine's own sizes and the measured loss disagreed, said once the
 * session rows are summed. Silent when nothing could be checked or all agreed:
 * agreement needs no report, disagreement is the thing to see. The rows
 * themselves are in the pane.
 */
export const engineLines = (rows: readonly SessionRow[]): string[] => {
  let checked = 0
  let disagreed = 0
  for (const row of rows) {
    checked += row.engine.checked
    disagreed += row.engine.disagreed
  }

  return disagreed === 0
    ? []
    : [
        '',
        `Engine sizes disagreed with the measured loss on ${disagreed} of ${plural(checked, 'compaction collapse', 'compaction collapses')} that could be checked.`,
      ]
}

/** What the report says about causes, given the rows it lists. */
export const causeNote = (rows: readonly SessionRow[]): string[] => {
  const blind = rows.filter(
    row => !row.compactionsRecorded && row.collapses > row.causes['model-change'],
  ).length
  const unproven = rows.some(row => row.compactionsRecorded && row.causes.unattributed > 0)
  const note: string[] = []

  if (blind > 0) {
    note.push(
      `${plural(blind, 'session', 'sessions')} with collapses kept no compaction times, so no cause is named here:`,
      'a changed prefix and a cache entry that lapsed are the same four numbers.',
    )
  }
  if (unproven) {
    note.push(
      `Where compaction times were recorded, unattributed means none fell in that collapse's window (${CAUSE_TEXT.unattributed}).`,
    )
  }
  if (note.length === 0) {
    note.push(
      `Causes named only where proven: ${CAUSE_TEXT.compaction}; ${CAUSE_TEXT['model-change']} (the two records name different models).`,
    )
  }

  return note
}

const plural = (n: number, one: string, many: string): string => `${n} ${n === 1 ? one : many}`

/**
 * The whole cross-session review as plain text: a line per session, the
 * sessions summed, and what the records say about the requests themselves.
 *
 * `current` marks the running session's own row where it is already recorded.
 */
export const sessionsReport = (
  entries: readonly StoredSession[],
  current?: string,
): string => {
  const rows = sessionRows(entries)
  if (rows.length === 0) return 'No session recorded yet — nothing to review.'

  const all = acrossSessions(rows)
  const s = sessionsShape(entries)
  const width = Math.max(...rows.map(row => row.sessionId.length))

  const lines = [
    `${plural(all.sessions, 'recorded session', 'recorded sessions')}, main-thread requests only.`,
    '',
    ...rows.map(row =>
      [
        `  ${row.sessionId.padEnd(width)}`,
        `${String(row.requests).padStart(4)} req`,
        `${tok(row.reprocessed).padStart(7)} reprocessed`,
        `${String(pct(row.rebuiltShare)).padStart(3)}% rebuilt`,
        collapseText(row),
        row.sessionId === current ? '(this session)' : '',
      ]
        .join('  ')
        .trimEnd(),
    ),
    '',
    `All sessions: ${plural(all.requests, 'request', 'requests')} · ${tok(all.reprocessed)} reprocessed · ${pct(all.rebuiltShare)}% rebuilt · ${plural(all.collapses, 'collapse', 'collapses')}`,
    '',
    'Rebuilt: reprocessed tokens the cache had held a request earlier and did not serve. The rest was new content.',
    '',
    `Request shape, recorded for ${s.described} of ${plural(s.requests, 'request', 'requests')}:`,
    `  messages carried: ${Math.round(s.meanMessages)} mean · ${s.peakMessages} peak`,
    `  tool calls asked for: ${s.toolUses} over ${plural(s.withToolUses, 'request', 'requests')}`,
    `  stops: ${tallyText(s.stops) || 'none recorded'}`,
    `  effort: ${tallyText(s.efforts) || 'none recorded'}`,
    ...spendLines(rows),
    ...engineLines(rows),
    '',
    ...causeNote(rows),
  ]

  return lines.join('\n')
}

/**
 * The fixture module an export writes. `expected` is filled from the CURRENT
 * logic, so a capture records today's behaviour — review it before trusting
 * it as a baseline, exactly as any golden file.
 */
export const fixtureSource = (f: Fixture): string =>
  [
    "import type { Fixture } from '../record'",
    '',
    `/** ${f.note} */`,
    `export const fixture: Fixture = ${JSON.stringify(f, null, 2)}`,
    '',
  ].join('\n')
