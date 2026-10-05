/**
 * The cost of rewriting context is set by WHERE the edit lands, not how big it is.
 *
 * Lumen's model: a context of T tokens edited at position `a` costs T - a to
 * re-prefill, because causal attention invalidates every cached key after the
 * edit. `a` is `first_invalid_token`; a one-word change forty turns back can
 * cost far more than rewriting a whole paragraph in the latest turn.
 *
 * Claude Code reports that same quantity per request, MEASURED, as the four
 * token counts of `ModelUsage`:
 *
 *   T     = input_tokens + cache_creation_input_tokens + cache_read_input_tokens
 *   a     = cache_read_input_tokens               (the prefix the cache held)
 *   T - a = input_tokens + cache_creation_input_tokens
 *
 * Nothing in this file is an estimate. What it cannot do is say *why* the
 * prefix was re-sent — see `attribute`.
 */

/**
 * Why the model stopped, in `TurnStepResult.stopReason`'s own spelling.
 * `TurnStopReason` is declared but not exported from 'claude-code', so the
 * union is mirrored here; `null` is the engine's "no response arrived".
 */
export type StopReason =
  | 'end_turn'
  | 'max_tokens'
  | 'stop_sequence'
  | 'tool_use'
  | 'pause_turn'
  | 'compaction'
  | 'refusal'
  | 'model_context_window_exceeded'
  | null

/** How hard the request asked the model to think (`TurnStepInput.effort`). */
export type Effort = 'low' | 'medium' | 'high' | 'xhigh' | 'max' | number

export type Sample = {
  turnId: string
  index: number
  /** A subagent's loop id; absent on the main thread. */
  agentId?: string
  model: string
  /** `cache_read_input_tokens` — the prefix served from cache. Lumen's `a`. */
  reused: number
  /** `cache_creation_input_tokens` — input this request wrote to the cache. */
  written: number
  /** `input_tokens` — input neither read from nor written to the cache. */
  uncached: number
  output: number
  at: number
  /**
   * When the step BEGAN, before the response streamed. `at` is taken after the
   * stream drains, so `at - prev.at` carries this step's duration; the interval
   * the cache cares about runs between two starts. Optional: records from
   * before it was kept have none, and `gapMs` reads that as unknown, not zero.
   */
  startedAt?: number
  /**
   * `TurnStepInput.messageCount` — how many messages the request carried.
   *
   * Optional, with the three below it: a record written before the harness
   * grew these fields carries none of them, and a record that does not say
   * is not a record that says zero. Everything that reads them says which.
   */
  messageCount?: number
  /** `TurnStepResult.stopReason`; `null` when no response arrived. */
  stopReason?: StopReason
  /** `TurnStepResult.toolUses.length` — tool calls the response asked for. */
  toolUseCount?: number
  /** `TurnStepInput.effort`; absent for a model without an effort setting. */
  effort?: Effort
}

export type Totals = {
  requests: number
  /**
   * `cache_read_input_tokens` summed over the requests: what the cache SERVED,
   * a billing quantity. Not a size — one prefix is counted once per request
   * that read it — so it is never "held".
   */
  reused: number
  /** Tokens put through the model: `rebuilt + fresh`. */
  reprocessed: number
  /** Reprocessed tokens that had been cached and were lost: work redone. */
  rebuilt: number
  /** Reprocessed tokens that were never cached: genuinely new content. */
  fresh: number
  /**
   * `rebuilt / reprocessed`; 0 when nothing was reprocessed. Unlike `rate`
   * it does not climb as a conversation grows, so it compares sessions.
   */
  rebuiltShare: number
  /**
   * `reused / (reused + reprocessed)` over the whole run. Kept for completeness,
   * not for display: it rises with conversation length on its own, so it can
   * neither compare two sessions nor show improvement.
   */
  rate: number
}

/**
 * Why a prefix was re-sent. `unattributed` is the honest default: a cache read
 * collapsing means EITHER the prefix changed OR the cache entry lapsed on TTL,
 * and the token counts cannot tell those apart.
 */
export type Cause = 'model-change' | 'compaction' | 'unattributed'

/** What triggered a compaction, in `SessionCompactTrigger`'s spelling. */
export type CompactTrigger = 'manual' | 'auto' | 'plugin' | 'precompute'

/** The summarizer's own request, in `ModelUsage`'s spelling. */
export type CompactionUsage = {
  input_tokens: number
  output_tokens: number
  cache_read_input_tokens: number
  cache_creation_input_tokens: number
}

/**
 * One compaction as the engine reported it.
 *
 * Only `at` is always known. Every other field is whatever that event actually
 * supplied: `tokensBefore` and `tokensAfter` are absent when core did not
 * record them (and `tokensAfter` always on `precompute`), `usage` when a hook
 * answered in core's place, a summary was reused, or the response reported
 * none. A field that is absent is unknown, never zero.
 */
export type Compaction = {
  at: number
  trigger?: CompactTrigger
  /** The conversation's size before, in tokens, as the engine counted it. */
  tokensBefore?: number
  /** Its size afterwards, in tokens. */
  tokensAfter?: number
  /** The summarizer's own request, which `turn.step` never sees. */
  usage?: CompactionUsage
}

/**
 * What the store holds: a bare timestamp from every build before the engine's
 * own figures were kept, a record from every build since. Both stay readable.
 */
export type StoredCompaction = number | Compaction

/** When a stored compaction ran. */
export const compactionAt = (c: StoredCompaction): number => (typeof c === 'number' ? c : c.at)

/** A stored compaction as a record; a bare timestamp carries nothing but its time. */
export const compactionOf = (c: StoredCompaction): Compaction => (typeof c === 'number' ? { at: c } : c)

/**
 * Did this event rewrite the transcript? A `precompute` computes a summary to
 * keep for the compaction that comes and installs nothing, so the prefix of the
 * next request is untouched by it: it cannot be the cause of a collapse. (The
 * summarizer's request still happened, so its `usage` is still a cost.)
 */
export const rewrites = (c: StoredCompaction): boolean => typeof c === 'number' || c.trigger !== 'precompute'

/** T — the whole input side of one request. Output is not part of the window. */
export const windowTokens = (s: Sample): number => s.reused + s.written + s.uncached

/** `a` — how far into the context the cache was still valid. */
export const frontier = (s: Sample): number => s.reused

/** T - a — what this request had to put through the model again. */
export const reprocessed = (s: Sample): number => s.written + s.uncached

/** `a / T`, in 0..1. A request with no input side reads as 0, not NaN. */
export const reuseRate = (s: Sample): number => {
  const t = windowTokens(s)
  return t === 0 ? 0 : s.reused / t
}

/** A fork pays for the whole prefix by construction, so it is never a signal. */
export const isMainThread = (s: Sample): boolean => s.agentId === undefined

/**
 * The samples worth keeping: the newest `keepMain` main-thread ones and the
 * newest `keepForks` fork ones, in their original order.
 *
 * The two are capped apart so a burst of subagent steps can never push
 * main-thread history off the end and shrink the session's reported totals.
 */
export const retain = (
  samples: readonly Sample[],
  keepMain: number,
  keepForks: number,
): Sample[] => {
  let main = samples.filter(isMainThread).length
  let forks = samples.length - main
  const kept: Sample[] = []

  for (const s of samples) {
    if (isMainThread(s)) {
      main -= 1
      if (main < keepMain) kept.push(s)
    } else {
      forks -= 1
      if (forks < keepForks) kept.push(s)
    }
  }

  return kept
}

/** Reprocessed tokens that had been cached and were lost: work redone. */
export const rebuilt = (s: Sample, prev: Sample | undefined): number =>
  prev === undefined ? 0 : Math.min(lostGround(s, prev), reprocessed(s))

/** Reprocessed tokens that were never cached: genuinely new content. */
export const fresh = (s: Sample, prev: Sample | undefined): number =>
  reprocessed(s) - rebuilt(s, prev)

/**
 * The samples MUST be in order, one thread: each is read against its
 * predecessor to tell rebuilt work from new content.
 */
export const totals = (list: readonly Sample[]): Totals => {
  let reused = 0
  let repro = 0
  let redone = 0
  for (let i = 0; i < list.length; i += 1) {
    const prev = i > 0 ? list[i - 1] : undefined
    reused += frontier(list[i])
    repro += reprocessed(list[i])
    redone += rebuilt(list[i], prev)
  }
  const t = reused + repro
  return {
    requests: list.length,
    reused,
    reprocessed: repro,
    rebuilt: redone,
    fresh: repro - redone,
    rebuiltShare: repro === 0 ? 0 : redone / repro,
    rate: t === 0 ? 0 : reused / t,
  }
}

// --- the request's shape: what it carried, beside what it cost ---
//
// Every reader here has to tell "not recorded" from "zero", because the oldest
// records in the store predate these fields. A record that does not say how
// many tools the step called is not a record of a step that called none.

/** Does this record carry the request shape, or does it predate it? */
export const isDescribed = (s: Sample): s is Sample & { messageCount: number } =>
  typeof s.messageCount === 'number' && Number.isFinite(s.messageCount)

/** Tool calls the response asked for; `null` when the record does not say. */
export const toolUses = (s: Sample): number | null =>
  typeof s.toolUseCount === 'number' && Number.isFinite(s.toolUseCount) ? s.toolUseCount : null

/** The stop reason as a tally key; `null` when the record does not say. */
export const stopLabel = (s: Sample): string | null =>
  s.stopReason === undefined ? null : s.stopReason === null ? 'none' : s.stopReason

/** The effort as a tally key; `null` when the record does not say. */
export const effortLabel = (s: Sample): string | null =>
  s.effort === undefined ? null : String(s.effort)

/** A tally of the labels seen, by label. Only labels actually seen appear. */
export type Tally = Record<string, number>

/**
 * What the recorded requests looked like. `described` says how many of them
 * carry the shape at all, so a reading always carries the size of its own
 * evidence rather than reading silence as zero.
 */
export type Shape = {
  requests: number
  /** Records carrying a message count — the shape's own coverage. */
  described: number
  /** Messages per described request. 0 when none is described, never NaN. */
  meanMessages: number
  /** The largest message count seen; 0 when none is recorded. */
  peakMessages: number
  /** Records that report a tool-use count. */
  withToolUses: number
  /** Tool calls over those records. */
  toolUses: number
  stops: Tally
  efforts: Tally
}

export const shape = (list: readonly Sample[]): Shape => {
  let described = 0
  let messages = 0
  let peakMessages = 0
  let withToolUses = 0
  let calls = 0
  const stops: Tally = {}
  const efforts: Tally = {}

  for (const s of list) {
    if (isDescribed(s)) {
      described += 1
      messages += s.messageCount
      peakMessages = Math.max(peakMessages, s.messageCount)
    }

    const used = toolUses(s)
    if (used !== null) {
      withToolUses += 1
      calls += used
    }

    const stop = stopLabel(s)
    if (stop !== null) stops[stop] = (stops[stop] ?? 0) + 1

    const effort = effortLabel(s)
    if (effort !== null) efforts[effort] = (efforts[effort] ?? 0) + 1
  }

  return {
    requests: list.length,
    described,
    meanMessages: described === 0 ? 0 : messages / described,
    peakMessages,
    withToolUses,
    toolUses: calls,
    stops,
    efforts,
  }
}

/** A tally as text, commonest first: `tool_use 14 · end_turn 5`. */
export const tallyText = (tally: Tally): string =>
  Object.entries(tally)
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .map(([label, n]) => `${label} ${n}`)
    .join(' · ')

/**
 * What the cache holds after a request: the prefix it served (and so still
 * holds) plus what the request wrote into it. NOT the window — `uncached` is,
 * by the engine's definition, neither read from nor written to the cache, so
 * it was never there to keep or to lose.
 */
export const cached = (s: Sample): number => s.reused + s.written

/**
 * Tokens the cache held after the previous request but did not serve for this
 * one: the ground the cache lost.
 *
 * This is the quantity the pane is about, and it is not the same as a low reuse
 * rate. A request that APPENDS 20k tokens to a fully cached 12k context reuses
 * only 37% of its window, but it lost nothing — the 20k was new and had to be
 * read whatever happened. A request whose prefix was rewritten serves less than
 * the previous request cached, and that shortfall is what the edit actually cost.
 *
 * The baseline is what was cached, not the previous window: input the engine
 * left uncached was never in the cache, so a request cannot lose it.
 */
export const lostGround = (s: Sample, prev: Sample): number =>
  Math.max(0, cached(prev) - frontier(s))

/**
 * Lost ground as a share of what the cache held a request earlier. The
 * denominator is the cache, not the window: the question is how much of the
 * cached prefix survived, and the loss can never exceed it, so the share stays
 * in 0..1 and does not shift with how much input happened to go uncached.
 */
export const lostShare = (s: Sample, prev: Sample): number => {
  const before = cached(prev)
  return before === 0 ? 0 : lostGround(s, prev) / before
}

/** A prior cache below this has too little in it to lose to read anything into. */
export const MIN_PRIOR_WINDOW = 5_000

/** Above this share of the prior context lost, the prefix was rewritten. */
export const LOST_SHARE = 0.25

/**
 * Did this request pay to re-send a prefix it had already cached?
 *
 * False for a subagent step (a fork always pays), false for the first request
 * (nothing was cached to lose), and false when the prior cache was too small
 * for its loss to mean anything.
 */
export const isCollapse = (s: Sample, prev: Sample | undefined): boolean => {
  if (!isMainThread(s)) return false
  if (prev === undefined) return false
  if (cached(prev) < MIN_PRIOR_WINDOW) return false

  return lostShare(s, prev) > LOST_SHARE
}

/**
 * Name the cause only where it is positively known. A compaction rewrites the
 * whole prefix and the engine says when one ran, so that one is certain;
 * everything else stays `unattributed`, because a changed prefix and a cache
 * entry that lapsed on TTL are the same four numbers.
 *
 * Every compaction's timestamp is kept, not just the latest. With one
 * timestamp, a second compaction silently relabelled the first one's collapse
 * as unattributed on the next render — the verdicts are recomputed each time,
 * so a stored history is the only thing that keeps an old label true.
 *
 * `prev` is required: isCollapse already refuses a request with nothing before
 * it, so a "first request" cause was unreachable from `view`.
 */
export const attribute = (s: Sample, prev: Sample, compactions: readonly StoredCompaction[]): Cause =>
  // Checked first: the two records name two different models, which needs no
  // outside clock. It stays a collapse; the whole prefix really was re-sent.
  s.model !== prev.model
    ? 'model-change'
    : compactedBetween(s, prev, compactions)
      ? 'compaction'
      : 'unattributed'

/** The compactions that rewrote the transcript between two requests; the window is open at `prev.at`. */
export const compactionsBetween = (
  s: Sample,
  prev: Sample,
  compactions: readonly StoredCompaction[],
): Compaction[] =>
  compactions
    .filter(c => rewrites(c) && compactionAt(c) > prev.at && compactionAt(c) <= s.at)
    .map(compactionOf)

const compactedBetween = (s: Sample, prev: Sample, compactions: readonly StoredCompaction[]): boolean =>
  compactionsBetween(s, prev, compactions).length > 0

/** How a cause reads, wherever a collapse is described: the pane and the review share it. */
export const CAUSE_TEXT: Record<Cause, string> = {
  'model-change': 'the model changed',
  compaction: 'compaction rewrote the prefix',
  unattributed: 'cause not attributable from token counts',
}

/** The cause for one collapse, with the records' own detail where it has some. */
export const causeText = (cause: Cause, s: Sample, prev: Sample): string =>
  cause === 'model-change' ? `${CAUSE_TEXT[cause]} (${prev.model} → ${s.model})` : CAUSE_TEXT[cause]

/**
 * The time from the previous request's start to this one's, in ms; `null` when
 * either record lacks `startedAt` or the two do not run forward. Never zero as
 * a stand-in: an unknown gap must not read as "no time passed".
 */
export const gapMs = (s: Sample, prev: Sample): number | null => {
  const a = prev.startedAt
  const b = s.startedAt
  if (typeof a !== 'number' || typeof b !== 'number') return null
  if (!Number.isFinite(a) || !Number.isFinite(b)) return null

  return b >= a ? b - a : null
}

/** A duration for a sentence: `45s`, `12m`, `3.3h`. */
export const gapText = (ms: number): string => {
  if (ms < 60_000) return `${Math.floor(ms / 1_000)}s`
  if (ms < 3_600_000) return `${Math.floor(ms / 60_000)}m`

  return `${(ms / 3_600_000).toFixed(1)}h`
}

/** Below this many tokens apart, two sizes of one event agree whatever their ratio. */
export const AGREE_FLOOR = 2_000

/** Above the floor, two sizes agree while they are within this share of the larger. */
export const AGREE_SHARE = 0.15

/** Do two token sizes of one event agree? Inclusive at the edge. */
export const sizesAgree = (a: number, b: number): boolean =>
  Math.abs(a - b) <= Math.max(AGREE_FLOOR, AGREE_SHARE * Math.max(Math.abs(a), Math.abs(b)))

/**
 * How much the engine says the compaction shrank the conversation:
 * `tokensBefore - tokensAfter`; null unless it recorded BOTH. Negative when it
 * reports the conversation grew. Never computed from one side alone.
 */
export const engineRemoved = (c: Compaction): number | null =>
  typeof c.tokensBefore === 'number' &&
  Number.isFinite(c.tokensBefore) &&
  typeof c.tokensAfter === 'number' &&
  Number.isFinite(c.tokensAfter)
    ? c.tokensBefore - c.tokensAfter
    : null

/**
 * THE ORACLE. Two measurements of one compaction, from different sides:
 *
 * - `removed` (engine): how much the conversation SHRANK, `tokensBefore -
 *   tokensAfter`.
 * - `lost` (this mod): how much cached prefix the NEXT request failed to
 *   serve, `lostGround`.
 *
 * They are not the same quantity and need not be equal. The relationship the
 * check ASSUMES is `lost ≈ removed`; the two sources of divergence it allows
 * for are the engine's sizes being its own counts rather than the API's (a few
 * percent), and a few hundred tokens of prefix wobble. It is NOT adjusted for
 * what the next request adds (a new user turn) or must write afresh (the
 * summary): both land in `written`, which `lostGround` never reads.
 *
 * ASSUMPTION NOT YET CHECKED AGAINST A LIVE COMPACTION: what survives a
 * compaction is only the unchanged head (system prompt, tools); every message
 * from the first replaced one on is invalid. If so, `lost` tracks the whole
 * old conversation (`tokensBefore`), not the shrink, and this check disagrees
 * on every compaction by about `tokensAfter`. That would be this check doing
 * its job: it is the first outside number the mod has had. The quantity
 * compared lives only in `engineRemoved`.
 *
 * The tolerance is `max(AGREE_FLOOR, AGREE_SHARE of the larger)`. 15%: the
 * engine's own sizes are estimates that run several percent off the API's, so
 * anything tighter cries wolf on a correct number; a missing term of the size
 * of a system prompt (~25k) or a summary (~10-20k) is well over 15% of a
 * 25-50k compaction, so anything looser hides the error it exists to catch.
 * The 2k floor keeps a small compaction from disagreeing over a rounding.
 *
 * Neither number is preferred and neither is adjusted to match the other.
 */
export type EngineCheck = { removed: number; lost: number; agrees: boolean }

export const checkAgainstEngine = (lost: number, c: Compaction): EngineCheck | null => {
  const removed = engineRemoved(c)

  return removed === null ? null : { removed, lost, agrees: sizesAgree(removed, lost) }
}

/** The check as a row: both numbers, the engine's own sizes, and whether they agree. */
export const engineText = (check: EngineCheck, c: Compaction): string => {
  const said =
    check.removed < 0
      ? `engine reported the conversation grew by ${tok(-check.removed)}`
      : `engine reported ${tok(check.removed)} removed`
  const sizes = `engine sized the conversation ${tok(c.tokensBefore ?? 0)} → ${tok(c.tokensAfter ?? 0)}`
  const verdict = check.agrees
    ? 'agree within tolerance'
    : `DISAGREE by ${tok(Math.abs(check.removed - check.lost))}, not reconciled`

  return `${said}; measured ${tok(check.lost)} lost — ${verdict} (${sizes})`
}

/** What a collapse's row says about the engine's own figures. */
/** `compared` is false where the note says why no comparison was made. */
export type EngineNote = { text: string; compared: boolean; disagrees: boolean }

/**
 * Only for a collapse a compaction caused, and only where ONE compaction ran in
 * the window: with two, one loss belongs to both and no pairing is honest.
 * Null for any other cause, where there is nothing to check.
 */
export const engineNote = (
  lost: number,
  cause: Cause,
  between: readonly Compaction[],
): EngineNote | null => {
  if (cause !== 'compaction') return null
  if (between.length > 1) {
    return { text: `${between.length} compactions ran in this window; engine sizes not compared`, compared: false, disagrees: false }
  }
  const check = checkAgainstEngine(lost, between[0])
  if (check === null) {
    return { text: 'engine did not record both sizes — nothing to check against', compared: false, disagrees: false }
  }

  return { text: engineText(check, between[0]), compared: true, disagrees: !check.agrees }
}

/**
 * What a compaction's own request put through the model: `input_tokens +
 * cache_creation_input_tokens`, the same T - a that `reprocessed` reads for a
 * step. Cache reads are not reprocessing.
 */
export const summarizerReprocessed = (u: CompactionUsage): number =>
  u.input_tokens + u.cache_creation_input_tokens

/**
 * The summarizer's request as a sentence. `turn.step` never fires for it, so it
 * is in no total the pane or the review prints; the sentence says so.
 */
export const summarizerText = (u: CompactionUsage): string =>
  `the summary request itself: ${tok(summarizerReprocessed(u))} reprocessed (${tok(u.cache_read_input_tokens)} served from cache), ${tok(u.output_tokens)} out — not in the reprocessed total`

/**
 * What compactions cost beyond the requests `turn.step` measured. `withUsage`
 * is the evidence behind `reprocessed`: a compaction whose usage was never
 * supplied (a hook answered, a summary was reused, an old bare timestamp) is
 * counted in `compactions` and contributes nothing it does not know.
 */
export type CompactionSpend = {
  compactions: number
  withUsage: number
  reprocessed: number
  output: number
}

export const compactionSpend = (list: readonly StoredCompaction[]): CompactionSpend => {
  let withUsage = 0
  let repro = 0
  let output = 0
  for (const c of list) {
    const u = compactionOf(c).usage
    if (u === undefined) continue
    withUsage += 1
    repro += summarizerReprocessed(u)
    output += u.output_tokens
  }

  return { compactions: list.length, withUsage, reprocessed: repro, output }
}

/**
 * The spend as one line, or null when no compaction was recorded. Said as
 * additional to the session's figures, never part of them.
 */
export const spendText = (c: CompactionSpend): string | null => {
  if (c.compactions === 0) return null
  const n = `${c.compactions} ${c.compactions === 1 ? 'compaction' : 'compactions'}`
  if (c.withUsage === 0) return `${n} · summary request usage not recorded`

  return `${n} · ${tok(c.reprocessed)} reprocessed by the summary requests (usage known for ${c.withUsage}), not counted above`
}

/**
 * What the records narrow without naming a cause: facts to read beside the
 * collapse. Nothing here says which whole-prefix event occurred.
 *
 * - A hold of exactly 0. An edit at position `a` leaves `a` tokens held, so a
 *   partial edit always holds something; zero leaves only whole-prefix events
 *   (the entry lapsed, the model changed, /clear, a new conversation). Said only
 *   for an unattributed collapse, where it narrows something; a named cause
 *   needs no narrowing. It does not say WHICH event.
 * - The gap since the previous request, only where both starts were recorded.
 *   Not a cause: the mod cannot read the cache lifetime, so a long gap is shown
 *   and never named.
 * - A compaction that also ran, when the model change took the cause.
 * - For a compaction, what its own summarizing request cost, where the engine
 *   reported it: a request `turn.step` never saw, so additional to the totals.
 */
export const observations = (
  s: Sample,
  prev: Sample,
  cause: Cause,
  compactions: readonly StoredCompaction[],
): string[] => {
  const notes: string[] = []
  if (cause === 'unattributed' && frontier(s) === 0) {
    notes.push('nothing was held — not a partial edit')
  }
  if (cause === 'model-change' && compactedBetween(s, prev, compactions)) {
    notes.push('a compaction also ran in this window')
  }
  if (cause === 'compaction') {
    for (const c of compactionsBetween(s, prev, compactions)) {
      if (c.usage !== undefined) notes.push(summarizerText(c.usage))
    }
  }
  const gap = gapMs(s, prev)
  if (gap !== null) notes.push(`${gapText(gap)} since the previous request`)

  return notes
}

/** A fixed-width meter. Rates outside 0..1 clamp rather than overrun the box. */
export const bar = (rate: number, width: number): string => {
  const w = Math.max(1, Math.floor(width))
  const safe = Number.isFinite(rate) ? Math.max(0, Math.min(1, rate)) : 0
  const filled = Math.max(0, Math.min(w, Math.round(safe * w)))
  return '█'.repeat(filled) + '░'.repeat(w - filled)
}

/** Token counts, short enough for a status line. */
export const tok = (n: number): string => {
  if (!Number.isFinite(n)) return '0'
  const v = Math.max(0, Math.round(n))
  if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(2)}M`
  if (v >= 10_000) return `${(v / 1_000).toFixed(1)}k`
  return String(v)
}

/** A whole percentage, for display only. */
export const pct = (rate: number): number =>
  Number.isFinite(rate) ? Math.round(Math.max(0, Math.min(1, rate)) * 100) : 0

/** The session's own three lines for the pane: how much, how much of it was rebuilt, how much new. */
export const sessionLines = (t: Totals): [string, string, string] => [
  `${t.requests} requests · ${tok(t.reprocessed)} reprocessed`,
  t.reprocessed === 0
    ? 'nothing reprocessed'
    : `${pct(t.rebuiltShare)}% of that was rebuilt — work the cache had held and lost`,
  t.reprocessed === 0 ? '' : `${tok(t.fresh)} was new content, never cached before`,
]

/** The status line: the number that compares sessions, then what it is a share of. */
export const statusText = (t: Totals): string =>
  `cache ${pct(t.rebuiltShare)}% rebuilt · ${tok(t.reprocessed)} reprocessed`

/** One flagged request: what it cost, and the most we can honestly say about why. */
export type Collapse = {
  /** Tokens the cache held a request ago and did not serve for this one. */
  lost: number
  /** `lost` as a share of what the cache held a request earlier. */
  share: number
  reprocessed: number
  cause: Cause
  /** The cause as a sentence, with the models named for a model change. */
  why: string
  /** What the records narrow beside it; see `observations`. */
  notes: string[]
  /** The engine's own sizes against the measured loss, for a compaction; see `engineNote`. */
  engine: EngineNote | null
}

/**
 * Everything the pane decides, decided here. The render hook is then a plain
 * mapping from this to elements, which keeps every judgement in a function a
 * test can call directly — the plugin test kit has no `state` noun, so a
 * mounted pane can only ever be read in the states the plugin starts in.
 */
export type View = {
  isEmpty: boolean
  last: {
    rate: number
    held: number
    reprocessed: number
    /** Ground lost against the request before it; 0 when there was none. */
    lost: number
    isCollapse: boolean
    /** Messages the request carried; null when the record predates the field. */
    messageCount: number | null
    /** Tool calls the response asked for; null when the record does not say. */
    toolUses: number | null
  } | null
  session: Totals
  /** Null when no subagent ran; never folded into `session`. */
  forks: Totals | null
  /** What the summary requests cost; additional to `session`, never folded into it. */
  compaction: CompactionSpend
  /** Most recent first, at most `SHOW_COLLAPSES`. */
  collapses: Collapse[]
}

export const SHOW_COLLAPSES = 3

/**
 * Every flagged request, oldest first. The pane shows the tail of this; a
 * review across sessions counts the whole of it.
 */
export const allCollapses = (
  all: readonly Sample[],
  compactions: readonly StoredCompaction[],
): Collapse[] => {
  const main = all.filter(isMainThread)
  const flagged: Collapse[] = []

  for (let i = 0; i < main.length; i += 1) {
    const s = main[i]
    const prev = i > 0 ? main[i - 1] : undefined
    if (isCollapse(s, prev) && prev !== undefined) {
      const cause = attribute(s, prev, compactions)
      flagged.push({
        lost: lostGround(s, prev),
        share: lostShare(s, prev),
        reprocessed: reprocessed(s),
        cause,
        why: causeText(cause, s, prev),
        notes: observations(s, prev, cause, compactions),
        engine: engineNote(lostGround(s, prev), cause, compactionsBetween(s, prev, compactions)),
      })
    }
  }

  return flagged
}

/**
 * The last request's shape as one line, or null when that record predates the
 * fields. A step that called no tools reports none; a step that never said
 * reports nothing at all.
 */
export const shapeText = (last: {
  messageCount: number | null
  toolUses: number | null
}): string | null => {
  const parts: string[] = []
  if (last.messageCount !== null) parts.push(`${last.messageCount} messages`)
  if (last.toolUses !== null) parts.push(`${last.toolUses} tool calls`)

  return parts.length === 0 ? null : parts.join(' · ')
}

export const view = (all: readonly Sample[], compactions: readonly StoredCompaction[]): View => {
  const main = all.filter(isMainThread)
  const forked = all.filter(s => !isMainThread(s))
  const flagged = allCollapses(all, compactions)

  const last = main.length > 0 ? main[main.length - 1] : undefined
  const beforeLast = main.length > 1 ? main[main.length - 2] : undefined

  return {
    isEmpty: main.length === 0,
    last:
      last === undefined
        ? null
        : {
            rate: reuseRate(last),
            held: frontier(last),
            reprocessed: reprocessed(last),
            lost: beforeLast === undefined ? 0 : lostGround(last, beforeLast),
            isCollapse: isCollapse(last, beforeLast),
            messageCount: isDescribed(last) ? last.messageCount : null,
            toolUses: toolUses(last),
          },
    session: totals(main),
    forks: forked.length === 0 ? null : totals(forked),
    compaction: compactionSpend(compactions),
    collapses: flagged.slice(-SHOW_COLLAPSES).reverse(),
  }
}
