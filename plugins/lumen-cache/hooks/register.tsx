import { atom, read, update } from 'claude-code'
import type { Register } from 'claude-code'

import {
  type Fixture,
  type StoredSession,
  compactionRecord,
  compactionsKey,
  fixtureSource,
  isCompactionsKey,
  isSessionKey,
  replay,
  sampleOf,
  sessionsReport,
  shouldTell,
  staleKeys,
  storeKey,
} from './record'
import {
  bar,
  isMainThread,
  pct,
  retain,
  sessionLines,
  spendText,
  shapeText,
  statusText,
  tok,
  totals,
  view,
} from './cost'

const PANE = 'lumen-cache'
const TITLE = 'Cache horizon'

/** Enough main-thread history for a long session; the oldest fall off the front. */
const KEEP_MAIN = 400

/**
 * Forks are not a signal (see isMainThread), so they get a small separate
 * budget: enough to total what subagents cost, never enough to crowd out the
 * main thread.
 */
const KEEP_FORKS = 60

const samples = atom({ plugin: 'lumen-cache', key: 'samples' } as const, [])
const compactions = atom({ plugin: 'lumen-cache', key: 'compactions' } as const, [])

/** Enough to keep old verdicts true without growing without bound. */
const KEEP_COMPACTIONS = 50

/** Sessions kept in the cross-session store; the store caps at 4 MiB of JSON. */
const KEEP_SESSIONS = 20

/** The slice of `$` the persistence helpers use. */
type Store = {
  store: {
    set: (key: string, value: unknown) => Promise<void>
    keys: () => Promise<string[]>
    delete: (key: string) => Promise<void>
  }
  session: { id: () => Promise<string> }
  ui: { toast: (text: string) => void }
}

/** Sessions already told their store writes are failing. */
const told = new Set<string>()

// The store is a convenience copy; the atoms are the record. A rejected
// write (a full 4 MiB store, say) must not take the hook down with it, or
// recording and the status line stop for the rest of the process. One toast
// per session: the status line is rewritten every request, so a note there
// would be gone at once, and a toast per request would be noise.
async function persist($: Store, key: string, value: unknown): Promise<void> {
  try {
    await $.store.set(key, value)
  } catch (err) {
    if (shouldTell(told, await $.session.id())) {
      $.ui.toast(
        `lumen-cache: could not save to the store (${err instanceof Error ? err.message : String(err)}). Measuring continues, but this session will not be kept.`,
      )
    }
  }
}

// The store is a 4 MiB budget shared by every session ever recorded.
async function prune($: Store): Promise<void> {
  try {
    for (const stale of staleKeys(await $.store.keys(), KEEP_SESSIONS)) {
      await $.store.delete(stale)
    }
  } catch {
    // Pruning is housekeeping; failing it must not end the session hook.
  }
}

export const register: Register = on => {
  on('session.start', async ($, e, next) => {
    await $.command.register({
      name: 'lumen-cache',
      description: 'Show how much of the context each request reuses vs reprocesses',
    })
    await $.command.register({
      name: 'lumen-cache-export',
      description: 'Write this session\u2019s measurements out as a replayable fixture',
    })
    await $.command.register({
      name: 'lumen-cache-sessions',
      description: 'Review every recorded session: what each one reprocessed and how much of it was rebuilt',
    })
    // Opened unasked it waits for a wide terminal; /lumen-cache opens it anywhere.
    void $.ui.open({ id: PANE, title: TITLE })

    await prune($)

    return next(e)
  })

  // /clear ends the conversation and fires no session.start for the next one,
  // so this is the only place a cleared session's keys can be pruned behind.
  on('session.end', async ($, e, next) => {
    await prune($)

    return next(e)
  })

  on('command.run', { command: 'lumen-cache-export' }, async $ => {
    const sampled = await read($, samples)
    if (sampled.length === 0) return { text: 'Nothing measured yet \u2014 nothing to export.' }

    const id = await $.session.id()
    const captured: Omit<Fixture, 'expected'> = {
      name: `capture-${id.slice(0, 8)}`,
      capturedAt: new Date().toISOString().slice(0, 10),
      note: `Captured live from session ${id}.`,
      compactions: await read($, compactions),
      samples: sampled,
    }
    const fixture: Fixture = { ...captured, expected: replay(captured) }
    const path = `${await $.session.cwd()}/${fixture.name}.ts`
    await $.fs.write(path, fixtureSource(fixture))

    return {
      text: [
        `Wrote ${sampled.length} measurements to ${path}`,
        '',
        'To make it a regression test, move it into hooks/fixtures/ and add:',
        `  import { fixture as ${fixture.name.replace(/-/g, '_')} } from './fixtures/${fixture.name}'`,
        '',
        'expected was filled from the current logic, so review it before trusting it as a baseline.',
      ].join('\n'),
    }
  })

  // Every session ever recorded, read straight back out of the store. The
  // decisions are all in sessionsReport, which takes the raw keys and values
  // so a test can hand it an old session's records as the store holds them.
  on('command.run', { command: 'lumen-cache-sessions' }, async $ => {
    const entries: StoredSession[] = []
    for (const key of (await $.store.keys()).filter(k => isSessionKey(k) || isCompactionsKey(k))) {
      entries.push({ key, value: await $.store.get(key) })
    }

    return { text: sessionsReport(entries, await $.session.id()) }
  })

  on('command.run', { command: 'lumen-cache' }, async $ => {
    await $.ui.open({ id: PANE, title: TITLE })
    const v = view(await read($, samples), await read($, compactions))

    return {
      text: v.isEmpty
        ? `${TITLE} pane opened. No model request measured yet.`
        : `${TITLE} pane opened. ${v.session.requests} main-thread requests; ${tok(v.session.reprocessed)} tokens reprocessed, ${pct(v.session.rebuiltShare)}% of them rebuilt.`,
    }
  })

  // A compaction rewrites the whole prefix, so it is the one cause we can name.
  //
  // The matcher requires at least one message. `/compact` with nothing to
  // compact dispatches with `messages: []`, and next() refuses to be passed an
  // empty transcript ("a compaction leaves at least one"), so an unguarded hook
  // throws and is skipped — noisily, in the transcript. No compaction happened
  // in that case, so there is nothing to record either.
  on('session.compact', { messages: [{}] }, async ($, e, next) => {
    const done = await next(e)
    // A veto leaves the conversation as it was: nothing compacted, nothing to record.
    if (done.skip !== undefined) return done

    const record = compactionRecord(Date.now(), e.trigger, done)
    await update($, compactions, list => [...list, record].slice(-KEEP_COMPACTIONS))
    // Mirrored, not moved: $.state dies with the session, and a collapse whose
    // compaction time died with it can never be attributed again.
    await persist($, compactionsKey(await $.session.id()), await read($, compactions))

    return done
  })

  // turn.step streams, so the hook is a generator and next(e) is the stream.
  on('turn.step', async function* ($, e, next) {
    // Before the stream drains: the interval a cache cares about starts here.
    const startedAt = Date.now()
    const result = yield* next(e)
    // Both halves of the step make the record: the request's shape from `e`,
    // its cost from the response. Null when the API reported no usage.
    const sample = sampleOf(e, result, Date.now(), startedAt)
    if (sample === null) return result

    await update($, samples, list => retain([...list, sample], KEEP_MAIN, KEEP_FORKS))

    const all = await read($, samples)
    // Kept across sessions too, so captures accumulate rather than vanish.
    await persist($, storeKey(await $.session.id()), all)

    $.ui.status(statusText(totals(all.filter(isMainThread))))

    return result
  })

  on('ui.render', { component: 'Pane', requestId: PANE }, async ($, e) => {
    const { Box, Text } = $.ui.resolve(e)
    const v = view(await read($, samples), await read($, compactions))
    const width = Math.max(10, Math.min(36, (e.props.bodyColumns ?? 44) - 8))

    if (v.isEmpty || v.last === null) {
      return (
        <Box flexDirection="column">
          <Text dimColor>No model request measured yet.</Text>
          <Text dimColor>The pane fills on the first response.</Text>
        </Box>
      )
    }

    const last = v.last

    return (
      <Box flexDirection="column" gap={1}>
        <Box flexDirection="column">
          <Text bold>Last request</Text>
          <Text color={last.isCollapse ? 'yellow' : 'green'}>{bar(last.rate, width)}</Text>
          <Text>{pct(last.rate)}% reused</Text>
          <Text dimColor>
            {tok(last.held)} held · {tok(last.reprocessed)} reprocessed
          </Text>
          {shapeText(last) !== null && <Text dimColor>{shapeText(last)}</Text>}
        </Box>

        <Box flexDirection="column">
          <Text bold>Session · main thread</Text>
          <Text>{sessionLines(v.session)[0]}</Text>
          <Text>{sessionLines(v.session)[1]}</Text>
          {sessionLines(v.session)[2] !== '' && <Text dimColor>{sessionLines(v.session)[2]}</Text>}
          {spendText(v.compaction) !== null && <Text dimColor>{spendText(v.compaction)}</Text>}
        </Box>

        {v.forks !== null && (
          <Box flexDirection="column">
            <Text bold dimColor>Subagents</Text>
            <Text dimColor>
              {v.forks.requests} requests · {tok(v.forks.reprocessed)} reprocessed
            </Text>
            <Text dimColor>A fork pays for the whole prefix; not a signal.</Text>
          </Box>
        )}

        {v.collapses.length > 0 && (
          <Box flexDirection="column">
            <Text bold color="yellow">
              {v.collapses.length === 1 ? '1 collapse' : `${v.collapses.length} collapses`}
            </Text>
            {v.collapses.map(one => (
              <Box flexDirection="column">
                <Text color="yellow">
                  {tok(one.lost)} lost · {pct(one.share)}% of the prior context
                </Text>
                <Text dimColor>{tok(one.reprocessed)} reprocessed</Text>
                <Text dimColor>{one.why}</Text>
                {one.engine !== null && (
                  <Text color={one.engine.disagrees ? 'red' : undefined} dimColor={!one.engine.disagrees}>
                    {one.engine.text}
                  </Text>
                )}
                {one.notes.map(note => (
                  <Text dimColor>{note}</Text>
                ))}
              </Box>
            ))}
          </Box>
        )}

        <Box flexDirection="column">
          <Text dimColor>cost = T - a, where a is what the cache held.</Text>
          <Text dimColor>Measured, not estimated.</Text>
        </Box>
      </Box>
    )
  })
}
