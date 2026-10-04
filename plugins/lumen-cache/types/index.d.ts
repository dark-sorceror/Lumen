/** `TurnStepResult.stopReason`'s union, which 'claude-code' does not export. */
export type CacheStopReason =
  | 'end_turn'
  | 'max_tokens'
  | 'stop_sequence'
  | 'tool_use'
  | 'pause_turn'
  | 'compaction'
  | 'refusal'
  | 'model_context_window_exceeded'
  | null

/** `TurnStepInput.effort`. */
export type CacheEffort = 'low' | 'medium' | 'high' | 'xhigh' | 'max' | number

/**
 * One measured request: what it cost, and what the request itself looked like.
 *
 * The four fields below `at` arrived after the first version of this mod, so
 * every one of them is optional: the state and the cross-session store both
 * hold records written without them, and a record that does not say is never
 * read as a record that says zero. This mirrors `Sample` in hooks/cost.ts.
 */
export type CacheSample = {
  turnId: string
  index: number
  agentId?: string
  model: string
  reused: number
  written: number
  uncached: number
  output: number
  at: number
  /** When the step began, before it streamed; absent on records from before it was kept. */
  startedAt?: number
  /** `TurnStepInput.messageCount` — messages the request carried. */
  messageCount?: number
  /** Why the model stopped; `null` when no response arrived. */
  stopReason?: CacheStopReason
  /** `TurnStepResult.toolUses.length` — tool calls the response asked for. */
  toolUseCount?: number
  /** Absent for a model without an effort setting. */
  effort?: CacheEffort
}

/**
 * One compaction. Only `at` is always known; the rest is whatever the event
 * supplied, and an absent field is unknown, never zero. Mirrors `Compaction`
 * in hooks/cost.ts.
 */
export type CacheCompaction = {
  at: number
  trigger?: 'manual' | 'auto' | 'plugin' | 'precompute'
  tokensBefore?: number
  tokensAfter?: number
  usage?: {
    input_tokens: number
    output_tokens: number
    cache_read_input_tokens: number
    cache_creation_input_tokens: number
  }
}

declare module 'claude-code' {
  interface PluginState {
    'lumen-cache': {
      samples: CacheSample[]
      /** A bare number is a record from before the engine's figures were kept. */
      compactions: (number | CacheCompaction)[]
    }
  }
}
