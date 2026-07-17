/**
 * TracePanel — live agent decision log.
 *
 * Renders the traceEvents list emitted by useChatStream as each SSE frame
 * arrives. Styled as a console/instrument readout: monospace labels, tabular
 * latency numbers in the scoreboard aesthetic, muted result summaries.
 *
 * Event rendering:
 *   tool_call_started   → "Calling {tool}…"  with an in-progress pulse
 *   tool_call_completed → checkmark + tool name + latency_ms + result_summary
 *   turn_complete       → divider line with iteration + total latency summary
 *   error               → flagged step in danger treatment
 *
 * Design decisions:
 *   - Dark panel with a left border in --color-border; feels like a terminal
 *     instrument panel native to the scoreboard aesthetic.
 *   - Latency numbers use tabular-nums and the scoreboard's oversized-num
 *     pattern scaled down (scoreboard-num-sm) so they read as data.
 *   - tool_call_started uses a subtle pulse dot matching session-status-dot.
 *   - turn_complete is a full-width hairline divider so multiple turns group
 *     visually as the event list grows.
 *   - No clear/reset: traceEvents accumulate across turns by design (live-only,
 *     gone on refresh per CLAUDE.md invariant).
 */

import { useEffect, useRef } from 'react'
import { Check, X, Activity } from 'lucide-react'
import type { TraceEvent } from '../hooks/useChatStream'

// ---- Individual event rows ----

interface TraceRowProps {
  event: TraceEvent
}

function ToolCallStartedRow({ event }: TraceRowProps) {
  const toolList = event.tools?.join(', ') ?? 'unknown'
  return (
    <div className="trace-row trace-row--started" role="listitem">
      <span className="trace-pulse-dot" aria-hidden="true" />
      <div className="trace-row-body">
        <span className="trace-label trace-label--calling">
          Calling
        </span>
        <span className="trace-tool-name">{toolList}</span>
        <span className="trace-ellipsis">…</span>
      </div>
    </div>
  )
}

function ToolCallCompletedRow({ event }: TraceRowProps) {
  return (
    <div className="trace-row trace-row--completed" role="listitem">
      <span className="trace-check" aria-label="completed">
        <Check size={14} strokeWidth={2.5} />
      </span>
      <div className="trace-row-body">
        <span className="trace-tool-name">{event.tool ?? 'unknown'}</span>
        {event.latency_ms != null && (
          <span className="trace-latency">
            <span className="trace-latency-num">{event.latency_ms}</span>
            <span className="trace-latency-unit">ms</span>
          </span>
        )}
        {event.result_summary && (
          <span className="trace-summary">{event.result_summary}</span>
        )}
      </div>
    </div>
  )
}

function TurnCompleteRow({ event }: TraceRowProps) {
  return (
    <div className="trace-turn-divider" role="listitem" aria-label="Turn complete">
      <div className="trace-turn-line" aria-hidden="true" />
      <div className="trace-turn-summary">
        <span className="trace-turn-label">Turn</span>
        {event.iterations != null && (
          <>
            <span className="trace-turn-num">{event.iterations}</span>
            <span className="trace-turn-unit">
              {event.iterations === 1 ? 'iteration' : 'iterations'}
            </span>
          </>
        )}
        {event.total_latency_ms != null && (
          <>
            <span className="trace-turn-sep">·</span>
            <span className="trace-turn-num">{event.total_latency_ms}</span>
            <span className="trace-turn-unit">ms total</span>
          </>
        )}
      </div>
      <div className="trace-turn-line" aria-hidden="true" />
    </div>
  )
}

function ErrorRow({ event }: TraceRowProps) {
  return (
    <div className="trace-row trace-row--error" role="listitem" aria-live="assertive">
      <span className="trace-error-flag" aria-label="error">
        <X size={14} strokeWidth={2.5} />
      </span>
      <div className="trace-row-body">
        <span className="trace-label trace-label--error">Agent error</span>
        {event.message && (
          <span className="trace-summary trace-summary--error">{event.message}</span>
        )}
      </div>
    </div>
  )
}

// ---- Empty state ----

function TracePanelEmpty() {
  return (
    <div className="trace-empty">
      <div className="trace-empty-icon" aria-hidden="true">
        <Activity size={28} strokeWidth={1.5} />
      </div>
      <p className="trace-empty-text">Agent activity will appear here</p>
    </div>
  )
}

// ---- Panel ----

interface TracePanelProps {
  events: TraceEvent[]
}

export function TracePanel({ events }: TracePanelProps) {
  const bodyRef = useRef<HTMLDivElement>(null)

  // Keep the newest event visible as the stream appends — pin the scroll
  // container's own scrollTop rather than scrollIntoView so the page never moves.
  useEffect(() => {
    const el = bodyRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [events])

  return (
    <aside className="trace-panel" aria-label="Agent trace log">
      <header className="trace-panel-header">
        <span className="trace-panel-title">Trace</span>
        <span className="trace-panel-subtitle">live · resets on refresh</span>
      </header>
      <div
        ref={bodyRef}
        className="trace-panel-body"
        role="list"
        aria-label="Agent decisions"
        aria-live="polite"
        aria-atomic="false"
        aria-relevant="additions"
      >
        {events.length === 0 ? (
          <TracePanelEmpty />
        ) : (
          events.map((event) => {
            if (event.type === 'tool_call_started') {
              return <ToolCallStartedRow key={event.id} event={event} />
            }
            if (event.type === 'tool_call_completed') {
              return <ToolCallCompletedRow key={event.id} event={event} />
            }
            if (event.type === 'turn_complete') {
              return <TurnCompleteRow key={event.id} event={event} />
            }
            if (event.type === 'error') {
              return <ErrorRow key={event.id} event={event} />
            }
            return null
          })
        )}
      </div>
    </aside>
  )
}
