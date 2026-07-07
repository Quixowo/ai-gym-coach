/**
 * useChatStream — streams POST /chat, parses SSE frames, exposes chat state.
 *
 * Transport: fetch-based stream reader (NOT EventSource) because /chat is a
 * POST with a JSON body and requires cookies.  See BUILD_SPEC.md §13.
 *
 * Frame format from backend:
 *   data: {"type":"text_delta","text":"..."}\n\n
 *   data: {"type":"tool_call_started","tools":[...]}\n\n
 *   data: {"type":"tool_call_completed","tool":"...","latency_ms":12,"result_summary":"..."}\n\n
 *   data: {"type":"turn_complete","iterations":2,"total_latency_ms":1840}\n\n
 *   data: {"type":"error","message":"..."}\n\n
 */

import { useState, useRef, useCallback } from 'react'
import { apiFetch } from '../api/client'

// ---- Public types ----

export type MessageRole = 'user' | 'assistant'

export interface ChatMessage {
  id: string
  role: MessageRole
  content: string
  /** True while this message is still being streamed in. */
  streaming?: boolean
  /** Non-null when this message finished with a graceful error event. */
  error?: string | null
  /** Rate-limited placeholder — display the 429 bubble. */
  rateLimited?: boolean
}

export interface TraceEvent {
  id: string
  type: 'tool_call_started' | 'tool_call_completed' | 'turn_complete'
  // tool_call_started
  tools?: string[]
  // tool_call_completed
  tool?: string
  latency_ms?: number
  result_summary?: string
  // turn_complete
  iterations?: number
  total_latency_ms?: number
}

export type ChatStatus = 'idle' | 'streaming' | 'done' | 'error'

export interface UseChatStreamReturn {
  messages: ChatMessage[]
  traceEvents: TraceEvent[]
  status: ChatStatus
  sendMessage: (text: string) => Promise<void>
  retry: () => void
}

// ---- Helpers ----

let _idCounter = 0
function makeId(): string {
  return `msg-${++_idCounter}-${Math.random().toString(36).slice(2, 6)}`
}

/**
 * Parse all complete SSE frames out of the accumulated buffer.
 * Returns { payloads, remaining } where remaining is any partial frame
 * that hasn't received its closing \n\n yet.
 *
 * Handles:
 *  - Multiple frames per chunk (split on \n\n)
 *  - Frames split across reads (buffered until \n\n arrives)
 *  - Lines without the "data: " prefix are silently skipped
 */
function parseFrames(buffer: string): { payloads: string[]; remaining: string } {
  const payloads: string[] = []
  const parts = buffer.split('\n\n')
  // The last element is either '' (if buffer ended with \n\n) or a partial frame.
  const remaining = parts.pop() ?? ''

  for (const part of parts) {
    for (const line of part.split('\n')) {
      if (line.startsWith('data: ')) {
        payloads.push(line.slice(6).trim())
        break // only one data line per SSE frame
      }
    }
  }

  return { payloads, remaining }
}

// ---- Hook ----

export function useChatStream(): UseChatStreamReturn {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [traceEvents, setTraceEvents] = useState<TraceEvent[]>([])
  const [status, setStatus] = useState<ChatStatus>('idle')

  // Parallel ref to messages so we can read the current list synchronously
  // without relying on stale closure captures.
  const messagesRef = useRef<ChatMessage[]>([])

  // Sync messagesRef whenever state changes.
  const setMessagesSync = useCallback((updater: ChatMessage[] | ((prev: ChatMessage[]) => ChatMessage[])) => {
    setMessages((prev) => {
      const next = typeof updater === 'function' ? updater(prev) : updater
      messagesRef.current = next
      return next
    })
  }, [])

  // Track the last user text so retry() can resend it.
  const lastUserText = useRef<string | null>(null)
  // ID of the in-progress assistant message.
  const assistantMsgId = useRef<string | null>(null)
  // Guard: prevent concurrent sends.
  const streamingRef = useRef(false)

  // ---- Core stream logic ----

  const runStream = useCallback(async (userText: string, historySnapshot: { role: 'user' | 'assistant'; content: string }[]) => {
    streamingRef.current = true
    lastUserText.current = userText

    // Append the user message immediately.
    const userMsgId = makeId()
    setMessagesSync((prev) => [
      ...prev,
      { id: userMsgId, role: 'user', content: userText },
    ])

    setStatus('streaming')

    // Create a placeholder assistant message.
    const aId = makeId()
    assistantMsgId.current = aId
    setMessagesSync((prev) => [
      ...prev,
      { id: aId, role: 'assistant', content: '', streaming: true },
    ])

    try {
      const res = await apiFetch('/chat', {
        method: 'POST',
        body: JSON.stringify({
          message: userText,
          history: historySnapshot,
        }),
      })

      // Handle non-streaming error responses (e.g. 429, 422, 500).
      if (!res.ok) {
        if (res.status === 429) {
          setMessagesSync((prev) =>
            prev.map((m) =>
              m.id === aId
                ? {
                    ...m,
                    content: "You're sending messages too fast — try again in a moment.",
                    streaming: false,
                    rateLimited: true,
                  }
                : m,
            ),
          )
          setStatus('error')
          return
        }

        let detail = `Request failed (${res.status}).`
        try {
          const body = await res.json() as { detail?: string }
          if (body.detail) detail = body.detail
        } catch { /* ignore */ }

        setMessagesSync((prev) =>
          prev.map((m) =>
            m.id === aId
              ? { ...m, content: '', streaming: false, error: detail }
              : m,
          ),
        )
        setStatus('error')
        return
      }

      if (!res.body) {
        throw new Error('Response has no body stream.')
      }

      // ---- Read the SSE stream ----
      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      // eslint-disable-next-line no-constant-condition
      while (true) {
        // reader.read() rejects on network drop — caught by the outer try/catch.
        const readResult: ReadableStreamReadResult<Uint8Array> = await reader.read()

        if (readResult.done) break

        buffer += decoder.decode(readResult.value, { stream: true })

        const { payloads, remaining } = parseFrames(buffer)
        buffer = remaining

        for (const payload of payloads) {
          let event: Record<string, unknown>
          try {
            event = JSON.parse(payload) as Record<string, unknown>
          } catch {
            continue // malformed frame — skip
          }

          const type = event.type as string

          if (type === 'text_delta') {
            const text = (event.text as string) ?? ''
            setMessagesSync((prev) =>
              prev.map((m) =>
                m.id === aId ? { ...m, content: m.content + text } : m,
              ),
            )
          } else if (type === 'tool_call_started') {
            const tools = (event.tools as string[]) ?? []
            setTraceEvents((prev) => [
              ...prev,
              { id: makeId(), type: 'tool_call_started', tools },
            ])
          } else if (type === 'tool_call_completed') {
            setTraceEvents((prev) => [
              ...prev,
              {
                id: makeId(),
                type: 'tool_call_completed',
                tool: event.tool as string,
                latency_ms: event.latency_ms as number,
                result_summary: event.result_summary as string,
              },
            ])
          } else if (type === 'turn_complete') {
            setTraceEvents((prev) => [
              ...prev,
              {
                id: makeId(),
                type: 'turn_complete',
                iterations: event.iterations as number,
                total_latency_ms: event.total_latency_ms as number,
              },
            ])
            setMessagesSync((prev) =>
              prev.map((m) =>
                m.id === aId ? { ...m, streaming: false } : m,
              ),
            )
            setStatus('done')
            return
          } else if (type === 'error') {
            const message = (event.message as string) ?? 'The coach encountered an error.'
            setMessagesSync((prev) =>
              prev.map((m) =>
                m.id === aId ? { ...m, streaming: false, error: message } : m,
              ),
            )
            setStatus('error')
            return
          }
        }
      }

      // Stream ended (reader.done) without a turn_complete — treat as disconnect.
      setMessagesSync((prev) =>
        prev.map((m) =>
          m.id === aId && m.streaming
            ? { ...m, streaming: false, error: 'Response interrupted.' }
            : m,
        ),
      )
      setStatus('error')
    } catch {
      // Network error or reader exception mid-stream.
      // Keep partial content visible, mark error.
      setMessagesSync((prev) =>
        prev.map((m) =>
          m.id === assistantMsgId.current && m.streaming
            ? { ...m, streaming: false, error: 'Connection lost.' }
            : m,
        ),
      )
      setStatus('error')
    } finally {
      streamingRef.current = false
      assistantMsgId.current = null
    }
  }, [setMessagesSync])

  const sendMessage = useCallback(
    async (text: string) => {
      const trimmed = text.trim()
      if (!trimmed) return
      if (streamingRef.current) return

      // Capture history synchronously from the ref before any state updates.
      // Only include finalized messages (not in-progress streaming or rate-limited).
      const history = messagesRef.current
        .filter((m) => !m.streaming && !m.rateLimited && !m.error)
        .map((m) => ({ role: m.role as 'user' | 'assistant', content: m.content }))

      await runStream(trimmed, history)
    },
    [runStream],
  )

  const retry = useCallback(() => {
    if (streamingRef.current) return
    const text = lastUserText.current
    if (!text) return

    // Remove the errored assistant message and the user message that triggered it,
    // then resend. This keeps the conversation clean without duplicating messages.
    setMessagesSync((prev) => {
      let end = prev.length - 1
      // Remove trailing assistant error/rate-limited msg.
      if (
        end >= 0 &&
        prev[end].role === 'assistant' &&
        (prev[end].error != null || prev[end].rateLimited)
      ) {
        end--
      }
      // Remove the user msg that triggered it.
      if (end >= 0 && prev[end].role === 'user' && prev[end].content === text) {
        end--
      }
      return prev.slice(0, end + 1)
    })

    setStatus('idle')
    setTraceEvents([])

    // History is everything that remains after the trim above.
    // We read from the ref after the setMessagesSync call above.
    // Because setMessages is batched, we compute history after the trim
    // using a microtask so the ref is updated.
    void Promise.resolve().then(() => {
      const history = messagesRef.current
        .filter((m) => !m.streaming && !m.rateLimited && !m.error)
        .map((m) => ({ role: m.role as 'user' | 'assistant', content: m.content }))
      void runStream(text, history)
    })
  }, [runStream, setMessagesSync])

  return { messages, traceEvents, status, sendMessage, retry }
}
