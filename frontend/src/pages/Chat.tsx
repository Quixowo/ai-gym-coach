/**
 * Chat — the coach conversation page.
 *
 * Layout: full-height panel with message list + fixed input at the bottom.
 * The trace panel (tool call events) is collected by useChatStream but
 * rendered in a later phase.
 *
 * Design: matches the existing token system (navy-charcoal bg, #FF4D00 accent,
 * Space Grotesk / Inter). User bubbles sit right-aligned; coach bubbles sit
 * left-aligned with a subtle accent-left-border to read as "system voice".
 * Streaming is signaled by a blinking cursor appended to the last coach token.
 */

import {
  useEffect,
  useRef,
  useState,
  type FormEvent,
  type KeyboardEvent,
} from 'react'
import { useChatStream, type ChatMessage } from '../hooks/useChatStream'

// ---- Message bubble ----

interface BubbleProps {
  message: ChatMessage
  onRetry?: () => void
}

function Bubble({ message, onRetry }: BubbleProps) {
  const isUser = message.role === 'user'

  if (isUser) {
    return (
      <div className="chat-bubble-row chat-bubble-row--user">
        <div className="chat-bubble chat-bubble--user">
          {message.content}
        </div>
      </div>
    )
  }

  // Coach bubble — may be streaming, errored, or rate-limited.
  const showCursor = message.streaming && !message.error && !message.rateLimited
  const isEmpty = !message.content && !message.streaming && !message.error && !message.rateLimited

  return (
    <div className="chat-bubble-row chat-bubble-row--coach">
      <div
        className={[
          'chat-bubble',
          'chat-bubble--coach',
          message.error || message.rateLimited ? 'chat-bubble--error' : '',
          message.rateLimited ? 'chat-bubble--rate-limited' : '',
        ]
          .filter(Boolean)
          .join(' ')}
      >
        {isEmpty ? (
          // Should rarely appear — the streaming placeholder.
          <span className="chat-cursor" aria-hidden="true" />
        ) : (
          <>
            {/* Rate-limit message */}
            {message.rateLimited && (
              <span>{message.content}</span>
            )}

            {/* Normal content (streaming or complete) */}
            {!message.rateLimited && (
              <span>
                {message.content}
                {showCursor && (
                  <span className="chat-cursor" aria-hidden="true" />
                )}
              </span>
            )}

            {/* Graceful error from backend's error event or network drop */}
            {message.error && !message.rateLimited && (
              <div className="chat-error-row">
                <span className="chat-error-label">
                  {message.content
                    ? 'Response interrupted — '
                    : `${message.error} — `}
                </span>
                {onRetry && (
                  <button
                    type="button"
                    className="chat-retry-btn"
                    onClick={onRetry}
                    aria-label="Retry sending the last message"
                  >
                    retry?
                  </button>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}

// ---- Empty state ----

function EmptyState() {
  return (
    <div className="empty-state" style={{ flex: 1, justifyContent: 'center' }}>
      <div className="empty-state-icon" aria-hidden="true">💬</div>
      <h3>Ask your coach</h3>
      <p>
        Ask about your workouts, progression, or training principles.
        Your conversation resets on refresh — that's intentional.
      </p>
    </div>
  )
}

// ---- Chat page ----

export function Chat() {
  const { messages, status, sendMessage, retry } = useChatStream()
  const [input, setInput] = useState('')
  const bottomRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const isStreaming = status === 'streaming'

  // Auto-scroll to newest message.
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  function handleSubmit(e?: FormEvent) {
    e?.preventDefault()
    const text = input.trim()
    if (!text || isStreaming) return
    setInput('')
    void sendMessage(text)
  }

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    // Enter sends; Shift+Enter inserts a newline.
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  // Determine if the last message errored (to know whether to show retry).
  const lastMsg = messages[messages.length - 1]
  const lastIsError =
    lastMsg &&
    lastMsg.role === 'assistant' &&
    (lastMsg.error != null || lastMsg.rateLimited)

  return (
    <div className="chat-page">
      {/* Message list */}
      <div className="chat-messages" aria-live="polite" aria-label="Conversation">
        {messages.length === 0 ? (
          <EmptyState />
        ) : (
          messages.map((msg) => (
            <Bubble
              key={msg.id}
              message={msg}
              onRetry={
                lastIsError && msg.id === lastMsg.id && !msg.rateLimited
                  ? retry
                  : undefined
              }
            />
          ))
        )}
        <div ref={bottomRef} aria-hidden="true" />
      </div>

      {/* Input area */}
      <div className="chat-input-wrap">
        <form className="chat-input-form" onSubmit={handleSubmit}>
          <textarea
            ref={inputRef}
            className="chat-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask your coach…"
            rows={1}
            disabled={isStreaming}
            aria-label="Message input"
          />
          <button
            type="submit"
            className="btn btn-primary chat-send-btn"
            disabled={isStreaming || !input.trim()}
            aria-label="Send message"
          >
            {isStreaming ? (
              <span className="spinner" style={{ width: 16, height: 16 }} />
            ) : (
              /* Simple send arrow */
              <svg
                width="16"
                height="16"
                viewBox="0 0 16 16"
                fill="none"
                aria-hidden="true"
              >
                <path
                  d="M2 8L14 2L8 14L7 9L2 8Z"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinejoin="round"
                />
              </svg>
            )}
          </button>
        </form>
        <p className="chat-input-hint">
          Enter to send · Shift+Enter for new line
        </p>
      </div>
    </div>
  )
}
