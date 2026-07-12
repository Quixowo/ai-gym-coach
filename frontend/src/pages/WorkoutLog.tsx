/**
 * WorkoutLog — the core page.
 *
 * Flow:
 *  1. No active session → show "Start workout" button + recent history.
 *  2. POST /workouts/sessions → if 409 (open session exists), offer to
 *     resume (GET /workouts/sessions, find open one) or finish it.
 *  3. Active session → exercise picker (debounced search), set entry form,
 *     scoreboard set display, "Finish workout".
 *  4. After finish → optional "Save as program" modal.
 *  5. Recent history shown at the bottom.
 */

import {
  useState,
  useEffect,
  useRef,
  useCallback,
  type FormEvent,
} from 'react'
import { ClipboardList, Dumbbell } from 'lucide-react'
import {
  listSessions,
  startSession,
  finishSession,
  logSet,
  getHistory,
  type WorkoutSession,
  type SetEntry,
  type HistoryEntry,
} from '../api/workouts'
import {
  searchExercises,
  listExercises,
  type Exercise,
  type ExerciseMatch,
} from '../api/exercises'
import { createProgram } from '../api/programs'
import { ApiError } from '../api/client'

// ---- Types ----

interface LoggedSet extends SetEntry {
  exercise_name: string
}

interface ExerciseGroup {
  exerciseId: string
  exerciseName: string
  sets: LoggedSet[]
}

// ---- Helper: group sets by exercise ----

function groupByExercise(sets: LoggedSet[]): ExerciseGroup[] {
  const map = new Map<string, ExerciseGroup>()
  for (const s of sets) {
    if (!map.has(s.exercise_id)) {
      map.set(s.exercise_id, {
        exerciseId: s.exercise_id,
        exerciseName: s.exercise_name,
        sets: [],
      })
    }
    map.get(s.exercise_id)!.sets.push(s)
  }
  return Array.from(map.values())
}

// ---- SaveAsProgramModal ----

interface SaveAsProgramModalProps {
  groups: ExerciseGroup[]
  onSave: (name: string) => Promise<void>
  onSkip: () => void
}

function SaveAsProgramModal({ groups, onSave, onSkip }: SaveAsProgramModalProps) {
  const [name, setName] = useState('')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function handleSave(e: FormEvent) {
    e.preventDefault()
    if (!name.trim()) { setError('Enter a name for this program.'); return }
    setSaving(true)
    try {
      await onSave(name.trim())
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Failed to save. Try again.')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div
      style={{
        position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.7)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        zIndex: 100, padding: 'var(--sp-4)',
      }}
      role="dialog"
      aria-modal="true"
      aria-labelledby="sap-title"
    >
      <div
        className="card"
        style={{ padding: 'var(--sp-6)', maxWidth: 400, width: '100%', display: 'flex', flexDirection: 'column', gap: 'var(--sp-4)' }}
      >
        <h2 id="sap-title">Save as program</h2>
        <p className="text-sm text-muted">
          Turn today's {groups.length} exercise{groups.length !== 1 ? 's' : ''} into
          a repeatable program with target weights and reps.
        </p>

        <form onSubmit={handleSave} style={{ display: 'flex', flexDirection: 'column', gap: 'var(--sp-3)' }}>
          <div className="field">
            <label htmlFor="prog-name">Program name</label>
            <input
              id="prog-name"
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. Push A"
              autoFocus
            />
          </div>

          {error && <div className="error-banner" role="alert">{error}</div>}

          <div style={{ display: 'flex', gap: 'var(--sp-3)' }}>
            <button type="submit" className="btn btn-success" disabled={saving} style={{ flex: 1 }}>
              {saving ? 'Saving…' : 'Save program'}
            </button>
            <button type="button" className="btn btn-ghost" onClick={onSkip}>
              Skip
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

// ---- ExercisePicker ----

interface ExercisePickerProps {
  onSelect: (ex: { id: string; name: string }) => void
}

function ExercisePicker({ onSelect }: ExercisePickerProps) {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<(ExerciseMatch | Exercise)[]>([])
  const [open, setOpen] = useState(false)
  const [loading, setLoading] = useState(false)
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  // Close on outside click.
  useEffect(() => {
    function onClick(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', onClick)
    return () => document.removeEventListener('mousedown', onClick)
  }, [])

  // Load full list on mount for fallback.
  const [allExercises, setAllExercises] = useState<Exercise[]>([])
  useEffect(() => {
    listExercises().then(setAllExercises).catch(() => { /* silent */ })
  }, [])

  function handleChange(val: string) {
    setQuery(val)
    if (debounceRef.current) clearTimeout(debounceRef.current)

    if (!val.trim()) {
      setResults(allExercises.slice(0, 20))
      setOpen(allExercises.length > 0)
      return
    }

    debounceRef.current = setTimeout(async () => {
      setLoading(true)
      try {
        const matches = await searchExercises(val)
        setResults(matches.length ? matches : allExercises.filter(ex =>
          ex.name.toLowerCase().includes(val.toLowerCase())
        ).slice(0, 20))
      } catch {
        setResults(allExercises.filter(ex =>
          ex.name.toLowerCase().includes(val.toLowerCase())
        ).slice(0, 20))
      } finally {
        setLoading(false)
      }
      setOpen(true)
    }, 250)
  }

  function getName(item: ExerciseMatch | Exercise): string {
    return item.name
  }

  function getId(item: ExerciseMatch | Exercise): string {
    if ('exercise_id' in item) return item.exercise_id
    return item.id
  }

  function handleSelect(item: ExerciseMatch | Exercise) {
    onSelect({ id: getId(item), name: getName(item) })
    setQuery('')
    setOpen(false)
    setResults([])
  }

  return (
    <div className="exercise-picker" ref={containerRef}>
      <div className="field">
        <label htmlFor="exercise-search">Exercise</label>
        <input
          id="exercise-search"
          type="search"
          value={query}
          onChange={(e) => handleChange(e.target.value)}
          onFocus={() => {
            if (!query) {
              setResults(allExercises.slice(0, 20))
              if (allExercises.length) setOpen(true)
            } else {
              setOpen(true)
            }
          }}
          placeholder="Search exercises…"
          autoComplete="off"
        />
      </div>

      {open && results.length > 0 && (
        <div className="exercise-results" role="listbox" aria-label="Exercise results">
          {loading && (
            <div style={{ padding: 'var(--sp-3)', display: 'flex', gap: 'var(--sp-2)', alignItems: 'center', color: 'var(--color-text-2)', fontSize: '0.875rem' }}>
              <span className="spinner" style={{ width: 14, height: 14 }} />
              Searching…
            </div>
          )}
          {results.map((item) => {
            const id = getId(item)
            const name = getName(item)
            const meta = 'primary_muscle_group' in item
              ? `${item.primary_muscle_group} · ${item.equipment}`
              : undefined
            return (
              <button
                key={id}
                type="button"
                className="exercise-result-item"
                role="option"
                aria-selected="false"
                onClick={() => handleSelect(item)}
              >
                <span className="exercise-result-name">{name}</span>
                {meta && <span className="exercise-result-meta">{meta}</span>}
              </button>
            )
          })}
        </div>
      )}
    </div>
  )
}

// ---- SetForm ----

interface SetFormProps {
  exercise: { id: string; name: string }
  sessionId: string
  onLogged: (set: LoggedSet) => void
  onClear: () => void
}

function SetForm({ exercise, sessionId: _sessionId, onLogged, onClear }: SetFormProps) {
  const [weight, setWeight] = useState('')
  const [reps, setReps] = useState('')
  const [rir, setRir] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Suppress unused param warning — sessionId is already scoped to user server-side.
  void _sessionId

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    setError(null)
    const weightVal = parseFloat(weight)
    const repsVal = parseInt(reps, 10)
    const rirVal = rir.trim() !== '' ? parseFloat(rir) : null

    if (isNaN(weightVal) || weightVal < 0) { setError('Enter a valid weight (≥ 0 lbs).'); return }
    if (isNaN(repsVal) || repsVal < 1 || repsVal > 100) { setError('Enter reps between 1 and 100.'); return }
    if (rirVal !== null && (isNaN(rirVal) || rirVal < 0 || rirVal > 10)) {
      setError('RIR must be 0–10 (0.5 steps).')
      return
    }

    setSubmitting(true)
    try {
      const entry = await logSet({
        exercise_id: exercise.id,
        weight: weightVal,
        reps: repsVal,
        rir: rirVal,
      })
      onLogged({ ...entry, exercise_name: exercise.name })
      // Keep weight but clear reps so the next set is fast to enter.
      setReps('')
      setRir('')
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Failed to log set.')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="set-form">
      <div className="flex items-center justify-between">
        <h3 style={{ fontSize: '1rem' }}>{exercise.name}</h3>
        <button type="button" className="btn btn-ghost btn-sm" onClick={onClear}>
          Change exercise
        </button>
      </div>

      <form onSubmit={handleSubmit}>
        <div className="set-form-fields">
          <div className="field">
            <label htmlFor="sf-weight">Weight (lbs)</label>
            <input
              id="sf-weight"
              type="number"
              inputMode="decimal"
              min="0"
              step="2.5"
              value={weight}
              onChange={(e) => setWeight(e.target.value)}
              placeholder="135"
            />
          </div>
          <div className="field">
            <label htmlFor="sf-reps">Reps</label>
            <input
              id="sf-reps"
              type="number"
              inputMode="numeric"
              min="1"
              max="100"
              value={reps}
              onChange={(e) => setReps(e.target.value)}
              placeholder="8"
            />
          </div>
          <div className="field">
            <label htmlFor="sf-rir">RIR <span style={{ color: 'var(--color-text-3)', fontWeight: 400 }}>(opt)</span></label>
            <input
              id="sf-rir"
              type="number"
              inputMode="decimal"
              min="0"
              max="10"
              step="0.5"
              value={rir}
              onChange={(e) => setRir(e.target.value)}
              placeholder="2"
            />
          </div>
        </div>

        {error && <div className="error-banner mt-3" role="alert">{error}</div>}

        <button
          type="submit"
          className="btn btn-primary btn-full mt-4"
          disabled={submitting}
        >
          {submitting ? (
            <>
              <span className="spinner" style={{ width: 16, height: 16 }} />
              Logging…
            </>
          ) : (
            'Log set'
          )}
        </button>
      </form>
    </div>
  )
}

// ---- Scoreboard (signature element) ----

function Scoreboard({ groups }: { groups: ExerciseGroup[] }) {
  if (!groups.length) return null

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--sp-5)' }}>
      {groups.map((group) => (
        <div key={group.exerciseId}>
          <div style={{
            fontFamily: 'var(--font-display)',
            fontWeight: 600,
            fontSize: '0.875rem',
            color: 'var(--color-text-2)',
            letterSpacing: '0.04em',
            textTransform: 'uppercase',
            marginBottom: 'var(--sp-2)',
          }}>
            {group.exerciseName}
          </div>
          <div className="scoreboard-grid">
            {group.sets.map((s) => (
              <div key={s.id} className="scoreboard-row">
                <span className="scoreboard-set-num">S{s.set_number}</span>

                <div className="scoreboard-val">
                  <span className="scoreboard-num">{s.weight}</span>
                  <span className="scoreboard-unit">lbs</span>
                </div>

                <span className="scoreboard-sep" aria-hidden="true">×</span>

                <div className="scoreboard-val">
                  <span className="scoreboard-num">{s.reps}</span>
                  <span className="scoreboard-unit">reps</span>
                </div>

                <div className="scoreboard-val" style={{ textAlign: 'right' }}>
                  {s.rir !== null ? (
                    <>
                      <span className="scoreboard-num" style={{ fontSize: '1rem', color: 'var(--color-text-2)' }}>
                        {s.rir}
                      </span>
                      <span className="scoreboard-unit">rir</span>
                    </>
                  ) : (
                    <span className="scoreboard-unit">—</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  )
}

// ---- History ----

function History({ entries }: { entries: HistoryEntry[] }) {
  if (!entries.length) {
    return (
      <div className="empty-state">
        <div className="empty-state-icon" aria-hidden="true">
          <ClipboardList size={40} strokeWidth={1.5} />
        </div>
        <h3>No history yet</h3>
        <p>Finish a workout and your sets will show up here.</p>
      </div>
    )
  }

  // Group by session date (most recent first, show last 30 entries).
  const recent = entries.slice(0, 30)

  return (
    <div>
      {recent.map((e) => (
        <div key={e.id} className="history-entry">
          <div className="history-exercise">{e.exercise_name}</div>
          <div className="history-stats">
            <span>{e.weight} lbs</span>
            <span>×</span>
            <span>{e.reps} reps</span>
            {e.rir !== null && <span>{e.rir} RIR</span>}
          </div>
        </div>
      ))}
    </div>
  )
}

// ---- WorkoutLog (main) ----

export function WorkoutLog() {
  const [activeSession, setActiveSession] = useState<WorkoutSession | null>(null)
  const [loggedSets, setLoggedSets] = useState<LoggedSet[]>([])
  const [selectedExercise, setSelectedExercise] = useState<{ id: string; name: string } | null>(null)
  const [history, setHistory] = useState<HistoryEntry[]>([])
  const [historyLoading, setHistoryLoading] = useState(true)
  const [historyError, setHistoryError] = useState<string | null>(null)
  const [pageError, setPageError] = useState<string | null>(null)
  const [starting, setStarting] = useState(false)
  const [finishing, setFinishing] = useState(false)
  const [showSaveModal, setShowSaveModal] = useState(false)
  const [savedProgram, setSavedProgram] = useState<string | null>(null)
  // After finishing, keep sets for the save-as-program modal.
  const [finishedGroups, setFinishedGroups] = useState<ExerciseGroup[]>([])

  // Load history on mount.
  useEffect(() => {
    getHistory()
      .then(setHistory)
      .catch(() => setHistoryError('Failed to load history.'))
      .finally(() => setHistoryLoading(false))
  }, [])

  // Check for open session on mount.
  useEffect(() => {
    listSessions().then((sessions) => {
      const open = sessions.find((s) => s.status === 'open')
      if (open) setActiveSession(open)
    }).catch(() => { /* silent — not critical */ })
  }, [])

  const groups = groupByExercise(loggedSets)

  async function handleStartWorkout() {
    setPageError(null)
    setStarting(true)
    try {
      const session = await startSession({})
      setActiveSession(session)
      setLoggedSets([])
      setSelectedExercise(null)
    } catch (err) {
      if (err instanceof ApiError && err.status === 409) {
        // Open session exists — find and resume it.
        try {
          const sessions = await listSessions()
          const open = sessions.find((s) => s.status === 'open')
          if (open) {
            setActiveSession(open)
            setLoggedSets([])
            setSelectedExercise(null)
          } else {
            setPageError('A session conflict was reported but no open session found. Refresh and try again.')
          }
        } catch {
          setPageError('An open session already exists. Refresh and try again.')
        }
      } else {
        setPageError(err instanceof ApiError ? err.message : 'Failed to start workout.')
      }
    } finally {
      setStarting(false)
    }
  }

  const handleSetLogged = useCallback((set: LoggedSet) => {
    setLoggedSets((prev) => [...prev, set])
  }, [])

  async function handleFinish() {
    if (!activeSession) return
    setFinishing(true)
    setPageError(null)
    try {
      await finishSession(activeSession.id)
      const currentGroups = groupByExercise(loggedSets)
      setFinishedGroups(currentGroups)
      setActiveSession(null)
      setSelectedExercise(null)
      // If any sets were logged, offer to save as program.
      if (currentGroups.length > 0) {
        setShowSaveModal(true)
      }
      // Refresh history.
      getHistory().then(setHistory).catch(() => { /* silent */ })
    } catch (err) {
      setPageError(err instanceof ApiError ? err.message : 'Failed to finish workout.')
    } finally {
      setFinishing(false)
    }
  }

  async function handleSaveAsProgram(name: string) {
    const exercises = finishedGroups.map((g, i) => {
      const lastSet = g.sets[g.sets.length - 1]
      return {
        exercise_id: g.exerciseId,
        order_index: i,
        target_sets: g.sets.length,
        target_reps: lastSet.reps,
        target_rir: lastSet.rir ?? null,
        target_weight: lastSet.weight,
      }
    })
    await createProgram({ name, exercises })
    setSavedProgram(name)
    setShowSaveModal(false)
    setFinishedGroups([])
    setLoggedSets([])
  }

  function handleSkipSave() {
    setShowSaveModal(false)
    setFinishedGroups([])
    setLoggedSets([])
  }

  return (
    <div className="page-shell" style={{ paddingTop: 'var(--sp-6)', paddingBottom: 'var(--sp-8)' }}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--sp-5)' }}>

        {/* Page header */}
        <div>
          <h1>Workout</h1>
          <p className="text-sm text-muted mt-3">Log sets as you go. Numbers update live.</p>
        </div>

        {pageError && (
          <div className="error-banner" role="alert">{pageError}</div>
        )}

        {savedProgram && (
          <div className="success-banner" role="status">
            Saved as "{savedProgram}". Find it in Programs.
          </div>
        )}

        {/* Two columns on desktop (≥1024px): active session/start on the
            left, history on the right. Single column, same DOM order, on
            mobile — the divider only renders in the stacked layout. */}
        <div className="workout-columns">
        {/* Active session */}
        {activeSession ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--sp-4)' }}>
            {/* Session header */}
            <div className="session-header">
              <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--sp-3)' }}>
                <div className="session-status-dot" aria-hidden="true" />
                <div>
                  <div style={{ fontFamily: 'var(--font-display)', fontWeight: 600, fontSize: '0.9375rem' }}>
                    Session in progress
                  </div>
                  <div style={{ fontSize: '0.8125rem', color: 'var(--color-text-3)' }}>
                    {new Date(activeSession.date).toLocaleString(undefined, {
                      weekday: 'short', month: 'short', day: 'numeric',
                      hour: '2-digit', minute: '2-digit',
                    })}
                  </div>
                </div>
              </div>
              <button
                className="btn btn-success btn-sm"
                onClick={handleFinish}
                disabled={finishing}
              >
                {finishing ? (
                  <><span className="spinner" style={{ width: 14, height: 14 }} /> Finishing…</>
                ) : (
                  'Finish workout'
                )}
              </button>
            </div>

            {/* Exercise picker or set form */}
            {!selectedExercise ? (
              <ExercisePicker onSelect={setSelectedExercise} />
            ) : (
              <SetForm
                exercise={selectedExercise}
                sessionId={activeSession.id}
                onLogged={handleSetLogged}
                onClear={() => setSelectedExercise(null)}
              />
            )}

            {/* Scoreboard (signature element) */}
            {groups.length > 0 && (
              <div>
                <h3 style={{ marginBottom: 'var(--sp-3)' }}>This session</h3>
                <Scoreboard groups={groups} />
                {!selectedExercise && (
                  <button
                    type="button"
                    className="btn btn-ghost btn-sm mt-4"
                    onClick={() => { /* picker already shown above */ }}
                    style={{ display: 'none' }}
                    aria-hidden="true"
                  />
                )}
              </div>
            )}

            {groups.length === 0 && (
              <div className="empty-state" style={{ padding: 'var(--sp-8) var(--sp-4)' }}>
                <div className="empty-state-icon" aria-hidden="true">
                  <Dumbbell size={40} strokeWidth={1.5} />
                </div>
                <h3>Pick an exercise to start</h3>
                <p>Search above, then log your first set.</p>
              </div>
            )}
          </div>
        ) : (
          /* No active session */
          <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--sp-4)' }}>
            <button
              className="btn btn-primary"
              onClick={handleStartWorkout}
              disabled={starting}
              style={{ alignSelf: 'flex-start' }}
            >
              {starting ? (
                <><span className="spinner" style={{ width: 16, height: 16 }} /> Starting…</>
              ) : (
                'Start workout'
              )}
            </button>
          </div>
        )}

        <hr className="divider" />

        {/* History */}
        <div className="history-section" style={{ marginTop: 0 }}>
          <h2 style={{ marginBottom: 'var(--sp-4)' }}>Recent history</h2>
          {historyLoading ? (
            <div className="loading-center">
              <span className="spinner" />
              <span>Loading history…</span>
            </div>
          ) : historyError ? (
            <div className="error-banner">{historyError}</div>
          ) : (
            <History entries={history} />
          )}
        </div>
        </div>
      </div>

      {/* Save-as-program modal */}
      {showSaveModal && (
        <SaveAsProgramModal
          groups={finishedGroups}
          onSave={handleSaveAsProgram}
          onSkip={handleSkipSave}
        />
      )}
    </div>
  )
}
