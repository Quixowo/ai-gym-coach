/**
 * ProgramBuilder — list, create, and edit programs.
 *
 * Key requirement: when PUT /programs/{id} returns 422 with a load-jump-cap
 * violation, the error detail (naming the exercise, prior, and requested
 * weights) is displayed prominently inline. The client never pre-blocks this;
 * the server is the enforcer and the UI just reports it.
 */

import { useState, useEffect, type FormEvent } from 'react'
import { ClipboardList, ChevronRight, ArrowLeft } from 'lucide-react'
import {
  listPrograms,
  getProgram,
  createProgram,
  updateProgram,
  type ProgramSummary,
  type ProgramDetail,
  type ProgramExercise,
  type ProgramExerciseInput,
} from '../api/programs'
import { listExercises, type Exercise } from '../api/exercises'
import { ApiError } from '../api/client'

// ---- ProgramList ----

interface ProgramListProps {
  programs: ProgramSummary[]
  loading: boolean
  error: string | null
  onSelect: (id: string) => void
  onNew: () => void
}

function ProgramList({ programs, loading, error, onSelect, onNew }: ProgramListProps) {
  if (loading) {
    return (
      <div className="loading-center">
        <span className="spinner" />
        <span>Loading programs…</span>
      </div>
    )
  }
  if (error) return <div className="error-banner" role="alert">{error}</div>

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h1>Programs</h1>
        <button className="btn btn-primary btn-sm" onClick={onNew}>
          New program
        </button>
      </div>

      {programs.length === 0 ? (
        <div className="empty-state">
          <div className="empty-state-icon" aria-hidden="true">
            <ClipboardList size={40} strokeWidth={1.5} />
          </div>
          <h3>No programs yet</h3>
          <p>Create a program to plan your next training block, or save one after a workout.</p>
          <button className="btn btn-primary" onClick={onNew}>
            Create your first program
          </button>
        </div>
      ) : (
        <div className="program-list-grid">
          {programs.map((p) => (
            <button
              key={p.id}
              className="program-card"
              onClick={() => onSelect(p.id)}
            >
              <div>
                <div style={{ fontFamily: 'var(--font-display)', fontWeight: 600, fontSize: '1rem', color: 'var(--color-text)' }}>
                  {p.name}
                </div>
                <div style={{ fontSize: '0.8125rem', color: 'var(--color-text-3)', marginTop: 2 }}>
                  Created {new Date(p.created_at).toLocaleDateString()}
                </div>
              </div>
              <ChevronRight size={20} strokeWidth={1.75} color="var(--color-text-3)" aria-hidden="true" />
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

// ---- ExerciseRow (within the editor) ----

interface ExerciseRowProps {
  ex: ProgramExercise
  exercises: Exercise[]
  onChange: (updated: ProgramExerciseInput) => void
  onRemove: () => void
  index: number
}

function ExerciseRow({ ex, onChange, onRemove, index }: ExerciseRowProps) {
  function field(
    label: string,
    key: keyof ProgramExerciseInput,
    value: number | null | undefined,
    placeholder: string,
    step = 1,
    min = 0,
  ) {
    return (
      <div className="field" style={{ minWidth: 72 }}>
        <label htmlFor={`ex-${ex.id}-${key}`}>{label}</label>
        <input
          id={`ex-${ex.id}-${key}`}
          type="number"
          inputMode="decimal"
          step={step}
          min={min}
          value={value ?? ''}
          placeholder={placeholder}
          onChange={(e) => {
            const parsed = e.target.value === '' ? null : parseFloat(e.target.value)
            onChange({
              exercise_id: ex.exercise_id,
              order_index: index,
              target_sets: key === 'target_sets' ? (parsed === null ? null : Math.round(parsed)) : ex.target_sets,
              target_reps: key === 'target_reps' ? (parsed === null ? null : Math.round(parsed)) : ex.target_reps,
              target_rir: key === 'target_rir' ? parsed : ex.target_rir,
              target_weight: key === 'target_weight' ? parsed : ex.target_weight,
            })
          }}
        />
      </div>
    )
  }

  return (
    <div style={{ border: '1px solid var(--color-border)', borderRadius: 'var(--radius-md)', padding: 'var(--sp-3) var(--sp-4)', display: 'flex', flexDirection: 'column', gap: 'var(--sp-3)', background: 'var(--color-surface)' }}>
      <div className="flex items-center justify-between">
        <div>
          <span style={{ fontFamily: 'var(--font-display)', fontWeight: 600, fontSize: '0.9375rem' }}>
            {ex.exercise_name}
          </span>
          <span style={{ marginLeft: 'var(--sp-2)', fontFamily: 'var(--font-display)', fontSize: '0.75rem', color: 'var(--color-text-3)', fontWeight: 600, letterSpacing: '0.04em', textTransform: 'uppercase' }}>
            #{index + 1}
          </span>
        </div>
        <button
          type="button"
          className="btn btn-ghost btn-sm"
          onClick={onRemove}
          aria-label={`Remove ${ex.exercise_name}`}
          style={{ color: 'var(--color-danger)' }}
        >
          Remove
        </button>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 'var(--sp-3)' }}>
        {field('Sets', 'target_sets', ex.target_sets, '3', 1, 1)}
        {field('Reps', 'target_reps', ex.target_reps, '8', 1, 1)}
        {field('RIR', 'target_rir', ex.target_rir, '2', 0.5, 0)}
        {field('Weight (lbs)', 'target_weight', ex.target_weight, '135', 2.5, 0)}
      </div>
    </div>
  )
}

// ---- ProgramEditor ----

interface ProgramEditorProps {
  programId: string | null   // null = new program
  allExercises: Exercise[]
  onBack: () => void
  onSaved: () => void
}

function ProgramEditor({ programId, allExercises, onBack, onSaved }: ProgramEditorProps) {
  const isNew = programId === null
  const [name, setName] = useState('')
  const [exercises, setExercises] = useState<ProgramExercise[]>([])
  const [exerciseInputs, setExerciseInputs] = useState<ProgramExerciseInput[]>([])
  const [loading, setLoading] = useState(!isNew)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [saveError, setSaveError] = useState<string | null>(null)
  // Separate state for 422 cap violations — shown prominently.
  const [capError, setCapError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)

  // Exercise add UI.
  const [addQuery, setAddQuery] = useState('')
  const [addOpen, setAddOpen] = useState(false)

  useEffect(() => {
    if (isNew) return
    setLoading(true)
    getProgram(programId!)
      .then((p: ProgramDetail) => {
        setName(p.name)
        setExercises(p.exercises)
        setExerciseInputs(p.exercises.map((ex, i) => ({
          exercise_id: ex.exercise_id,
          order_index: i,
          target_sets: ex.target_sets,
          target_reps: ex.target_reps,
          target_rir: ex.target_rir,
          target_weight: ex.target_weight,
        })))
      })
      .catch(() => setLoadError('Failed to load program.'))
      .finally(() => setLoading(false))
  }, [isNew, programId])

  function handleExerciseChange(index: number, updated: ProgramExerciseInput) {
    setExerciseInputs((prev) => prev.map((ex, i) => i === index ? updated : ex))
  }

  function handleRemoveExercise(index: number) {
    setExercises((prev) => prev.filter((_, i) => i !== index))
    setExerciseInputs((prev) =>
      prev.filter((_, i) => i !== index).map((ex, i) => ({ ...ex, order_index: i }))
    )
  }

  function handleAddExercise(ex: Exercise) {
    const newPE: ProgramExercise = {
      id: `new-${ex.id}`,
      exercise_id: ex.id,
      exercise_name: ex.name,
      order_index: exercises.length,
      target_sets: null,
      target_reps: null,
      target_rir: null,
      target_weight: null,
    }
    setExercises((prev) => [...prev, newPE])
    setExerciseInputs((prev) => [...prev, {
      exercise_id: ex.id,
      order_index: prev.length,
      target_sets: null,
      target_reps: null,
      target_rir: null,
      target_weight: null,
    }])
    setAddQuery('')
    setAddOpen(false)
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    setSaveError(null)
    setCapError(null)

    if (!name.trim()) {
      setSaveError('Enter a name for this program.')
      return
    }
    setSaving(true)
    try {
      if (isNew) {
        await createProgram({ name: name.trim(), exercises: exerciseInputs })
      } else {
        await updateProgram(programId!, { name: name.trim(), exercises: exerciseInputs })
      }
      onSaved()
    } catch (err) {
      if (err instanceof ApiError && err.status === 422) {
        // Could be a load-jump cap violation — show prominently.
        setCapError(err.message)
      } else {
        setSaveError(err instanceof ApiError ? err.message : 'Failed to save. Try again.')
      }
    } finally {
      setSaving(false)
    }
  }

  if (loading) {
    return (
      <div className="loading-center">
        <span className="spinner" />
        <span>Loading program…</span>
      </div>
    )
  }
  if (loadError) return <div className="error-banner" role="alert">{loadError}</div>

  const filteredExercises = allExercises
    .filter((ex) => ex.name.toLowerCase().includes(addQuery.toLowerCase()))
    .slice(0, 15)

  return (
    <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 'var(--sp-5)' }}>
      {/* Header */}
      <div className="flex items-center gap-3">
        <button
          type="button"
          className="btn btn-ghost btn-sm"
          onClick={onBack}
          aria-label="Back to programs"
        >
          <ArrowLeft size={16} strokeWidth={1.75} aria-hidden="true" />
          Back
        </button>
        <h1 style={{ fontSize: '1.5rem' }}>{isNew ? 'New program' : 'Edit program'}</h1>
      </div>

      {/* Name */}
      <div className="field">
        <label htmlFor="prog-name">Program name</label>
        <input
          id="prog-name"
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="e.g. Push A"
          required
        />
      </div>

      {/* Exercises */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--sp-3)' }}>
        <h3>Exercises</h3>
        {exercises.length === 0 && (
          <div style={{ color: 'var(--color-text-2)', fontSize: '0.9rem', padding: 'var(--sp-3) 0' }}>
            Add at least one exercise below.
          </div>
        )}
        {exercises.map((ex, i) => (
          <ExerciseRow
            key={`${ex.exercise_id}-${i}`}
            ex={ex}
            exercises={allExercises}
            index={i}
            onChange={(updated) => handleExerciseChange(i, updated)}
            onRemove={() => handleRemoveExercise(i)}
          />
        ))}

        {/* Add exercise */}
        <div style={{ position: 'relative' }}>
          <div className="field">
            <label htmlFor="add-ex">Add exercise</label>
            <input
              id="add-ex"
              type="search"
              value={addQuery}
              onChange={(e) => { setAddQuery(e.target.value); setAddOpen(true) }}
              onFocus={() => setAddOpen(true)}
              placeholder="Search and add…"
              autoComplete="off"
            />
          </div>
          {addOpen && filteredExercises.length > 0 && (
            <div className="exercise-results" style={{ zIndex: 30 }}>
              {filteredExercises.map((ex) => (
                <button
                  key={ex.id}
                  type="button"
                  className="exercise-result-item"
                  onClick={() => handleAddExercise(ex)}
                >
                  <span className="exercise-result-name">{ex.name}</span>
                  <span className="exercise-result-meta">{ex.primary_muscle_group} · {ex.equipment}</span>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* 422 load-jump cap error — shown prominently per acceptance gate */}
      {capError && (
        <div
          className="error-banner"
          role="alert"
          style={{
            borderColor: 'rgba(var(--color-danger-rgb), 0.7)',
            background: 'rgba(var(--color-danger-rgb), 0.15)',
            padding: 'var(--sp-4)',
          }}
        >
          <strong style={{ display: 'block', marginBottom: 'var(--sp-1)' }}>
            Weight increase rejected
          </strong>
          {capError}
          <div style={{ marginTop: 'var(--sp-2)', fontSize: '0.8125rem', color: 'var(--color-text-2)' }}>
            Increases are capped at 10% per update. Lower the target weight and try again.
          </div>
        </div>
      )}

      {saveError && (
        <div className="error-banner" role="alert">{saveError}</div>
      )}

      <button
        type="submit"
        className="btn btn-primary"
        disabled={saving}
        style={{ alignSelf: 'flex-start' }}
      >
        {saving ? (
          <><span className="spinner" style={{ width: 16, height: 16 }} /> Saving…</>
        ) : (
          isNew ? 'Create program' : 'Save changes'
        )}
      </button>
    </form>
  )
}

// ---- ProgramBuilder (main) ----

export function ProgramBuilder() {
  const [programs, setPrograms] = useState<ProgramSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedId, setSelectedId] = useState<string | null | 'new'>(null)
  const [allExercises, setAllExercises] = useState<Exercise[]>([])

  function loadPrograms() {
    setLoading(true)
    setError(null)
    listPrograms()
      .then(setPrograms)
      .catch(() => setError('Failed to load programs.'))
      .finally(() => setLoading(false))
  }

  useEffect(() => {
    loadPrograms()
    listExercises().then(setAllExercises).catch(() => { /* silent */ })
  }, [])

  function handleSaved() {
    setSelectedId(null)
    loadPrograms()
  }

  if (selectedId !== null) {
    return (
      <div className="page-shell" style={{ paddingTop: 'var(--sp-6)', paddingBottom: 'var(--sp-8)' }}>
        <ProgramEditor
          programId={selectedId === 'new' ? null : selectedId}
          allExercises={allExercises}
          onBack={() => setSelectedId(null)}
          onSaved={handleSaved}
        />
      </div>
    )
  }

  return (
    <div className="page-shell" style={{ paddingTop: 'var(--sp-6)', paddingBottom: 'var(--sp-8)' }}>
      <ProgramList
        programs={programs}
        loading={loading}
        error={error}
        onSelect={setSelectedId}
        onNew={() => setSelectedId('new')}
      />
    </div>
  )
}
