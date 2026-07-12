import { useState, type FormEvent } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import { ApiError } from '../api/client'

type ExperienceLevel = 'beginner' | 'intermediate' | 'advanced'
type PrimaryGoal = 'hypertrophy' | 'strength' | 'fat_loss' | 'general'

export function Register() {
  const { register } = useAuth()
  const navigate = useNavigate()

  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [displayName, setDisplayName] = useState('')
  const [experienceLevel, setExperienceLevel] = useState<ExperienceLevel>('beginner')
  const [primaryGoal, setPrimaryGoal] = useState<PrimaryGoal>('strength')
  const [injuryNotes, setInjuryNotes] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    setError(null)

    if (password.length < 8) {
      setError('Password must be at least 8 characters.')
      return
    }

    setSubmitting(true)
    try {
      await register({
        email,
        password,
        display_name: displayName,
        experience_level: experienceLevel,
        primary_goal: primaryGoal,
        injury_notes: injuryNotes.trim() || undefined,
      })
      navigate('/workout', { replace: true })
    } catch (err) {
      if (err instanceof ApiError) {
        // 409 = email already registered
        setError(err.message)
      } else {
        setError('Something went wrong. Try again.')
      }
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="auth-page">
      <div className="auth-card">
        <div className="auth-header">
          <div className="auth-brand">Hey<span className="auth-brand-accent">Coach</span></div>
          <h1>Create account</h1>
          <p className="text-sm text-muted">Get set up in under a minute.</p>
        </div>

        <form className="auth-form" onSubmit={handleSubmit} noValidate>
          <div className="field">
            <label htmlFor="display-name">Name</label>
            <input
              id="display-name"
              type="text"
              autoComplete="name"
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
              placeholder="How you want to be called"
              required
            />
          </div>

          <div className="field">
            <label htmlFor="reg-email">Email</label>
            <input
              id="reg-email"
              type="email"
              autoComplete="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
              required
            />
          </div>

          <div className="field">
            <label htmlFor="reg-password">Password</label>
            <input
              id="reg-password"
              type="password"
              autoComplete="new-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="At least 8 characters"
              required
            />
          </div>

          <div className="field">
            <label htmlFor="experience">Experience level</label>
            <select
              id="experience"
              value={experienceLevel}
              onChange={(e) => setExperienceLevel(e.target.value as ExperienceLevel)}
            >
              <option value="beginner">Beginner</option>
              <option value="intermediate">Intermediate</option>
              <option value="advanced">Advanced</option>
            </select>
          </div>

          <div className="field">
            <label htmlFor="goal">Primary goal</label>
            <select
              id="goal"
              value={primaryGoal}
              onChange={(e) => setPrimaryGoal(e.target.value as PrimaryGoal)}
            >
              <option value="strength">Strength</option>
              <option value="hypertrophy">Hypertrophy</option>
              <option value="fat_loss">Fat loss</option>
              <option value="general">General fitness</option>
            </select>
          </div>

          <div className="field">
            <label htmlFor="injury-notes">Injury notes <span style={{ color: 'var(--color-text-3)', fontWeight: 400 }}>(optional)</span></label>
            <textarea
              id="injury-notes"
              rows={2}
              value={injuryNotes}
              onChange={(e) => setInjuryNotes(e.target.value)}
              placeholder="Any injuries the coach should know about"
            />
          </div>

          {error && (
            <div className="error-banner" role="alert">
              {error}
            </div>
          )}

          <button
            type="submit"
            className="btn btn-primary btn-full"
            disabled={submitting}
          >
            {submitting ? (
              <>
                <span className="spinner" style={{ width: 16, height: 16 }} />
                Creating account…
              </>
            ) : (
              'Create account'
            )}
          </button>
        </form>

        <div className="auth-footer">
          Already have an account?{' '}
          <Link to="/login">Log in</Link>
        </div>
      </div>
    </div>
  )
}
