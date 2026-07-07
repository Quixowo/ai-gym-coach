import { NavLink } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import { useState } from 'react'

export function Nav() {
  const { user, logout } = useAuth()
  const [loggingOut, setLoggingOut] = useState(false)

  async function handleLogout() {
    setLoggingOut(true)
    try {
      await logout()
    } finally {
      setLoggingOut(false)
    }
  }

  return (
    <nav className="app-nav" aria-label="Main navigation">
      <div className="app-nav-inner">
        <div className="nav-brand">
          <span>■</span> Gym Coach
        </div>

        <ul className="nav-links" role="list">
          <li>
            <NavLink
              to="/workout"
              className={({ isActive }) => (isActive ? 'active' : '')}
            >
              Workout
            </NavLink>
          </li>
          <li>
            <NavLink
              to="/programs"
              className={({ isActive }) => (isActive ? 'active' : '')}
            >
              Programs
            </NavLink>
          </li>
          <li>
            <NavLink
              to="/chat"
              className={({ isActive }) => (isActive ? 'active' : '')}
            >
              Coach
            </NavLink>
          </li>
        </ul>

        <div className="nav-user">
          {user && (
            <span className="nav-user-name" title={user.email}>
              {user.display_name}
            </span>
          )}
          <button
            className="btn btn-ghost btn-sm"
            onClick={handleLogout}
            disabled={loggingOut}
          >
            {loggingOut ? 'Logging out…' : 'Log out'}
          </button>
        </div>
      </div>
    </nav>
  )
}
