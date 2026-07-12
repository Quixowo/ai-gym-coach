import { NavLink } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import { useEffect, useRef, useState } from 'react'
import { Menu, X } from 'lucide-react'

export function Nav() {
  const { user, logout } = useAuth()
  const [loggingOut, setLoggingOut] = useState(false)
  const [menuOpen, setMenuOpen] = useState(false)
  const navRef = useRef<HTMLElement>(null)

  // Close the mobile menu on outside click — same pattern as ExercisePicker.
  useEffect(() => {
    function onClick(e: MouseEvent) {
      if (navRef.current && !navRef.current.contains(e.target as Node)) {
        setMenuOpen(false)
      }
    }
    document.addEventListener('mousedown', onClick)
    return () => document.removeEventListener('mousedown', onClick)
  }, [])

  async function handleLogout() {
    setLoggingOut(true)
    try {
      await logout()
    } finally {
      setLoggingOut(false)
    }
  }

  const linkClass = ({ isActive }: { isActive: boolean }) => (isActive ? 'active' : '')

  return (
    <nav className="app-nav" aria-label="Main navigation" ref={navRef}>
      <div className="app-nav-inner">
        <div className="nav-brand">
          Hey<span className="nav-brand-accent">Coach</span>
        </div>

        <button
          type="button"
          className="nav-toggle"
          onClick={() => setMenuOpen((o) => !o)}
          aria-expanded={menuOpen}
          aria-controls="nav-collapsible"
          aria-label={menuOpen ? 'Close menu' : 'Open menu'}
        >
          {menuOpen ? <X size={20} strokeWidth={1.75} /> : <Menu size={20} strokeWidth={1.75} />}
        </button>

        <div
          id="nav-collapsible"
          className={`nav-collapsible${menuOpen ? ' nav-collapsible--open' : ''}`}
        >
          <ul className="nav-links" role="list">
            <li>
              <NavLink to="/workout" className={linkClass} onClick={() => setMenuOpen(false)}>
                Workout
              </NavLink>
            </li>
            <li>
              <NavLink to="/programs" className={linkClass} onClick={() => setMenuOpen(false)}>
                Programs
              </NavLink>
            </li>
            <li>
              <NavLink to="/chat" className={linkClass} onClick={() => setMenuOpen(false)}>
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
      </div>
    </nav>
  )
}
