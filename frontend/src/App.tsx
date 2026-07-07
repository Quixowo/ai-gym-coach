import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { AuthProvider, useAuth } from './context/AuthContext'
import { Login } from './pages/Login'
import { Register } from './pages/Register'
import { WorkoutLog } from './pages/WorkoutLog'
import { ProgramBuilder } from './pages/ProgramBuilder'
import { Chat } from './pages/Chat'
import { Nav } from './components/Nav'
import type { ReactNode } from 'react'

function RequireAuth({ children }: { children: ReactNode }) {
  const { status } = useAuth()
  if (status === 'loading') {
    return (
      <div className="loading-center">
        <div className="spinner" />
        <span>Loading…</span>
      </div>
    )
  }
  if (status === 'unauthenticated') {
    return <Navigate to="/login" replace />
  }
  return <>{children}</>
}

function RequireGuest({ children }: { children: ReactNode }) {
  const { status } = useAuth()
  if (status === 'loading') {
    return (
      <div className="loading-center">
        <div className="spinner" />
        <span>Loading…</span>
      </div>
    )
  }
  if (status === 'authenticated') {
    return <Navigate to="/workout" replace />
  }
  return <>{children}</>
}

function AppRoutes() {
  const { status } = useAuth()

  return (
    <>
      {status === 'authenticated' && <Nav />}
      <Routes>
        <Route path="/login" element={<RequireGuest><Login /></RequireGuest>} />
        <Route path="/register" element={<RequireGuest><Register /></RequireGuest>} />
        <Route
          path="/workout"
          element={<RequireAuth><WorkoutLog /></RequireAuth>}
        />
        <Route
          path="/programs"
          element={<RequireAuth><ProgramBuilder /></RequireAuth>}
        />
        <Route
          path="/chat"
          element={<RequireAuth><Chat /></RequireAuth>}
        />
        <Route path="/" element={<Navigate to="/workout" replace />} />
        <Route path="*" element={<Navigate to="/workout" replace />} />
      </Routes>
    </>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <AppRoutes />
      </AuthProvider>
    </BrowserRouter>
  )
}
