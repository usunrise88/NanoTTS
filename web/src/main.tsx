import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { createRouter, RouterProvider } from '@tanstack/react-router'
import { routeTree } from './routeTree.gen'
import './styles.css'

const router = createRouter({
  routeTree,
  // Hovering a link starts its work, so the common case never reaches the
  // threshold where a skeleton would be needed at all.
  defaultPreload: 'intent',
  defaultPreloadStaleTime: 10_000,
  // The bundle can be mounted at / or behind a proxy at /asr/.
  basepath: window.location.pathname.replace(/[^/]*$/, ''),
})

declare module '@tanstack/react-router' {
  interface Register {
    router: typeof router
  }
}

const root = document.getElementById('root')
if (!root) throw new Error('#root is missing')

createRoot(root).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>,
)
