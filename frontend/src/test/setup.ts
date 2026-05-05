import '@testing-library/jest-dom'
import { vi } from 'vitest'

global.ResizeObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
}))

global.IntersectionObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
}))

Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
})

vi.mock('vis-network', () => ({
  Network: vi.fn().mockImplementation(() => ({
    on: vi.fn(),
    off: vi.fn(),
    setData: vi.fn(),
    fit: vi.fn(),
    destroy: vi.fn(),
    getSelectedNodes: vi.fn(() => []),
    selectNodes: vi.fn(),
    unselectAll: vi.fn(),
  })),
  DataSet: vi.fn().mockImplementation(() => ({
    add: vi.fn(),
    update: vi.fn(),
    remove: vi.fn(),
    clear: vi.fn(),
    get: vi.fn(() => []),
    length: 0,
  })),
}))

vi.mock('vis-data', () => ({
  DataSet: vi.fn().mockImplementation(() => ({
    add: vi.fn(),
    update: vi.fn(),
    remove: vi.fn(),
    clear: vi.fn(),
    get: vi.fn(() => []),
    length: 0,
  })),
}))
