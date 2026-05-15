import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render } from '@testing-library/react'
import { GraphCanvas } from '../components/graph/GraphCanvas'

const networkState = vi.hoisted(() => ({
  instances: [] as Array<{
    data: any
    handlers: Record<string, Function>
    onceHandlers: Record<string, Function>
    destroy: ReturnType<typeof vi.fn>
    fit: ReturnType<typeof vi.fn>
    focus: ReturnType<typeof vi.fn>
    on: ReturnType<typeof vi.fn>
    once: ReturnType<typeof vi.fn>
    getNodeAt: ReturnType<typeof vi.fn>
  }>
}))

vi.mock('vis-network/standalone', () => {
  class MockNetwork {
    data: any
    handlers: Record<string, Function> = {}
    onceHandlers: Record<string, Function> = {}
    destroy = vi.fn()
    fit = vi.fn()
    focus = vi.fn()
    getNodeAt = vi.fn()
    on = vi.fn((event: string, handler: Function) => {
      this.handlers[event] = handler
    })
    once = vi.fn((event: string, handler: Function) => {
      this.onceHandlers[event] = handler
    })

    constructor(_container: HTMLElement, data: any) {
      this.data = data
      networkState.instances.push(this)
    }
  }

  return { Network: MockNetwork }
})

vi.mock('vis-data', () => {
  class MockDataSet {
    items: any[]

    constructor(items: any[]) {
      this.items = items.map(item => ({ ...item }))
    }

    get(id?: string) {
      if (id === undefined) {
        return this.items.map(item => ({ ...item }))
      }
      return this.items.find(item => item.id === id)
    }

    update(updates: any | any[]) {
      const updateList = Array.isArray(updates) ? updates : [updates]
      updateList.forEach(update => {
        const index = this.items.findIndex(item => item.id === update.id)
        if (index >= 0) {
          this.items[index] = { ...this.items[index], ...update }
        } else {
          this.items.push({ ...update })
        }
      })
    }
  }

  return { DataSet: MockDataSet }
})

afterEach(() => {
  cleanup()
  networkState.instances.length = 0
})

describe('GraphCanvas hover color rendering', () => {
  const nodes = [
    { id: 'disease-1', label: 'Disease', properties: { name: '疾病A' }, degree: 2 },
    { id: 'symptom-1', label: 'Symptom', properties: { name: '症状B' }, degree: 2 },
    { id: 'drug-1', label: 'Drug', properties: { name: '药物C' }, degree: 1 },
  ]

  const edges = [
    { id: 'edge-1', from: 'disease-1', to: 'symptom-1', type: 'HAS_SYMPTOM' },
    { id: 'edge-2', from: 'symptom-1', to: 'drug-1', type: 'TREATED_BY' },
  ]

  const getRenderedNode = (network: (typeof networkState.instances)[number], nodeId: string) => {
    return network.data.nodes.get().find((node: any) => node.id === nodeId)
  }

  it('restores category colors for newly connected nodes across repeated hovers', () => {
    render(<GraphCanvas nodes={nodes} edges={edges} />)

    const network = networkState.instances[0]
    network.handlers.hoverNode({ node: 'disease-1' })

    expect(getRenderedNode(network, 'drug-1').color.background).toBe('rgba(200, 200, 200, 0.3)')

    network.handlers.hoverNode({ node: 'symptom-1' })

    expect(getRenderedNode(network, 'disease-1').color.background).toBe('#EF4444')
    expect(getRenderedNode(network, 'symptom-1').color.background).toBe('#F97316')
    expect(getRenderedNode(network, 'drug-1').color.background).toBe('#22C55E')
  })

  it('restores category colors when the pointer leaves a node or the canvas', () => {
    const { container } = render(<GraphCanvas nodes={nodes} edges={edges} />)

    const network = networkState.instances[0]
    network.handlers.hoverNode({ node: 'disease-1' })
    expect(getRenderedNode(network, 'drug-1').color.background).toBe('rgba(200, 200, 200, 0.3)')

    network.handlers.blurNode()
    expect(getRenderedNode(network, 'drug-1').color.background).toBe('#22C55E')

    network.handlers.hoverNode({ node: 'disease-1' })
    fireEvent.mouseLeave(container.firstElementChild as Element)

    expect(getRenderedNode(network, 'disease-1').color.background).toBe('#EF4444')
    expect(getRenderedNode(network, 'symptom-1').color.background).toBe('#F97316')
    expect(getRenderedNode(network, 'drug-1').color.background).toBe('#22C55E')
  })

  it('initializes category colors after switching to another graph view', () => {
    const { rerender } = render(<GraphCanvas nodes={nodes} edges={edges} />)

    const firstNetwork = networkState.instances[0]
    firstNetwork.handlers.hoverNode({ node: 'disease-1' })

    rerender(
      <GraphCanvas
        nodes={[
          { id: 'exam-1', label: 'Examination', properties: { name: '检查D' }, degree: 1 },
          { id: 'department-1', label: 'Department', properties: { name: '科室E' }, degree: 1 },
        ]}
        edges={[
          { id: 'edge-3', from: 'exam-1', to: 'department-1', type: 'BELONGS_TO' },
        ]}
      />
    )

    const secondNetwork = networkState.instances[1]
    expect(firstNetwork.destroy).toHaveBeenCalled()
    expect(getRenderedNode(secondNetwork, 'exam-1').color.background).toBe('#3B82F6')
    expect(getRenderedNode(secondNetwork, 'department-1').color.background).toBe('#EC4899')
  })
})
