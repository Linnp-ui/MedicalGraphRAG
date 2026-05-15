import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { GraphView } from '../components/GraphView'
import * as api from '../lib/api'

vi.mock('../lib/api', () => ({
  api: {
    getGraphData: vi.fn(),
    searchNodes: vi.fn(),
    getNodeDetail: vi.fn(),
    getNodeNeighbors: vi.fn(),
    getQueryResultGraph: vi.fn(),
  },
}))

vi.mock('../components/graph/GraphCanvas', () => ({
  GraphCanvas: vi.fn(() => null)
}))

describe('GraphView', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('初始加载', () => {
    it('应该显示加载状态', () => {
      vi.mocked(api.api.getGraphData).mockImplementation(() => new Promise(() => {}))
      
      render(<GraphView />)
      
      expect(screen.getByText('加载图谱数据...')).toBeInTheDocument()
    })

    it('应该成功加载图谱数据', async () => {
      const mockData = {
        nodes: [
          { id: '1', label: 'Person', properties: { name: '张三' }, degree: 5 }
        ],
        edges: [],
        stats: {
          total_nodes: 1,
          total_edges: 0,
          node_labels: ['Person'],
          relationship_types: []
        }
      }

      vi.mocked(api.api.getGraphData).mockResolvedValueOnce(mockData)
      
      render(<GraphView />)
      
      await waitFor(() => {
        expect(screen.getByText('医疗知识图谱')).toBeInTheDocument()
      })
    })

    it('应该显示错误信息当加载失败时', async () => {
      vi.mocked(api.api.getGraphData).mockRejectedValueOnce(new Error('加载失败'))
      
      render(<GraphView />)
      
      await waitFor(() => {
        expect(screen.getByText('加载图谱数据失败')).toBeInTheDocument()
      })
    })
  })

  describe('搜索功能', () => {
    it('应该显示搜索框', async () => {
      const mockData = {
        nodes: [],
        edges: [],
        stats: { total_nodes: 0, total_edges: 0, node_labels: [], relationship_types: [] }
      }

      vi.mocked(api.api.getGraphData).mockResolvedValueOnce(mockData)
      
      render(<GraphView />)
      
      await waitFor(() => {
        expect(screen.getByPlaceholderText('搜索节点...')).toBeInTheDocument()
      })
    })

    it('应该调用搜索API当输入搜索关键词时', async () => {
      const mockData = {
        nodes: [],
        edges: [],
        stats: { total_nodes: 0, total_edges: 0, node_labels: [], relationship_types: [] }
      }

      const mockSearchResults = {
        results: [
          { id: '1', label: 'Person', properties: { name: '张三' } }
        ],
        total: 1
      }

      vi.mocked(api.api.getGraphData).mockResolvedValueOnce(mockData)
      vi.mocked(api.api.searchNodes).mockResolvedValueOnce(mockSearchResults)
      
      render(<GraphView />)
      
      await waitFor(() => {
        expect(screen.getByPlaceholderText('搜索节点...')).toBeInTheDocument()
      })

      const searchInput = screen.getByPlaceholderText('搜索节点...')
      await userEvent.type(searchInput, '张三')

      await waitFor(() => {
        expect(api.api.searchNodes).toHaveBeenCalledWith('张三', { limit: 10 })
      }, { timeout: 1000 })
    })

    it('应该显示搜索建议', async () => {
      const mockData = {
        nodes: [],
        edges: [],
        stats: { total_nodes: 0, total_edges: 0, node_labels: [], relationship_types: [] }
      }

      const mockSearchResults = {
        results: [
          { id: '1', label: 'Person', properties: { name: '张三' } }
        ],
        total: 1
      }

      vi.mocked(api.api.getGraphData).mockResolvedValueOnce(mockData)
      vi.mocked(api.api.searchNodes).mockResolvedValueOnce(mockSearchResults)
      
      render(<GraphView />)
      
      await waitFor(() => {
        expect(screen.getByPlaceholderText('搜索节点...')).toBeInTheDocument()
      })

      const searchInput = screen.getByPlaceholderText('搜索节点...')
      await userEvent.type(searchInput, '张三')

      await waitFor(() => {
        expect(screen.getByText('张三')).toBeInTheDocument()
      }, { timeout: 1000 })
    })
  })

  describe('节点类型筛选', () => {
    it('应该显示节点类型按钮', async () => {
      const mockData = {
        nodes: [],
        edges: [],
        stats: {
          total_nodes: 0,
          total_edges: 0,
          node_labels: ['Person', 'Organization'],
          relationship_types: []
        }
      }

      vi.mocked(api.api.getGraphData).mockResolvedValueOnce(mockData)
      
      render(<GraphView />)
      
      await waitFor(() => {
        expect(screen.getByText('Person')).toBeInTheDocument()
        expect(screen.getByText('Organization')).toBeInTheDocument()
      })
    })

    it('应该调用API筛选节点类型当点击类型按钮时', async () => {
      const mockData = {
        nodes: [],
        edges: [],
        stats: {
          total_nodes: 0,
          total_edges: 0,
          node_labels: ['Person', 'Organization'],
          relationship_types: []
        }
      }

      vi.mocked(api.api.getGraphData)
        .mockResolvedValueOnce(mockData)
        .mockResolvedValueOnce({
          ...mockData,
          stats: { ...mockData.stats, node_labels: ['Person'] }
        })
      
      render(<GraphView />)
      
      await waitFor(() => {
        expect(screen.getByText('Person')).toBeInTheDocument()
      })

      const personButton = screen.getByText('Person')
      fireEvent.click(personButton)

      await waitFor(() => {
        expect(api.api.getGraphData).toHaveBeenCalledWith(
          expect.objectContaining({ node_label: 'Person' })
        )
      })
    })
  })

  describe('统计信息显示', () => {
    it('应该显示正确的统计信息', async () => {
      const mockData = {
        nodes: [
          { id: '1', label: 'Person', properties: {}, degree: 5 },
          { id: '2', label: 'Organization', properties: {}, degree: 3 }
        ],
        edges: [
          { id: 'e1', from: '1', to: '2', type: 'WORKS_FOR', properties: {} }
        ],
        stats: {
          total_nodes: 2,
          total_edges: 1,
          node_labels: ['Person', 'Organization'],
          relationship_types: ['WORKS_FOR']
        }
      }

      vi.mocked(api.api.getGraphData).mockResolvedValueOnce(mockData)
      
      render(<GraphView />)
      
      await waitFor(() => {
        expect(screen.getByText('当前节点数:')).toBeInTheDocument()
        expect(screen.getByText('总节点数:').nextElementSibling).toHaveTextContent('2')
        expect(screen.getByText('当前关系数:')).toBeInTheDocument()
        expect(screen.getByText('总关系数:').nextElementSibling).toHaveTextContent('1')
      }, { timeout: 3000 })
    })
  })

  describe('刷新功能', () => {
    it('应该重新加载数据当点击刷新按钮时', async () => {
      const user = userEvent.setup()
      const mockData = {
        nodes: [],
        edges: [],
        stats: { total_nodes: 0, total_edges: 0, node_labels: [], relationship_types: [] }
      }

      vi.mocked(api.api.getGraphData).mockResolvedValue(mockData)
      
      render(<GraphView />)
      
      await waitFor(() => {
        expect(screen.getByText('刷新')).toBeInTheDocument()
      })

      const refreshButton = screen.getByText('刷新')
      await user.click(refreshButton)

      await waitFor(() => {
        expect(api.api.getGraphData).toHaveBeenCalledTimes(2)
      })
    })
  })

  describe('错误处理', () => {
    it('应该显示错误信息当API调用失败时', async () => {
      vi.mocked(api.api.getGraphData).mockRejectedValueOnce(new Error('网络错误'))
      
      render(<GraphView />)
      
      await waitFor(() => {
        expect(screen.getByText('加载图谱数据失败')).toBeInTheDocument()
        expect(screen.getByText('网络错误')).toBeInTheDocument()
      })
    })

    it('应该允许重试当发生错误时', async () => {
      const user = userEvent.setup()
      const mockData = {
        nodes: [],
        edges: [],
        stats: { total_nodes: 0, total_edges: 0, node_labels: [], relationship_types: [] }
      }

      vi.mocked(api.api.getGraphData)
        .mockRejectedValueOnce(new Error('网络错误'))
        .mockResolvedValueOnce(mockData)
      
      render(<GraphView />)
      
      await waitFor(() => {
        expect(screen.getByText('加载图谱数据失败')).toBeInTheDocument()
      }, { timeout: 3000 })

      const refreshButton = screen.getByText('刷新')
      await user.click(refreshButton)

      await waitFor(() => {
        expect(screen.getByText('医疗知识图谱')).toBeInTheDocument()
      }, { timeout: 3000 })
    })
  })

  describe('边界条件', () => {
    it('应该处理空数据', async () => {
      const mockData = {
        nodes: [],
        edges: [],
        stats: { total_nodes: 0, total_edges: 0, node_labels: [], relationship_types: [] }
      }

      vi.mocked(api.api.getGraphData).mockResolvedValueOnce(mockData)
      
      render(<GraphView />)
      
      await waitFor(() => {
        expect(screen.getByText('暂无数据')).toBeInTheDocument()
      })
    })

    it('应该处理大量数据', async () => {
      const nodes = Array.from({ length: 1000 }, (_, i) => ({
        id: `node_${i}`,
        label: 'Person',
        properties: { name: `用户${i}` },
        degree: Math.floor(Math.random() * 10)
      }))

      const edges = Array.from({ length: 500 }, (_, i) => ({
        id: `edge_${i}`,
        from: `node_${i}`,
        to: `node_${(i + 1) % 1000}`,
        type: 'KNOWS',
        properties: {}
      }))

      const mockData = {
        nodes,
        edges,
        stats: {
          total_nodes: 1000,
          total_edges: 500,
          node_labels: ['Person'],
          relationship_types: ['KNOWS']
        }
      }

      vi.mocked(api.api.getGraphData).mockResolvedValueOnce(mockData)
      
      render(<GraphView />)
      
      await waitFor(() => {
        expect(screen.getByText('总节点数:').nextElementSibling).toHaveTextContent('1000')
        expect(screen.getByText('总关系数:').nextElementSibling).toHaveTextContent('500')
      }, { timeout: 5000 })
    })
  })
})
