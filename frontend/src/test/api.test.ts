import { describe, it, expect, vi, beforeEach } from 'vitest'

const mockFetch = vi.fn()
global.fetch = mockFetch

class TestApiClient {
  private baseUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    const headers = {
      'Content-Type': 'application/json',
      ...options?.headers,
    };

    const response = await fetch(url, { ...options, headers });
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
    }
    return response.json();
  }

  async getGraphData(params?: {
    node_label?: string;
    limit?: number;
    offset?: number;
  }): Promise<any> {
    const queryParams = new URLSearchParams();
    if (params?.node_label) queryParams.append('node_label', params.node_label);
    if (params?.limit) queryParams.append('limit', params.limit.toString());
    if (params?.offset) queryParams.append('offset', params.offset.toString());
    
    return this.request(`/graph/data?${queryParams}`);
  }

  async searchNodes(query: string, params?: {
    node_label?: string;
    limit?: number;
  }): Promise<any> {
    const queryParams = new URLSearchParams();
    queryParams.append('query', query);
    if (params?.node_label) queryParams.append('node_label', params.node_label);
    if (params?.limit) queryParams.append('limit', params.limit.toString());
    
    return this.request(`/graph/search?${queryParams}`);
  }

  async getNodeDetail(nodeId: string): Promise<any> {
    return this.request(`/graph/node/${nodeId}`);
  }

  async getNodeNeighbors(nodeId: string, params?: {
    depth?: number;
    relationship_type?: string;
  }): Promise<any> {
    const queryParams = new URLSearchParams();
    if (params?.depth) queryParams.append('depth', params.depth.toString());
    if (params?.relationship_type) queryParams.append('relationship_type', params.relationship_type);
    
    return this.request(`/graph/node/${nodeId}/neighbors?${queryParams}`);
  }

  async getQueryResultGraph(request: {
    query: string;
    node_ids: string[];
    max_depth?: number;
  }): Promise<any> {
    return this.request('/graph/query-result', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }
}

describe('APIClient', () => {
  let client: TestApiClient
  const baseUrl = 'http://localhost:8000/api/v1'

  beforeEach(() => {
    client = new TestApiClient(baseUrl)
    mockFetch.mockReset()
  })

  describe('getGraphData', () => {
    it('should fetch graph data without parameters', async () => {
      const mockData = {
        nodes: [{ id: '1', label: 'Person', properties: { name: '张三' } }],
        edges: [],
        stats: { total_nodes: 1, total_edges: 0, node_labels: ['Person'], relationship_types: [] }
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockData,
      } as Response)

      const result = await client.getGraphData()

      expect(mockFetch).toHaveBeenCalledWith(`${baseUrl}/graph/data?`, expect.any(Object))
      expect(result).toEqual(mockData)
    })

    it('should fetch graph data with node_label parameter', async () => {
      const mockData = {
        nodes: [{ id: '1', label: 'Person', properties: { name: '张三' } }],
        edges: [],
        stats: { total_nodes: 1, total_edges: 0, node_labels: ['Person'], relationship_types: [] }
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockData,
      } as Response)

      const result = await client.getGraphData({ node_label: 'Person' })

      expect(mockFetch).toHaveBeenCalledWith(expect.stringContaining('node_label=Person'), expect.any(Object))
      expect(result).toEqual(mockData)
    })

    it('should fetch graph data with limit parameter', async () => {
      const mockData = {
        nodes: [],
        edges: [],
        stats: { total_nodes: 0, total_edges: 0, node_labels: [], relationship_types: [] }
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockData,
      } as Response)

      await client.getGraphData({ limit: 100 })

      expect(mockFetch).toHaveBeenCalledWith(expect.stringContaining('limit=100'), expect.any(Object))
    })

    it('should throw error when request fails', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        json: async () => ({ detail: 'Failed to fetch graph data' }),
      } as Response)

      await expect(client.getGraphData()).rejects.toThrow('Failed to fetch graph data')
    })
  })

  describe('searchNodes', () => {
    it('should search nodes with query', async () => {
      const mockData = {
        results: [{ id: '1', label: 'Person', properties: { name: '张三' } }],
        total: 1
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockData,
      } as Response)

      const result = await client.searchNodes('张三')

      expect(mockFetch).toHaveBeenCalledWith(expect.stringContaining('query=%E5%BC%A0%E4%B8%89'), expect.any(Object))
      expect(result).toEqual(mockData)
    })

    it('should search nodes with limit', async () => {
      const mockData = {
        results: [],
        total: 0
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockData,
      } as Response)

      await client.searchNodes('test', { limit: 10 })

      expect(mockFetch).toHaveBeenCalledWith(expect.stringContaining('limit=10'), expect.any(Object))
    })

    it('should handle empty search results', async () => {
      const mockData = {
        results: [],
        total: 0
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockData,
      } as Response)

      const result = await client.searchNodes('nonexistent')

      expect(result.results).toHaveLength(0)
      expect(result.total).toBe(0)
    })
  })

  describe('getNodeDetail', () => {
    it('should fetch node detail by id', async () => {
      const mockData = {
        node: { id: '1', label: 'Person', properties: { name: '张三' } },
        relationships: { incoming: [], outgoing: [] },
        neighbors: []
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockData,
      } as Response)

      const result = await client.getNodeDetail('1')

      expect(mockFetch).toHaveBeenCalledWith(`${baseUrl}/graph/node/1`, expect.any(Object))
      expect(result).toEqual(mockData)
    })

    it('should throw error when node not found', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
        statusText: 'Not Found',
      } as Response)

      await expect(client.getNodeDetail('999999')).rejects.toThrow()
    })
  })

  describe('getNodeNeighbors', () => {
    it('should fetch node neighbors with default depth', async () => {
      const mockData = {
        nodes: [
          { id: '1', label: 'Person', properties: { name: '张三' } },
          { id: '2', label: 'Person', properties: { name: '李四' } }
        ],
        edges: [{ id: 'e1', from: '1', to: '2', type: 'KNOWS', properties: {} }],
        center_node: '1'
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockData,
      } as Response)

      const result = await client.getNodeNeighbors('1')

      expect(result.nodes).toHaveLength(2)
      expect(result.edges).toHaveLength(1)
    })

    it('should fetch node neighbors with custom depth', async () => {
      const mockData = {
        nodes: [],
        edges: [],
        center_node: '1'
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockData,
      } as Response)

      await client.getNodeNeighbors('1', { depth: 2 })

      expect(mockFetch).toHaveBeenCalledWith(expect.stringContaining('depth=2'), expect.any(Object))
    })
  })

  describe('getQueryResultGraph', () => {
    it('should fetch query result graph', async () => {
      const mockData = {
        nodes: [{ id: '1', label: 'Person', properties: {} }],
        edges: [],
        stats: { total_nodes: 1, total_edges: 0, node_labels: ['Person'], relationship_types: [] }
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockData,
      } as Response)

      const result = await client.getQueryResultGraph({
        query: '张三是谁？',
        node_ids: ['1'],
        max_depth: 2
      })

      expect(result).toEqual(mockData)
    })

    it('should send POST request with correct body', async () => {
      const mockData = {
        nodes: [],
        edges: [],
        stats: { total_nodes: 0, total_edges: 0, node_labels: [], relationship_types: [] }
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockData,
      } as Response)

      await client.getQueryResultGraph({
        query: 'test query',
        node_ids: ['1', '2'],
        max_depth: 3
      })

      expect(mockFetch).toHaveBeenCalledWith(
        `${baseUrl}/graph/query-result`,
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({
            query: 'test query',
            node_ids: ['1', '2'],
            max_depth: 3
          })
        })
      )
    })
  })
})
