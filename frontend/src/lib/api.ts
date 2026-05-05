const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

export interface HealthResponse {
  status: string;
  neo4j_connected: boolean;
  version: string;
}

export interface SchemaResponse {
  schema: string;
  node_labels: string[];
  relationship_types: string[];
}

export interface QueryRequest {
  question: string;
  use_hybrid?: boolean;
  history?: Array<{ role: string; content: string }>;
}

export interface QueryResponse {
  question: string;
  answer: string;
  routing: string;
  documents_count: number;
  sources: Array<{ content: string; score: number }>;
}

export interface IngestRequest {
  file_path?: string | null;
  directory?: string | null;
  extract_entities?: boolean;
  create_embeddings?: boolean;
}

export interface IngestResponse {
  status: string;
  documents_processed: number;
  results: Array<{
    document_id: string;
    chunks_created: number;
    entities_extracted: number;
    relationships_created: number;
  }>;
}

export interface RetrievalVectorResponse {
  results: Array<{
    id: string;
    content: string;
    score: number;
    metadata: any;
  }>;
}

export interface RetrievalGraphResponse {
  cypher: string;
  results: Array<{ d: any }>;
}

export interface RetrievalHybridResponse {
  vector_results: any[];
  graph_results: any[];
  combined_score: number;
}

export interface GraphData {
  nodes: any[];
  edges: any[];
  stats: any;
}

export interface GraphSearchResponse {
  results: GraphSearchResult[];
  total: number;
}

export interface GraphSearchResult {
  id: string;
  label: string;
  properties: Record<string, any>;
  score?: number;
}

export interface NodeDetail {
  node: any;
  relationships: any;
  neighbors: any[];
}

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl?: string) {
    this.baseUrl = baseUrl || BASE_URL;
  }

  private async request<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    const headers = {
      'Content-Type': 'application/json',
      ...options?.headers,
    };

    try {
      const response = await fetch(url, { ...options, headers });
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }
      return response.json();
    } catch (error) {
      console.error(`API Error (${endpoint}):`, error);
      throw error;
    }
  }

  async checkHealth(): Promise<HealthResponse> {
    return this.request<HealthResponse>('/health');
  }

  async getSchema(): Promise<SchemaResponse> {
    return this.request<SchemaResponse>('/schema');
  }

  async query(data: QueryRequest): Promise<QueryResponse> {
    return this.request<QueryResponse>('/query', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async ingest(data: IngestRequest): Promise<IngestResponse> {
    return this.request<IngestResponse>('/ingest', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async retrieveVector(query: string, topK: number = 5): Promise<RetrievalVectorResponse> {
    return this.request<RetrievalVectorResponse>(`/retrieval/vector?query=${encodeURIComponent(query)}&top_k=${topK}`, {
      method: 'POST',
    });
  }

  async retrieveGraph(query: string): Promise<RetrievalGraphResponse> {
    return this.request<RetrievalGraphResponse>(`/retrieval/graph?query=${encodeURIComponent(query)}`, {
      method: 'POST',
    });
  }

  async retrieveHybrid(query: string, alpha: number = 0.5): Promise<RetrievalHybridResponse> {
    return this.request<RetrievalHybridResponse>(`/retrieval/hybrid?query=${encodeURIComponent(query)}&alpha=${alpha}`, {
      method: 'POST',
    });
  }

  async getGraphData(params?: {
    node_label?: string;
    limit?: number;
    offset?: number;
  }): Promise<GraphData> {
    const queryParams = new URLSearchParams();
    if (params?.node_label) queryParams.append('node_label', params.node_label);
    if (params?.limit) queryParams.append('limit', params.limit.toString());
    if (params?.offset) queryParams.append('offset', params.offset.toString());
    
    return this.request<GraphData>(`/graph/data?${queryParams}`);
  }

  async searchNodes(query: string, params?: {
    node_label?: string;
    limit?: number;
  }): Promise<GraphSearchResponse> {
    const queryParams = new URLSearchParams();
    queryParams.append('query', query);
    if (params?.node_label) queryParams.append('node_label', params.node_label);
    if (params?.limit) queryParams.append('limit', params.limit.toString());
    
    return this.request<GraphSearchResponse>(`/graph/search?${queryParams}`);
  }

  async getNodeDetail(nodeId: string): Promise<NodeDetail> {
    return this.request<NodeDetail>(`/graph/node/${nodeId}`);
  }

  async getNodeNeighbors(nodeId: string, params?: {
    depth?: number;
    relationship_type?: string;
  }): Promise<GraphData> {
    const queryParams = new URLSearchParams();
    if (params?.depth) queryParams.append('depth', params.depth.toString());
    if (params?.relationship_type) queryParams.append('relationship_type', params.relationship_type);
    
    return this.request<GraphData>(`/graph/node/${nodeId}/neighbors?${queryParams}`);
  }

  async getQueryResultGraph(request: {
    query: string;
    node_ids: string[];
    max_depth?: number;
  }): Promise<GraphData> {
    return this.request<GraphData>('/graph/query-result', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }
}

export const api = new ApiClient();
export { ApiClient };
