export interface GraphNode {
  id: string;
  label: string;
  properties: Record<string, any>;
  degree?: number;
}

export interface GraphEdge {
  id: string;
  from: string;
  to: string;
  type: string;
  properties?: Record<string, any>;
}

export interface GraphStats {
  total_nodes: number;
  total_edges: number;
  node_labels: string[];
  relationship_types: string[];
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
  stats: GraphStats;
}

export interface GraphSearchResult {
  id: string;
  label: string;
  properties: Record<string, any>;
  score?: number;
}

export interface GraphSearchResponse {
  results: GraphSearchResult[];
  total: number;
}

export interface NodeRelationships {
  incoming: Array<{
    from_node: string;
    type: string;
    properties?: Record<string, any>;
  }>;
  outgoing: Array<{
    to_node: string;
    type: string;
    properties?: Record<string, any>;
  }>;
}

export interface NodeDetail {
  node: GraphNode;
  relationships: NodeRelationships;
  neighbors: GraphNode[];
}

export interface GraphFilters {
  nodeLabels: string[];
  relationshipTypes: string[];
  propertyRange?: Record<string, [any, any]>;
}

export interface ViewSettings {
  showLabels: boolean;
  showArrows: boolean;
  layout: LayoutType;
  zoomLevel: number;
}

export type LayoutType = 'force' | 'hierarchical' | 'circular';
