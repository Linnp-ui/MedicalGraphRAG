export const nodeColors: Record<string, string> = {
  Person: '#3B82F6',
  Organization: '#10B981',
  Location: '#F59E0B',
  Event: '#8B5CF6',
  Concept: '#EC4899',
  default: '#6B7280'
};

export const edgeColors: Record<string, string> = {
  KNOWS: '#3B82F6',
  WORKS_FOR: '#10B981',
  LOCATED_IN: '#F59E0B',
  PART_OF: '#8B5CF6',
  RELATED_TO: '#EC4899',
  default: '#9CA3AF'
};

export const nodeSizeConfig = {
  min: 20,
  max: 60,
  scale: 'log' as const
};

export const visNetworkOptions = {
  physics: {
    enabled: true,
    barnesHut: {
      gravitationalConstant: -3000,
      centralGravity: 0.3,
      springLength: 100,
      springConstant: 0.05,
      damping: 0.09
    },
    stabilization: {
      enabled: true,
      iterations: 200,
      updateInterval: 25,
      onlyDynamicEdges: false,
      fit: true
    },
    timestep: 0.5,
    adaptiveTimestep: true
  },
  interaction: {
    hover: true,
    tooltipDelay: 200,
    zoomView: true,
    dragView: true
  },
  rendering: {
    hideEdgesOnDrag: true,
    hideNodesOnDrag: false,
    hideEdgesOnZoom: true,
    hideNodesOnZoom: false
  },
  nodes: {
    shape: 'dot',
    scaling: {
      min: nodeSizeConfig.min,
      max: nodeSizeConfig.max,
      label: {
        enabled: true,
        min: 14,
        max: 30,
        maxVisible: 30,
        drawThreshold: 5
      }
    },
    borderWidth: 2,
    borderWidthSelected: 4,
    font: {
      size: 14,
      color: '#1F2937'
    }
  },
  edges: {
    width: 2,
    smooth: {
      enabled: true,
      type: 'continuous',
      roundness: 0.5
    },
    arrows: {
      to: {
        enabled: true,
        scaleFactor: 0.8
      }
    },
    font: {
      size: 12,
      align: 'middle'
    }
  }
};

export function getNodeColor(label: string): string {
  return nodeColors[label] || nodeColors.default;
}

export function getEdgeColor(type: string): string {
  return edgeColors[type] || edgeColors.default;
}

export function calculateNodeSize(degree: number, maxDegree: number): number {
  if (maxDegree === 0) return nodeSizeConfig.min;
  
  const normalizedDegree = degree / maxDegree;
  const logScale = Math.log(1 + normalizedDegree * 9) / Math.log(10);
  
  return nodeSizeConfig.min + logScale * (nodeSizeConfig.max - nodeSizeConfig.min);
}
