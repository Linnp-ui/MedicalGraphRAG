import type { GraphNode, GraphEdge } from '../types/graph';
import { getNodeColor, getEdgeColor, calculateNodeSize } from './graphConfig';

export function convertToVisNodes(nodes: GraphNode[]): any {
  const vis = require('vis-data');
  const maxDegree = Math.max(...nodes.map(n => n.degree || 0), 1);
  
  const visNodes = nodes.map(node => ({
    id: node.id,
    label: node.properties.name || node.properties.title || node.id,
    color: {
      background: getNodeColor(node.label),
      border: getNodeColor(node.label),
      highlight: {
        background: getNodeColor(node.label),
        border: '#1F2937'
      },
      hover: {
        background: getNodeColor(node.label),
        border: '#374151'
      }
    },
    size: calculateNodeSize(node.degree || 0, maxDegree),
    group: node.label,
    title: `${node.label}: ${node.properties.name || node.id}\n${Object.entries(node.properties)
      .filter(([key]) => key !== 'name' && key !== 'title')
      .map(([key, value]) => `${key}: ${value}`)
      .join('\n')}`,
    font: {
      size: 14,
      color: '#1F2937'
    }
  }));
  
  return new vis.DataSet(visNodes);
}

export function convertToVisEdges(edges: GraphEdge[]): any {
  const vis = require('vis-data');
  
  const visEdges = edges.map(edge => ({
    id: edge.id,
    from: edge.from,
    to: edge.to,
    label: edge.type,
    color: {
      color: getEdgeColor(edge.type),
      highlight: '#1F2937',
      hover: '#374151'
    },
    arrows: 'to',
    smooth: {
      enabled: true,
      type: 'curvedCW',
      roundness: 0.2
    },
    title: `${edge.type}${edge.properties && Object.keys(edge.properties).length > 0 
      ? '\n' + Object.entries(edge.properties)
          .map(([key, value]) => `${key}: ${value}`)
          .join('\n')
      : ''}`
  }));
  
  return new vis.DataSet(visEdges);
}

export function filterNodesByLabel(
  nodes: GraphNode[],
  labels: string[]
): GraphNode[] {
  if (labels.length === 0) return nodes;
  return nodes.filter(node => labels.includes(node.label));
}

export function filterEdgesByType(
  edges: GraphEdge[],
  types: string[]
): GraphEdge[] {
  if (types.length === 0) return edges;
  return edges.filter(edge => types.includes(edge.type));
}

export function getConnectedNodes(
  nodeId: string,
  edges: GraphEdge[]
): string[] {
  const connected = new Set<string>();
  
  edges.forEach(edge => {
    if (edge.from === nodeId) {
      connected.add(edge.to);
    } else if (edge.to === nodeId) {
      connected.add(edge.from);
    }
  });
  
  return Array.from(connected);
}

export function highlightNode(
  nodeId: string,
  nodes: any
): void {
  const node = nodes.get(nodeId);
  if (node) {
    nodes.update({
      id: nodeId,
      borderWidth: 4,
      shadow: true
    });
  }
}

export function unhighlightNode(
  nodeId: string,
  nodes: any
): void {
  const node = nodes.get(nodeId);
  if (node) {
    nodes.update({
      id: nodeId,
      borderWidth: 2,
      shadow: false
    });
  }
}

export function focusOnNode(
  network: any,
  nodeId: string,
  options?: { scale?: number; animation?: boolean }
): void {
  const scale = options?.scale || 1.0;
  const animation = options?.animation !== false;
  
  network.focus(nodeId, {
    scale,
    animation: animation ? {
      duration: 500,
      easingFunction: 'easeInOutQuad'
    } : false
  });
}

export function fitToView(network: any): void {
  network.fit({
    animation: {
      duration: 500,
      easingFunction: 'easeInOutQuad'
    }
  });
}
