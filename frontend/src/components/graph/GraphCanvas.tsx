import { useEffect, useRef, useState } from 'react';
import { Network } from 'vis-network/standalone';
import { DataSet } from 'vis-data';
import type { Data, Edge, IdType, Node, Options } from 'vis-network/standalone';
import type { GraphNode, GraphEdge } from '../../types/graph';

interface VisNode extends Node {
  degree?: number;
}
import { getNodeColor, getEdgeColor, calculateNodeSize } from '../../lib/graphConfig';
import { ContextMenu } from './ContextMenu';

interface GraphCanvasProps {
  nodes: GraphNode[];
  edges: GraphEdge[];
  onNodeClick?: (nodeId: string) => void;
  onShowRelations?: (nodeId: string) => void;
  highlightedNodes?: Set<string>;
  centerNodeId?: string | null;
  className?: string;
}

interface ContextMenuState {
  visible: boolean;
  x: number;
  y: number;
  nodeId: string;
  nodeName: string;
  nodeLabel: string;
}

export function GraphCanvas({ 
  nodes, 
  edges, 
  onNodeClick, 
  onShowRelations,
  highlightedNodes,
  centerNodeId,
  className = '' 
}: GraphCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const networkRef = useRef<Network | null>(null);
  const onNodeClickRef = useRef(onNodeClick);
  const nodesRef = useRef<DataSet<VisNode> | null>(null);
  const edgesRef = useRef<DataSet<Edge> | null>(null);
  const highlightedNodesRef = useRef(highlightedNodes);
  const nodesDataRef = useRef(nodes);
  const edgesDataRef = useRef(edges);
  const [contextMenu, setContextMenu] = useState<ContextMenuState>({
    visible: false,
    x: 0,
    y: 0,
    nodeId: '',
    nodeName: '',
    nodeLabel: ''
  });

  onNodeClickRef.current = onNodeClick;
  highlightedNodesRef.current = highlightedNodes;
  nodesDataRef.current = nodes;
  edgesDataRef.current = edges;

  const toNodeId = (value: IdType): string => String(value);

  useEffect(() => {
    const container = containerRef.current;
    if (!container || nodes.length === 0) return;

    const maxDegree = Math.max(...nodes.map(n => n.degree || 0), 1);
    const getBaseNodeStyle = (node: VisNode) => {
      const isHighlighted = !!highlightedNodesRef.current?.has(String(node.id));
      const nodeColor = getNodeColor(node.group as string);

      return {
        color: {
          background: isHighlighted ? '#F59E0B' : nodeColor,
          border: isHighlighted ? '#D97706' : nodeColor,
          highlight: { background: '#F59E0B', border: '#D97706' },
          hover: { background: isHighlighted ? '#FBBF24' : nodeColor, border: isHighlighted ? '#D97706' : nodeColor }
        },
        font: {
          color: '#1F2937'
        },
        size: isHighlighted ? calculateNodeSize(node.degree || 0, maxDegree) * 1.3 : calculateNodeSize(node.degree || 0, maxDegree),
        borderWidth: isHighlighted ? 3 : 2
      };
    };

    const getDimmedNodeStyle = () => ({
      color: {
        background: 'rgba(200, 200, 200, 0.3)',
        border: 'rgba(200, 200, 200, 0.5)',
        highlight: { background: 'rgba(200, 200, 200, 0.3)', border: 'rgba(200, 200, 200, 0.5)' },
        hover: { background: 'rgba(200, 200, 200, 0.3)', border: 'rgba(200, 200, 200, 0.5)' }
      },
      font: {
        color: 'rgba(150, 150, 150, 0.5)'
      }
    });

    const getBaseEdgeStyle = (edge: Edge) => {
      const fromId = String(edge.from);
      const toId = String(edge.to);
      const isHighlighted = !!highlightedNodesRef.current?.has(fromId) && !!highlightedNodesRef.current?.has(toId);
      const edgeType = edge.label as string;

      return {
        color: {
          color: isHighlighted ? '#F59E0B' : getEdgeColor(edgeType),
          highlight: '#F59E0B',
          hover: '#374151'
        },
        width: isHighlighted ? 3 : 1.5,
        font: {
          size: 10,
          align: 'middle',
          color: '#1F2937'
        }
      };
    };

    const getDimmedEdgeStyle = () => ({
      color: {
        color: 'rgba(180, 180, 180, 0.3)',
        highlight: 'rgba(180, 180, 180, 0.3)',
        hover: 'rgba(180, 180, 180, 0.3)'
      },
      width: 1,
      font: {
        size: 10,
        align: 'middle',
        color: 'rgba(150, 150, 150, 0.5)'
      }
    });

    const restoreGraphStyles = () => {
      const allNodes = nodesRef.current?.get() || [];
      const allEdges = edgesRef.current?.get() || [];
      
      const nodeUpdates = allNodes.map(node => ({
        id: node.id,
        ...getBaseNodeStyle(node)
      }));

      const edgeUpdates = allEdges.map(edge => ({
        id: edge.id,
        ...getBaseEdgeStyle(edge)
      }));

      if (nodeUpdates.length > 0) {
        nodesRef.current?.update(nodeUpdates);
      }
      if (edgeUpdates.length > 0) {
        edgesRef.current?.update(edgeUpdates);
      }
    };

    const visNodes = new DataSet<VisNode>(
      nodes.map(node => {
        const isHighlighted = !!highlightedNodesRef.current?.has(node.id);
        const nodeColor = getNodeColor(node.label);
        const nodeSize = isHighlighted ? calculateNodeSize(node.degree || 0, maxDegree) * 1.3 : calculateNodeSize(node.degree || 0, maxDegree);
        return {
          id: node.id,
          label: node.properties.name || node.properties.title || node.id,
          color: {
            background: isHighlighted ? '#F59E0B' : nodeColor,
            border: isHighlighted ? '#D97706' : nodeColor,
            highlight: { background: '#F59E0B', border: '#D97706' },
            hover: { background: isHighlighted ? '#FBBF24' : nodeColor, border: isHighlighted ? '#D97706' : nodeColor }
          },
          size: nodeSize,
          group: node.label,
          degree: node.degree || 0,
          title: `${node.label}: ${node.properties.name || node.id}\n${Object.entries(node.properties)
            .filter(([key]) => key !== 'name' && key !== 'title')
            .map(([key, value]) => `${key}: ${value}`)
            .join('\n')}`,
          font: {
            size: isHighlighted ? 16 : 14,
            color: '#1F2937',
            bold: isHighlighted ? '16px arial #1F2937' : undefined
          },
          borderWidth: isHighlighted ? 3 : 2
        };
      })
    );

    const visEdges = new DataSet<Edge>(
      edges.map(edge => {
        const isHighlighted = !!highlightedNodesRef.current?.has(edge.from) && !!highlightedNodesRef.current?.has(edge.to);
        const edgeColor = getEdgeColor(edge.type);
        return {
          id: edge.id,
          from: edge.from,
          to: edge.to,
          label: edge.type,
          color: {
            color: isHighlighted ? '#F59E0B' : edgeColor,
            highlight: '#F59E0B',
            hover: '#374151'
          },
          arrows: 'to',
          width: isHighlighted ? 3 : 1.5,
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
        };
      })
    );

    const data: Data = {
      nodes: visNodes,
      edges: visEdges
    };

    nodesRef.current = visNodes;
    edgesRef.current = visEdges;

    const nodeCount = nodes.length;
    const gravitationalConstant = nodeCount > 100 ? -8000 : nodeCount > 50 ? -5000 : -3000;
    const springLength = nodeCount > 100 ? 200 : nodeCount > 50 ? 150 : 100;
    const centralGravity = nodeCount > 100 ? 0.1 : nodeCount > 50 ? 0.2 : 0.3;

    const options: Options = {
      physics: {
        enabled: true,
        barnesHut: {
          gravitationalConstant: gravitationalConstant,
          centralGravity: centralGravity,
          springLength: springLength,
          springConstant: 0.04,
          damping: 0.15,
          avoidOverlap: 0.5
        },
        stabilization: {
          enabled: true,
          iterations: nodeCount > 100 ? 500 : 300,
          updateInterval: 25
        }
      },
      layout: {
        improvedLayout: true,
        clusterThreshold: 150
      },
      interaction: {
        hover: true,
        tooltipDelay: 200,
        zoomView: true,
        dragView: true
      },
      nodes: {
        shape: 'dot',
        borderWidth: 2,
        borderWidthSelected: 4,
        font: {
          size: nodeCount > 100 ? 12 : 14,
          color: '#1F2937'
        }
      },
      edges: {
        width: 1.5,
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
          size: 10,
          align: 'middle'
        }
      }
    };

    networkRef.current = new Network(container, data, options);

    networkRef.current.on('click', (params: { nodes: IdType[] }) => {
      if (params.nodes.length > 0 && onNodeClickRef.current) {
        onNodeClickRef.current(toNodeId(params.nodes[0]));
      }
    });

    networkRef.current.on('oncontext', (params: { event: MouseEvent; pointer: { DOM: { x: number; y: number } } }) => {
      params.event.preventDefault();
      
      const nodeId = networkRef.current?.getNodeAt(params.pointer.DOM);
      if (nodeId) {
        const normalizedNodeId = toNodeId(nodeId);
        const node = nodes.find(n => n.id === normalizedNodeId);
        if (node) {
          setContextMenu({
            visible: true,
            x: params.event.pageX,
            y: params.event.pageY,
            nodeId: normalizedNodeId,
            nodeName: node.properties.name || node.properties.title || normalizedNodeId,
            nodeLabel: node.label
          });
        }
      }
    });

    networkRef.current.on('hoverNode', (params: { node: IdType }) => {
      const hoveredNodeId = toNodeId(params.node);
      const allEdges = edgesRef.current?.get() || [];
      const allNodes = nodesRef.current?.get() || [];
      
      const connectedNodeIds = new Set<string>();
      connectedNodeIds.add(hoveredNodeId);
      
      allEdges.forEach(edge => {
        if (String(edge.from) === hoveredNodeId) connectedNodeIds.add(String(edge.to));
        if (String(edge.to) === hoveredNodeId) connectedNodeIds.add(String(edge.from));
      });

      const nodeUpdates: Partial<Node>[] = [];
      const edgeUpdates: Partial<Edge>[] = [];

      allNodes.forEach(node => {
        nodeUpdates.push({
          id: node.id,
          ...(connectedNodeIds.has(String(node.id)) ? getBaseNodeStyle(node) : getDimmedNodeStyle())
        });
      });

      allEdges.forEach(edge => {
        const edgeFromId = String(edge.from);
        const edgeToId = String(edge.to);
        const isConnectedEdge = edgeFromId === hoveredNodeId || edgeToId === hoveredNodeId;
        const isHighlighted = connectedNodeIds.has(edgeFromId) && connectedNodeIds.has(edgeToId);
        const edgeColor = getEdgeColor(edge.label as string);
        edgeUpdates.push({
          id: edge.id,
          ...(isConnectedEdge ? {
            color: {
              color: isHighlighted ? '#F59E0B' : edgeColor,
              highlight: '#F59E0B',
              hover: '#374151'
            },
            width: isHighlighted ? 3 : 1.5
          } : getDimmedEdgeStyle())
        });
      });

      if (nodeUpdates.length > 0) {
        nodesRef.current?.update(nodeUpdates);
      }
      if (edgeUpdates.length > 0) {
        edgesRef.current?.update(edgeUpdates);
      }
    });

    networkRef.current.on('blurNode', restoreGraphStyles);
    container.addEventListener('mouseleave', restoreGraphStyles);

    networkRef.current.once('stabilizationIterationsDone', () => {
      if (networkRef.current) {
        if (centerNodeId) {
          networkRef.current.focus(centerNodeId, {
            scale: 1.5,
            animation: {
              duration: 500,
              easingFunction: 'easeInOutQuad'
            }
          });
        } else {
          networkRef.current.fit({
            animation: {
              duration: 500,
              easingFunction: 'easeInOutQuad'
            }
          });
        }
      }
    });

    return () => {
      container.removeEventListener('mouseleave', restoreGraphStyles);
      if (networkRef.current) {
        networkRef.current.destroy();
        networkRef.current = null;
      }
    };
  }, [nodes, edges, highlightedNodes, centerNodeId]);

  const handleCloseContextMenu = () => {
    setContextMenu(prev => ({ ...prev, visible: false }));
  };

  return (
    <>
      <div
        ref={containerRef}
        className={`w-full h-full ${className}`}
        style={{ backgroundColor: '#F9FAFB' }}
        onContextMenu={(e) => e.preventDefault()}
      />
      {contextMenu.visible && (
        <ContextMenu
          x={contextMenu.x}
          y={contextMenu.y}
          nodeId={contextMenu.nodeId}
          nodeName={contextMenu.nodeName}
          nodeLabel={contextMenu.nodeLabel}
          onShowRelations={onShowRelations || (() => {})}
          onClose={handleCloseContextMenu}
        />
      )}
    </>
  );
}
