import React, { useEffect, useRef, useState } from 'react';
import { Network } from 'vis-network/standalone';
import { DataSet } from 'vis-data';
import type { GraphNode, GraphEdge } from '../../types/graph';
import { getNodeColor, getEdgeColor, calculateNodeSize } from '../../lib/graphConfig';
import { ContextMenu } from './ContextMenu';

interface GraphCanvasProps {
  nodes: GraphNode[];
  edges: GraphEdge[];
  onNodeClick?: (nodeId: string) => void;
  onShowRelations?: (nodeId: string) => void;
  highlightedNodes?: Set<string>;
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
  className = '' 
}: GraphCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const networkRef = useRef<Network | null>(null);
  const onNodeClickRef = useRef(onNodeClick);
  const [contextMenu, setContextMenu] = useState<ContextMenuState>({
    visible: false,
    x: 0,
    y: 0,
    nodeId: '',
    nodeName: '',
    nodeLabel: ''
  });

  onNodeClickRef.current = onNodeClick;

  useEffect(() => {
    if (!containerRef.current || nodes.length === 0) return;

    const maxDegree = Math.max(...nodes.map(n => n.degree || 0), 1);

    const visNodes = new DataSet(
      nodes.map(node => {
        const isHighlighted = highlightedNodes?.has(node.id);
        return {
          id: node.id,
          label: node.properties.name || node.properties.title || node.id,
          color: {
            background: isHighlighted ? '#F59E0B' : getNodeColor(node.label),
            border: isHighlighted ? '#D97706' : getNodeColor(node.label),
            highlight: {
              background: '#F59E0B',
              border: '#D97706'
            },
            hover: {
              background: isHighlighted ? '#FBBF24' : getNodeColor(node.label),
              border: '#374151'
            }
          },
          size: isHighlighted ? calculateNodeSize(node.degree || 0, maxDegree) * 1.3 : calculateNodeSize(node.degree || 0, maxDegree),
          group: node.label,
          title: `${node.label}: ${node.properties.name || node.id}\n${Object.entries(node.properties)
            .filter(([key]) => key !== 'name' && key !== 'title')
            .map(([key, value]) => `${key}: ${value}`)
            .join('\n')}`,
          font: {
            size: isHighlighted ? 16 : 14,
            color: '#1F2937',
            bold: isHighlighted
          },
          borderWidth: isHighlighted ? 3 : 2
        };
      })
    );

    const visEdges = new DataSet(
      edges.map(edge => {
        const isHighlighted = highlightedNodes?.has(edge.from) && highlightedNodes?.has(edge.to);
        return {
          id: edge.id,
          from: edge.from,
          to: edge.to,
          label: edge.type,
          color: {
            color: isHighlighted ? '#F59E0B' : getEdgeColor(edge.type),
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

    const data = {
      nodes: visNodes,
      edges: visEdges
    };

    const nodeCount = nodes.length;
    const gravitationalConstant = nodeCount > 100 ? -8000 : nodeCount > 50 ? -5000 : -3000;
    const springLength = nodeCount > 100 ? 200 : nodeCount > 50 ? 150 : 100;
    const centralGravity = nodeCount > 100 ? 0.1 : nodeCount > 50 ? 0.2 : 0.3;

    const options = {
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

    networkRef.current = new Network(containerRef.current, data, options);

    networkRef.current.on('click', (params: any) => {
      if (params.nodes.length > 0 && onNodeClickRef.current) {
        onNodeClickRef.current(params.nodes[0]);
      }
    });

    networkRef.current.on('oncontext', (params: any) => {
      params.event.preventDefault();
      
      const nodeId = networkRef.current?.getNodeAt(params.pointer.DOM);
      if (nodeId) {
        const node = nodes.find(n => n.id === nodeId);
        if (node) {
          setContextMenu({
            visible: true,
            x: params.event.pageX,
            y: params.event.pageY,
            nodeId: nodeId,
            nodeName: node.properties.name || node.properties.title || nodeId,
            nodeLabel: node.label
          });
        }
      }
    });

    networkRef.current.once('stabilizationIterationsDone', () => {
      if (networkRef.current) {
        networkRef.current.fit({
          animation: {
            duration: 500,
            easingFunction: 'easeInOutQuad'
          }
        });
      }
    });

    return () => {
      if (networkRef.current) {
        networkRef.current.destroy();
        networkRef.current = null;
      }
    };
  }, [nodes, edges, highlightedNodes]);

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
