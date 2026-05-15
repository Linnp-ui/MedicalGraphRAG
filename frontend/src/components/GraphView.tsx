import React, { useState, useEffect, useCallback, useRef } from 'react';
import { RefreshCw, AlertCircle, Search, Database, X, Filter, ZoomIn, ZoomOut, Maximize, Loader2, GitBranch, ChevronDown, Check } from 'lucide-react';
import { api } from '../lib/api';
import type { GraphData, GraphSearchResult } from '../lib/api';
import { GraphCanvas } from './graph/GraphCanvas';

interface NodeTypeInfo {
  label: string;
  count: number;
  color: string;
}

const NODE_TYPE_COLORS: Record<string, string> = {
  'Disease': '#EF4444',
  'Symptom': '#F97316',
  'Drug': '#22C55E',
  'Examination': '#3B82F6',
  'Treatment': '#8B5CF6',
  'Anatomy': '#06B6D4',
  'Department': '#EC4899',
  'Person': '#3B82F6',
  'Document': '#10B981',
  'Organization': '#8B5CF6',
  'default': '#6B7280'
};

export function GraphView() {
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedLabel, setSelectedLabel] = useState<string | null>(null);
  const [searchResults, setSearchResults] = useState<GraphSearchResult[]>([]);
  const [showSearchResults, setShowSearchResults] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  const [highlightedNodes, setHighlightedNodes] = useState<Set<string>>(new Set());
  const [centerNodeId, setCenterNodeId] = useState<string | null>(null);
  const [showLabelDropdown, setShowLabelDropdown] = useState(false);
  const [nodeTypes, setNodeTypes] = useState<NodeTypeInfo[]>([]);
  
  const searchTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const searchContainerRef = useRef<HTMLDivElement>(null);
  const labelDropdownRef = useRef<HTMLDivElement>(null);

  const loadGraphData = useCallback(async (nodeLabel?: string) => {
    setIsLoading(true);
    setError(null);
    setHighlightedNodes(new Set());
    setCenterNodeId(null);
    try {
      const data = await api.getGraphData({ 
        node_label: nodeLabel,
        limit: 500 
      });
      setGraphData(data);
      setSelectedLabel(nodeLabel || null);
      
      if (data.stats?.node_labels) {
        const types: NodeTypeInfo[] = data.stats.node_labels.map((label: string) => ({
          label,
          count: 0,
          color: NODE_TYPE_COLORS[label] || NODE_TYPE_COLORS.default
        }));
        setNodeTypes(types);
      }
    } catch (err: any) {
      setError(err.message || '加载图谱数据失败');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadGraphData();
  }, [loadGraphData]);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (searchContainerRef.current && !searchContainerRef.current.contains(event.target as Node)) {
        setShowSearchResults(false);
      }
      if (labelDropdownRef.current && !labelDropdownRef.current.contains(event.target as Node)) {
        setShowLabelDropdown(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleSearchInputChange = async (value: string) => {
    setSearchQuery(value);
    
    if (searchTimeoutRef.current) {
      clearTimeout(searchTimeoutRef.current);
    }

    if (!value.trim()) {
      setSearchResults([]);
      setShowSearchResults(false);
      return;
    }

    searchTimeoutRef.current = setTimeout(async () => {
      setIsSearching(true);
      try {
        const results = await api.searchNodes(value, { limit: 10 });
        setSearchResults(results.results);
        setShowSearchResults(true);
      } catch (err: any) {
        console.error('Search failed:', err);
        setSearchResults([]);
      } finally {
        setIsSearching(false);
      }
    }, 300);
  };

  const handleSearchResultClick = async (result: GraphSearchResult) => {
    setSearchQuery('');
    setSearchResults([]);
    setShowSearchResults(false);
    
    setIsLoading(true);
    try {
      const graphData = await api.getNodeNeighbors(result.id, { depth: 2 });
      setGraphData(graphData);
      setSelectedLabel(null);
      setCenterNodeId(result.id);
      
      const relatedNodeIds = new Set<string>();
      relatedNodeIds.add(result.id);
      graphData.nodes.forEach((n: any) => relatedNodeIds.add(n.id));
      setHighlightedNodes(relatedNodeIds);
    } catch (err: any) {
      setError(err.message || '加载节点失败');
    } finally {
      setIsLoading(false);
    }
  };

  const handleLabelClick = async (label: string) => {
    if (selectedLabel === label) {
      await loadGraphData();
    } else {
      await loadGraphData(label);
    }
    setShowLabelDropdown(false);
  };

  const handleClearFilter = async () => {
    await loadGraphData();
  };

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    
    setIsLoading(true);
    try {
      const results = await api.searchNodes(searchQuery, { limit: 20 });
      if (results.results.length > 0) {
        const nodeIds = results.results.map(r => r.id);
        const graphData = await api.getQueryResultGraph({
          query: searchQuery,
          node_ids: nodeIds,
          max_depth: 2
        });
        setGraphData(graphData);
        setSelectedLabel(null);
        
        const relatedNodeIds = new Set<string>();
        nodeIds.forEach(id => relatedNodeIds.add(id));
        graphData.nodes.forEach((n: any) => relatedNodeIds.add(n.id));
        setHighlightedNodes(relatedNodeIds);
      }
    } catch (err: any) {
      setError(err.message || '搜索失败');
    } finally {
      setIsLoading(false);
    }
  };

  const handleShowRelations = async (nodeId: string) => {
    setIsLoading(true);
    try {
      const data = await api.getNodeNeighbors(nodeId, { depth: 2 });
      setGraphData(data);
      setSelectedLabel(null);
      setCenterNodeId(nodeId);
      setHighlightedNodes(new Set());
    } catch (err: any) {
      setError(err.message || '加载关联节点失败');
    } finally {
      setIsLoading(false);
    }
  };

  const handleNodeClick = async (nodeId: string) => {
    if (highlightedNodes.size > 0 && !highlightedNodes.has(nodeId)) {
      setHighlightedNodes(new Set());
    }
  };

  const handleClearHighlight = () => {
    setHighlightedNodes(new Set());
    setCenterNodeId(null);
  };

  return (
    <div className="flex h-full w-full bg-slate-50">
      <div className="flex-1 flex flex-col h-full overflow-hidden">
        <div className="bg-white border-b border-gray-200 px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">医疗知识图谱</h1>
              <p className="text-gray-500 mt-1">可视化展示医疗知识图谱结构</p>
            </div>
            <button
              onClick={() => loadGraphData()}
              disabled={isLoading}
              className="flex items-center gap-2 px-4 py-2 bg-white border border-gray-300 rounded-lg text-sm font-medium text-gray-700 hover:bg-gray-50 transition-colors disabled:opacity-50"
            >
              <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
              刷新
            </button>
          </div>
          
          <div className="mt-4 flex gap-2" ref={searchContainerRef}>
            <div className="flex-1 relative">
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => handleSearchInputChange(e.target.value)}
                placeholder="搜索节点..."
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
              />
              
              {isSearching && (
                <div className="absolute right-3 top-1/2 transform -translate-y-1/2">
                  <Loader2 className="w-4 h-4 text-gray-400 animate-spin" />
                </div>
              )}

              {showSearchResults && searchResults.length > 0 && (
                <div className="absolute top-full left-0 right-0 mt-1 bg-white border border-gray-200 rounded-lg shadow-lg max-h-64 overflow-y-auto z-20">
                  {searchResults.map((result) => (
                    <button
                      key={result.id}
                      onClick={() => handleSearchResultClick(result)}
                      className="w-full px-4 py-3 text-left hover:bg-gray-50 border-b border-gray-100 last:border-b-0 flex items-center justify-between"
                    >
                      <div>
                        <div className="font-medium text-gray-900">
                          {result.properties.name || result.properties.title || result.id}
                        </div>
                        <div className="text-sm text-gray-500 flex items-center gap-2">
                          <span className="px-2 py-0.5 bg-indigo-50 text-indigo-600 rounded text-xs">
                            {result.label}
                          </span>
                          <span className="text-xs text-gray-400">
                            ID: {result.id}
                          </span>
                        </div>
                      </div>
                      {result.score !== undefined && (
                        <div className="text-xs text-gray-400">
                          {Math.round(result.score * 100)}%
                        </div>
                      )}
                    </button>
                  ))}
                </div>
              )}

              {showSearchResults && searchResults.length === 0 && searchQuery && !isSearching && (
                <div className="absolute top-full left-0 right-0 mt-1 bg-white border border-gray-200 rounded-lg shadow-lg p-4 text-center text-gray-500 z-20">
                  未找到匹配的节点
                </div>
              )}
            </div>
            
            <div className="relative" ref={labelDropdownRef}>
              <button
                onClick={() => setShowLabelDropdown(!showLabelDropdown)}
                className="flex items-center gap-2 px-4 py-2 bg-white border border-gray-300 rounded-lg text-sm font-medium text-gray-700 hover:bg-gray-50 transition-colors"
              >
                <Filter className="w-4 h-4" />
                类型筛选
                <ChevronDown className={`w-4 h-4 transition-transform ${showLabelDropdown ? 'rotate-180' : ''}`} />
              </button>
              
              {showLabelDropdown && nodeTypes.length > 0 && (
                <div className="absolute top-full left-0 mt-1 bg-white border border-gray-200 rounded-lg shadow-lg min-w-[200px] max-h-64 overflow-y-auto z-20">
                  <button
                    onClick={() => handleLabelClick('')}
                    className={`w-full px-4 py-2 text-left text-sm hover:bg-gray-50 flex items-center justify-between ${
                      !selectedLabel ? 'bg-indigo-50 text-indigo-700' : 'text-gray-700'
                    }`}
                  >
                    <span>全部类型</span>
                    {!selectedLabel && <Check className="w-4 h-4" />}
                  </button>
                  {nodeTypes.map((type) => (
                    <button
                      key={type.label}
                      onClick={() => handleLabelClick(type.label)}
                      className={`w-full px-4 py-2 text-left text-sm hover:bg-gray-50 flex items-center justify-between ${
                        selectedLabel === type.label ? 'bg-indigo-50 text-indigo-700' : 'text-gray-700'
                      }`}
                    >
                      <div className="flex items-center gap-2">
                        <span 
                          className="w-3 h-3 rounded-full" 
                          style={{ backgroundColor: type.color }}
                        />
                        <span>{type.label}</span>
                      </div>
                      {selectedLabel === type.label && <Check className="w-4 h-4" />}
                    </button>
                  ))}
                </div>
              )}
            </div>
            
            <button
              onClick={handleSearch}
              disabled={isLoading || isSearching}
              className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors disabled:opacity-50"
            >
              <Search className="w-5 h-5" />
            </button>
          </div>
        </div>

        {error && (
          <div className="mx-6 mt-4 bg-rose-50 border border-rose-200 rounded-lg p-4 flex items-start gap-3 text-rose-700">
            <AlertCircle className="w-5 h-5 shrink-0 mt-0.5" />
            <div>
              <h4 className="font-medium">加载图谱数据失败</h4>
              <p className="text-sm mt-1 text-rose-600">{error}</p>
            </div>
          </div>
        )}

        {selectedLabel && (
          <div className="mx-6 mt-4 bg-indigo-50 border border-indigo-200 rounded-lg p-4 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Filter className="w-4 h-4 text-indigo-600" />
              <span className="text-sm text-indigo-700">
                当前筛选: <span className="font-medium">{selectedLabel}</span>
              </span>
            </div>
            <button
              onClick={handleClearFilter}
              className="flex items-center gap-1 px-3 py-1 bg-white text-indigo-600 rounded-md text-sm hover:bg-indigo-100 transition-colors"
            >
              <X className="w-4 h-4" />
              清除筛选
            </button>
          </div>
        )}

        {highlightedNodes.size > 0 && (
          <div className="mx-6 mt-4 bg-amber-50 border border-amber-200 rounded-lg p-4 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <GitBranch className="w-4 h-4 text-amber-600" />
              <span className="text-sm text-amber-700">
                已高亮 <span className="font-medium">{highlightedNodes.size}</span> 个关联节点
                {centerNodeId && <span className="ml-2">（中心节点: {centerNodeId}）</span>}
              </span>
            </div>
            <button
              onClick={handleClearHighlight}
              className="flex items-center gap-1 px-3 py-1 bg-white text-amber-600 rounded-md text-sm hover:bg-amber-100 transition-colors"
            >
              <X className="w-4 h-4" />
              清除高亮
            </button>
          </div>
        )}

        <div className="flex-1 relative">
            {isLoading && (
              <div className="absolute inset-0 flex items-center justify-center bg-white bg-opacity-75 z-10">
                <div className="text-center">
                  <RefreshCw className="w-8 h-8 animate-spin text-indigo-600 mx-auto mb-2" />
                  <div className="text-gray-600">加载图谱数据...</div>
                </div>
              </div>
            )}

            {graphData && !isLoading && graphData.nodes.length > 0 && (
              <>
                <GraphCanvas
                  nodes={graphData.nodes}
                  edges={graphData.edges}
                  className="absolute inset-0"
                  onNodeClick={handleNodeClick}
                  onShowRelations={handleShowRelations}
                  highlightedNodes={highlightedNodes}
                  centerNodeId={centerNodeId}
                />
                
                <div className="absolute bottom-4 right-4 flex gap-2">
                  <button
                    onClick={() => {
                      const container = document.querySelector('.vis-network') as any;
                      if (container && container.network) {
                        const scale = container.network.getScale();
                        container.network.moveTo({ scale: scale * 1.2 });
                      }
                    }}
                    className="p-2 bg-white rounded-lg shadow-md hover:bg-gray-50 transition-colors"
                    title="放大"
                  >
                    <ZoomIn className="w-5 h-5 text-gray-700" />
                  </button>
                  <button
                    onClick={() => {
                      const container = document.querySelector('.vis-network') as any;
                      if (container && container.network) {
                        const scale = container.network.getScale();
                        container.network.moveTo({ scale: scale / 1.2 });
                      }
                    }}
                    className="p-2 bg-white rounded-lg shadow-md hover:bg-gray-50 transition-colors"
                    title="缩小"
                  >
                    <ZoomOut className="w-5 h-5 text-gray-700" />
                  </button>
                  <button
                    onClick={() => {
                      const container = document.querySelector('.vis-network') as any;
                      if (container && container.network) {
                        container.network.fit({
                          animation: {
                            duration: 500,
                            easingFunction: 'easeInOutQuad'
                          }
                        });
                      }
                    }}
                    className="p-2 bg-white rounded-lg shadow-md hover:bg-gray-50 transition-colors"
                    title="适应视图"
                  >
                    <Maximize className="w-5 h-5 text-gray-700" />
                  </button>
                </div>
              </>
            )}

            {graphData && !isLoading && graphData.nodes.length === 0 && (
              <div className="w-full h-full flex items-center justify-center bg-gray-50">
                <div className="text-center">
                  <Database className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                  <h2 className="text-xl font-semibold text-gray-500 mb-2">暂无数据</h2>
                  <p className="text-sm text-gray-400">
                    {selectedLabel 
                          ? `数据库中没有 ${selectedLabel} 类型的节点`
                          : '数据库中暂无图谱数据'}
                  </p>
                </div>
              </div>
            )}
          </div>
      </div>

      <div className="w-80 bg-white border-l border-gray-200 overflow-y-auto">
        <div className="p-4 space-y-4">
          {graphData && graphData.stats && (
            <>
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
                <h3 className="text-sm font-medium text-gray-700 mb-3">统计信息</h3>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">当前节点数:</span>
                    <span className="font-medium">{graphData.nodes?.length || 0}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">当前关系数:</span>
                    <span className="font-medium">{graphData.edges?.length || 0}</span>
                  </div>
                  {!selectedLabel && (
                    <>
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-600">总节点数:</span>
                        <span className="font-medium">{graphData.stats.total_nodes || 0}</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-600">总关系数:</span>
                        <span className="font-medium">{graphData.stats.total_edges || 0}</span>
                      </div>
                    </>
                  )}
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">节点类型:</span>
                    <span className="font-medium">{graphData.stats.node_labels?.length || 0}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">关系类型:</span>
                    <span className="font-medium">{graphData.stats.relationship_types?.length || 0}</span>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
                <h3 className="text-sm font-medium text-gray-700 mb-3">节点类型（点击筛选）</h3>
                <div className="flex flex-wrap gap-2">
                  {(graphData.stats.node_labels || []).map((label: string) => (
                    <button
                      key={label}
                      onClick={() => handleLabelClick(label)}
                      className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                        selectedLabel === label
                          ? 'bg-indigo-600 text-white shadow-md'
                          : 'bg-indigo-50 text-indigo-700 hover:bg-indigo-100 hover:shadow'
                      }`}
                    >
                      {label}
                    </button>
                  ))}
                </div>
              </div>

              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
                <h3 className="text-sm font-medium text-gray-700 mb-3">关系类型</h3>
                <div className="flex flex-wrap gap-2">
                  {(graphData.stats.relationship_types || []).slice(0, 10).map((type: string) => (
                    <span key={type} className="px-2 py-1 bg-emerald-50 text-emerald-700 rounded text-xs font-mono">
                      {type}
                    </span>
                  ))}
                  {(graphData.stats.relationship_types?.length || 0) > 10 && (
                    <span className="text-xs text-gray-400">
                      +{(graphData.stats.relationship_types?.length || 0) - 10} 更多
                    </span>
                  )}
                </div>
              </div>

              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
                <h3 className="text-sm font-medium text-gray-700 mb-3">操作提示</h3>
                <div className="text-xs text-gray-500 space-y-2">
                  <p>• <strong>左键点击</strong> - 选中节点</p>
                  <p>• <strong>右键点击</strong> - 打开菜单</p>
                  <p>• <strong>拖拽</strong> - 移动画布</p>
                  <p>• <strong>滚轮</strong> - 缩放视图</p>
                  <p>• <strong>展示相关联系</strong> - 显示关联节点</p>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
