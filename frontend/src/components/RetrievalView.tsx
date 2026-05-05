import React, { useState } from 'react';
import { Search, Layers, Network, FileText, Loader2, AlertCircle } from 'lucide-react';
import { api } from '../lib/api';
import { cn } from '../lib/utils';

type RetrievalType = 'vector' | 'graph' | 'hybrid';

export function RetrievalView() {
  const [query, setQuery] = useState('');
  const [type, setType] = useState<RetrievalType>('hybrid');
  const [alpha, setAlpha] = useState(0.5);
  const [topK, setTopK] = useState(5);
  
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      let data;
      if (type === 'vector') {
        data = await api.retrieveVector(query, topK);
      } else if (type === 'graph') {
        data = await api.retrieveGraph(query);
      } else {
        data = await api.retrieveHybrid(query, alpha);
      }
      setResult(data);
    } catch (err: any) {
      setError(err.message || '检索失败');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex-1 overflow-y-auto bg-slate-50 p-8">
      <div className="max-w-5xl mx-auto space-y-8">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 tracking-tight">检索测试</h1>
          <p className="text-slate-500 mt-1">测试并比较不同的检索策略。</p>
        </div>

        <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
          <div className="p-6 border-b border-slate-100">
            <form onSubmit={handleSearch} className="space-y-6">
              {/* Type Selector */}
              <div className="flex gap-2 p-1 bg-slate-100 rounded-lg w-fit">
                {[
                  { id: 'hybrid', label: '混合', icon: Layers },
                  { id: 'vector', label: '向量', icon: FileText },
                  { id: 'graph', label: '图谱', icon: Network },
                ].map((t) => {
                  const Icon = t.icon;
                  return (
                    <button
                      key={t.id}
                      type="button"
                      onClick={() => setType(t.id as RetrievalType)}
                      className={cn(
                        "flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-all",
                        type === t.id ? "bg-white text-indigo-600 shadow-sm" : "text-slate-600 hover:text-slate-900"
                      )}
                    >
                      <Icon className="w-4 h-4" />
                      {t.label}
                    </button>
                  );
                })}
              </div>

              {/* Search Bar */}
              <div className="flex gap-4">
                <div className="relative flex-1">
                  <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
                  <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="输入检索查询..."
                    className="w-full pl-11 pr-4 py-3 bg-slate-50 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500 transition-all text-slate-900"
                  />
                </div>
                <button
                  type="submit"
                  disabled={isLoading || !query.trim()}
                  className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-700 text-white px-8 py-3 rounded-xl font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed shadow-sm"
                >
                  {isLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : '搜索'}
                </button>
              </div>

              {/* Advanced Options */}
              <div className="flex gap-8 pt-2">
                {type === 'hybrid' && (
                  <div className="flex-1 max-w-xs">
                    <label className="flex justify-between text-sm font-medium text-slate-700 mb-2">
                      <span>向量权重 (Alpha)</span>
                      <span className="text-indigo-600">{alpha.toFixed(2)}</span>
                    </label>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.1"
                      value={alpha}
                      onChange={(e) => setAlpha(parseFloat(e.target.value))}
                      className="w-full accent-indigo-600"
                    />
                    <div className="flex justify-between text-xs text-slate-400 mt-1">
                      <span>仅图谱 (0)</span>
                      <span>仅向量 (1)</span>
                    </div>
                  </div>
                )}
                
                {type === 'vector' && (
                  <div className="flex-1 max-w-xs">
                    <label className="block text-sm font-medium text-slate-700 mb-2">
                      Top K 结果数: {topK}
                    </label>
                    <input
                      type="range"
                      min="1"
                      max="20"
                      step="1"
                      value={topK}
                      onChange={(e) => setTopK(parseInt(e.target.value))}
                      className="w-full accent-indigo-600"
                    />
                  </div>
                )}
              </div>
            </form>
          </div>

          {/* Results Area */}
          <div className="p-6 bg-slate-50/50 min-h-[300px]">
            {error && (
              <div className="bg-rose-50 border border-rose-200 rounded-xl p-4 flex items-start gap-3 text-rose-700">
                <AlertCircle className="w-5 h-5 shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-medium">检索失败</h4>
                  <p className="text-sm mt-1 text-rose-600/90">{error}</p>
                </div>
              </div>
            )}

            {!isLoading && !error && !result && (
              <div className="h-full flex flex-col items-center justify-center text-slate-400 py-12">
                <Search className="w-12 h-12 mb-4 opacity-20" />
                <p>输入查询以查看检索结果</p>
              </div>
            )}

            {result && (
              <div className="space-y-6">
                {/* Hybrid Results */}
                {type === 'hybrid' && (
                  <div className="space-y-6">
                    <div className="bg-indigo-50 text-indigo-800 px-4 py-2 rounded-lg text-sm font-medium inline-block">
                      综合得分: {result.combined_score?.toFixed(4) || 'N/A'}
                    </div>
                    
                    <div className="grid md:grid-cols-2 gap-6">
                      <div>
                        <h3 className="text-sm font-semibold text-slate-900 mb-3 flex items-center gap-2">
                          <FileText className="w-4 h-4 text-indigo-500" /> 向量结果
                        </h3>
                        <pre className="bg-slate-900 text-slate-300 p-4 rounded-xl text-xs overflow-x-auto max-h-[400px] overflow-y-auto">
                          {JSON.stringify(result.vector_results, null, 2)}
                        </pre>
                      </div>
                      <div>
                        <h3 className="text-sm font-semibold text-slate-900 mb-3 flex items-center gap-2">
                          <Network className="w-4 h-4 text-emerald-500" /> 图谱结果
                        </h3>
                        <pre className="bg-slate-900 text-slate-300 p-4 rounded-xl text-xs overflow-x-auto max-h-[400px] overflow-y-auto">
                          {JSON.stringify(result.graph_results, null, 2)}
                        </pre>
                      </div>
                    </div>
                  </div>
                )}

                {/* Vector Results */}
                {type === 'vector' && (
                  <div className="space-y-4">
                    <h3 className="text-sm font-semibold text-slate-900">找到 {result.results?.length || 0} 个分块</h3>
                    <div className="grid gap-4">
                      {result.results?.map((item: any, idx: number) => (
                        <div key={idx} className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
                          <div className="flex justify-between items-center mb-2">
                            <span className="text-xs font-mono text-slate-500">ID: {item.id}</span>
                            <span className="text-xs font-medium bg-emerald-50 text-emerald-700 px-2 py-1 rounded-md">
                              Score: {item.score?.toFixed(4)}
                            </span>
                          </div>
                          <p className="text-sm text-slate-700">{item.content}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Graph Results */}
                {type === 'graph' && (
                  <div className="space-y-4">
                    <div className="bg-slate-900 rounded-xl p-4 shadow-sm">
                      <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">生成的Cypher</h3>
                      <code className="text-sm text-emerald-400 font-mono break-all">
                        {result.cypher}
                      </code>
                    </div>
                    
                    <h3 className="text-sm font-semibold text-slate-900 pt-4">图谱节点 ({result.results?.length || 0})</h3>
                    <div className="grid gap-4">
                      {result.results?.map((item: any, idx: number) => (
                        <div key={idx} className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm overflow-x-auto">
                          <pre className="text-xs text-slate-700 font-mono">
                            {JSON.stringify(item.d, null, 2)}
                          </pre>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
