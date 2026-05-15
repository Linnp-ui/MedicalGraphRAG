import React, { useState, useEffect } from 'react';
import { Network, Database, RefreshCw, AlertCircle } from 'lucide-react';
import { api, SchemaResponse } from '../lib/api';

export function SchemaView() {
  const [schema, setSchema] = useState<SchemaResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchSchema = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await api.getSchema();
      setSchema(data);
    } catch (err: any) {
      setError(err.message || '获取图谱结构失败');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchSchema();
  }, []);

  return (
    <div className="flex-1 overflow-y-auto bg-slate-50 p-8">
      <div className="max-w-4xl mx-auto space-y-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-slate-900 tracking-tight">医疗知识图谱结构</h1>
            <p className="text-slate-500 mt-1">Neo4j 医疗知识图谱结构概览。</p>
          </div>
          <button
            onClick={fetchSchema}
            disabled={isLoading}
            className="flex items-center gap-2 px-4 py-2 bg-white border border-slate-200 rounded-lg text-sm font-medium text-slate-700 hover:bg-slate-50 hover:text-indigo-600 transition-colors disabled:opacity-50 shadow-sm"
          >
            <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
            刷新
          </button>
        </div>

        {error && (
          <div className="bg-rose-50 border border-rose-200 rounded-xl p-4 flex items-start gap-3 text-rose-700">
            <AlertCircle className="w-5 h-5 shrink-0 mt-0.5" />
            <div>
              <h4 className="font-medium">加载图谱结构失败</h4>
              <p className="text-sm mt-1 text-rose-600/90">{error}</p>
            </div>
          </div>
        )}

        {schema && !isLoading && (
          <div className="grid md:grid-cols-2 gap-6">
            {/* Node Labels Card */}
            <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
              <div className="flex items-center gap-3 mb-6">
                <div className="w-10 h-10 rounded-full bg-indigo-50 flex items-center justify-center">
                  <Database className="w-5 h-5 text-indigo-600" />
                </div>
                <div>
                  <h2 className="text-lg font-semibold text-slate-900">医疗实体类型</h2>
                  <p className="text-sm text-slate-500">医疗知识图谱中的实体类型</p>
                </div>
              </div>
              
              <div className="flex flex-wrap gap-2">
                {schema.node_labels?.length > 0 ? (
                  schema.node_labels.map((label, idx) => (
                    <span key={idx} className="px-3 py-1.5 bg-indigo-50 text-indigo-700 border border-indigo-100 rounded-lg text-sm font-medium">
                      {label}
                    </span>
                  ))
                ) : (
                  <span className="text-sm text-slate-400 italic">未找到节点标签。</span>
                )}
              </div>
            </div>

            {/* Relationship Types Card */}
            <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
              <div className="flex items-center gap-3 mb-6">
                <div className="w-10 h-10 rounded-full bg-emerald-50 flex items-center justify-center">
                  <Network className="w-5 h-5 text-emerald-600" />
                </div>
                <div>
                  <h2 className="text-lg font-semibold text-slate-900">医疗关系类型</h2>
                  <p className="text-sm text-slate-500">医疗实体之间的连接关系</p>
                </div>
              </div>
              
              <div className="flex flex-wrap gap-2">
                {schema.relationship_types?.length > 0 ? (
                  schema.relationship_types.map((type, idx) => (
                    <span key={idx} className="px-3 py-1.5 bg-emerald-50 text-emerald-700 border border-emerald-100 rounded-lg text-sm font-medium font-mono">
                      {type}
                    </span>
                  ))
                ) : (
                  <span className="text-sm text-slate-400 italic">未找到关系类型。</span>
                )}
              </div>
            </div>

            {/* Raw Schema Output */}
            <div className="md:col-span-2 bg-slate-900 rounded-2xl shadow-sm border border-slate-800 p-6 overflow-hidden">
              <h2 className="text-sm font-semibold text-slate-300 mb-4 uppercase tracking-wider">原始医疗图谱结构输出</h2>
              <pre className="text-sm text-slate-400 font-mono whitespace-pre-wrap overflow-x-auto">
                {schema.schema}
              </pre>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
