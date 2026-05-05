import React, { useState, useRef } from 'react';
import { Upload, File, CheckCircle2, AlertCircle, Loader2, X } from 'lucide-react';
import { api, IngestResponse } from '../lib/api';
import { cn } from '../lib/utils';

export function IngestView() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [extractEntities, setExtractEntities] = useState(true);
  const [createEmbeddings, setCreateEmbeddings] = useState(true);
  
  const [isIngesting, setIsIngesting] = useState(false);
  const [result, setResult] = useState<IngestResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setResult(null);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile) return;

    setIsIngesting(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('extract_entities', String(extractEntities));
      formData.append('create_embeddings', String(createEmbeddings));

      const response = await fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1'}/ingest/upload`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || '上传失败');
      }
      
      const data = await response.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message || '文档入库失败。');
    } finally {
      setIsIngesting(false);
    }
  };

  const clearFile = () => {
    setSelectedFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="flex-1 overflow-y-auto bg-slate-50 p-8">
      <div className="max-w-3xl mx-auto space-y-8">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 tracking-tight">知识入库</h1>
          <p className="text-slate-500 mt-1">上传文档以构建知识图谱和向量数据库。</p>
        </div>

        <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
          <div className="p-6">
            <form onSubmit={handleSubmit} className="space-y-6">
              {/* File Input */}
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1.5">
                  选择文件
                </label>
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleFileChange}
                  accept=".pdf,.docx,.doc,.txt,.csv"
                  className="hidden"
                />
                {!selectedFile ? (
                  <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    className="w-full flex flex-col items-center justify-center gap-3 px-4 py-8 bg-slate-50 border-2 border-dashed border-slate-200 rounded-lg hover:border-indigo-400 hover:bg-slate-100 transition-all cursor-pointer"
                  >
                    <Upload className="w-10 h-10 text-slate-400" />
                    <div className="text-center">
                      <p className="text-sm font-medium text-slate-700">点击上传文件</p>
                      <p className="text-xs text-slate-500 mt-1">PDF, DOCX, TXT, CSV (最大 10MB)</p>
                    </div>
                  </button>
                ) : (
                  <div className="flex items-center justify-between px-4 py-3 bg-indigo-50 border border-indigo-200 rounded-lg">
                    <div className="flex items-center gap-3">
                      <File className="w-5 h-5 text-indigo-600" />
                      <span className="text-sm font-medium text-slate-700">{selectedFile.name}</span>
                      <span className="text-xs text-slate-500">({(selectedFile.size / 1024).toFixed(1)} KB)</span>
                    </div>
                    <button
                      type="button"
                      onClick={clearFile}
                      className="p-1 text-slate-400 hover:text-slate-600 transition-colors"
                    >
                      <X className="w-5 h-5" />
                    </button>
                  </div>
                )}
              </div>

              {/* Processing Options */}
              <div className="space-y-3 pt-4 border-t border-slate-100">
                <h3 className="text-sm font-medium text-slate-900">处理选项</h3>
                
                <label className="flex items-start gap-3 cursor-pointer group">
                  <div className="flex items-center h-5">
                    <input
                      type="checkbox"
                      checked={extractEntities}
                      onChange={(e) => setExtractEntities(e.target.checked)}
                      className="w-4 h-4 text-indigo-600 border-slate-300 rounded focus:ring-indigo-500"
                    />
                  </div>
                  <div className="flex flex-col">
                    <span className="text-sm font-medium text-slate-700 group-hover:text-slate-900">提取实体与关系</span>
                    <span className="text-xs text-slate-500">使用LLM构建知识图谱结构。（较慢，质量更高）</span>
                  </div>
                </label>

                <label className="flex items-start gap-3 cursor-pointer group">
                  <div className="flex items-center h-5">
                    <input
                      type="checkbox"
                      checked={createEmbeddings}
                      onChange={(e) => setCreateEmbeddings(e.target.checked)}
                      className="w-4 h-4 text-indigo-600 border-slate-300 rounded focus:ring-indigo-500"
                    />
                  </div>
                  <div className="flex flex-col">
                    <span className="text-sm font-medium text-slate-700 group-hover:text-slate-900">创建向量嵌入</span>
                    <span className="text-xs text-slate-500">生成向量用于相似度搜索。（较快，混合搜索必需）</span>
                  </div>
                </label>
              </div>

              <div className="pt-4">
                <button
                  type="submit"
                  disabled={isIngesting || !selectedFile}
                  className="w-full flex items-center justify-center gap-2 bg-indigo-600 hover:bg-indigo-700 text-white px-6 py-3 rounded-xl font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed shadow-sm"
                >
                  {isIngesting ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin" />
                      处理中...
                    </>
                  ) : (
                    <>
                      <Upload className="w-5 h-5" />
                      开始入库
                    </>
                  )}
                </button>
              </div>
            </form>
          </div>
        </div>

        {/* Results Area */}
        {error && (
          <div className="bg-rose-50 border border-rose-200 rounded-xl p-4 flex items-start gap-3 text-rose-700">
            <AlertCircle className="w-5 h-5 shrink-0 mt-0.5" />
            <div>
              <h4 className="font-medium">入库失败</h4>
              <p className="text-sm mt-1 text-rose-600/90">{error}</p>
            </div>
          </div>
        )}

        {result && (
          <div className="bg-emerald-50 border border-emerald-200 rounded-xl p-6 space-y-4">
            <div className="flex items-center gap-3 text-emerald-800">
              <CheckCircle2 className="w-6 h-6 text-emerald-600" />
              <h3 className="text-lg font-semibold">入库完成</h3>
            </div>
            
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-white rounded-lg p-4 border border-emerald-100 shadow-sm">
                <div className="text-sm text-slate-500 mb-1">状态</div>
                <div className="font-medium text-slate-900 capitalize">{result.status}</div>
              </div>
              <div className="bg-white rounded-lg p-4 border border-emerald-100 shadow-sm">
                <div className="text-sm text-slate-500 mb-1">处理文档数</div>
                <div className="font-medium text-slate-900">{result.documents_processed}</div>
              </div>
            </div>

            {result.results && result.results.length > 0 && (
              <div className="mt-6">
                <h4 className="text-sm font-medium text-slate-900 mb-3">详细结果</h4>
                <div className="bg-white rounded-lg border border-emerald-100 overflow-hidden shadow-sm">
                  <table className="w-full text-sm text-left">
                    <thead className="bg-slate-50 border-b border-slate-200 text-slate-600">
                      <tr>
                        <th className="px-4 py-3 font-medium">文档ID</th>
                        <th className="px-4 py-3 font-medium text-center">分块数</th>
                        <th className="px-4 py-3 font-medium text-center">实体数</th>
                        <th className="px-4 py-3 font-medium text-center">关系数</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      {result.results.map((item, idx) => (
                        <tr key={idx} className="hover:bg-slate-50/50">
                          <td className="px-4 py-3 font-mono text-xs text-slate-600 max-w-[180px] truncate">{item.document_id}</td>
                          <td className="px-4 py-3 text-center text-slate-700">{item.chunks_created}</td>
                          <td className="px-4 py-3 text-center text-slate-700">{item.entities_extracted}</td>
                          <td className="px-4 py-3 text-center text-slate-700">{item.relationships_created}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
