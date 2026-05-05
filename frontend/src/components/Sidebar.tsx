import React from 'react';
import { MessageSquare, Database, Network, Search, Activity, Loader2, Plus, Trash2 } from 'lucide-react';
import { cn } from '../lib/utils';
import { HealthResponse } from '../lib/api';
import { Conversation } from '../lib/useChatHistory';

interface SidebarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  health: HealthResponse | null;
  isCheckingHealth: boolean;
  conversations: Conversation[];
  activeId: string | null;
  onSelectConversation: (id: string) => void;
  onNewConversation: () => void;
  onDeleteConversation: (id: string) => void;
}

export function Sidebar({
  activeTab,
  setActiveTab,
  health,
  isCheckingHealth,
  conversations,
  activeId,
  onSelectConversation,
  onNewConversation,
  onDeleteConversation,
}: SidebarProps) {
  const tabs = [
    { id: 'chat', label: '问答对话', icon: MessageSquare },
    { id: 'ingest', label: '知识入库', icon: Database },
    { id: 'schema', label: '图谱结构', icon: Network },
    { id: 'graph', label: '图谱可视化', icon: Activity },
    { id: 'retrieval', label: '检索测试', icon: Search },
  ];

  return (
    <div className="w-64 bg-slate-900 text-slate-300 flex flex-col h-screen border-r border-slate-800">
      <div className="p-6 flex items-center gap-3 border-b border-slate-800">
        <div className="w-8 h-8 rounded-lg bg-indigo-500/20 flex items-center justify-center">
          <Network className="w-5 h-5 text-indigo-400" />
        </div>
        <span className="text-xl font-bold text-white tracking-tight">GraphRAG</span>
      </div>

      <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
        {/* Tab buttons */}
        {tabs.map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={cn(
                "w-full flex items-center gap-3 px-3 py-2.5 rounded-md text-sm font-medium transition-colors",
                isActive
                  ? "bg-indigo-500/10 text-indigo-400"
                  : "hover:bg-slate-800 hover:text-white text-slate-400"
              )}
            >
              <Icon className="w-4 h-4" />
              {tab.label}
            </button>
          );
        })}

        {/* Conversation list (only show on chat tab) */}
        {activeTab === 'chat' && (
          <div className="mt-4 pt-4 border-t border-slate-800">
            <div className="flex items-center justify-between px-3 mb-2">
              <span className="text-xs font-medium text-slate-500 uppercase tracking-wider">对话历史</span>
              <button
                onClick={onNewConversation}
                className="p-1 text-slate-500 hover:text-indigo-400 transition-colors"
                title="新对话"
              >
                <Plus className="w-3.5 h-3.5" />
              </button>
            </div>
            <div className="space-y-0.5">
              {conversations.map((conv) => {
                const isActive = conv.id === activeId;
                return (
                  <div
                    key={conv.id}
                    className={cn(
                      "group flex items-center gap-2 px-3 py-2 rounded-md text-sm transition-colors cursor-pointer",
                      isActive
                        ? "bg-slate-800 text-white"
                        : "text-slate-400 hover:bg-slate-800/60 hover:text-slate-300"
                    )}
                    onClick={() => onSelectConversation(conv.id)}
                  >
                    <MessageSquare className="w-3.5 h-3.5 shrink-0" />
                    <span className="truncate flex-1 text-xs">{conv.title}</span>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onDeleteConversation(conv.id);
                      }}
                      className="opacity-0 group-hover:opacity-100 p-0.5 text-slate-500 hover:text-rose-400 transition-all"
                    >
                      <Trash2 className="w-3 h-3" />
                    </button>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </nav>

      <div className="p-4 border-t border-slate-800 bg-slate-900/50">
        <div className="flex items-center justify-between text-xs">
          <div className="flex items-center gap-2">
            {isCheckingHealth ? (
              <Loader2 className="w-3 h-3 animate-spin text-slate-500" />
            ) : (
              <div className={cn(
                "w-2 h-2 rounded-full",
                health?.status === 'healthy' ? "bg-emerald-500" : "bg-rose-500 shadow-[0_0_8px_rgba(244,63,94,0.5)]"
              )} />
            )}
            <span className="text-slate-400 font-medium">
              {isCheckingHealth ? '检查中...' : (health?.status === 'healthy' ? '系统在线' : '系统离线')}
            </span>
          </div>
          {health?.version && (
            <span className="text-slate-500 font-mono">v{health.version}</span>
          )}
        </div>
        {health?.status === 'healthy' && (
          <div className="mt-2 text-[10px] text-slate-500 flex items-center gap-1.5">
            <Database className="w-3 h-3" />
            Neo4j: {health.neo4j_connected ? <span className="text-emerald-400">已连接</span> : <span className="text-rose-400">未连接</span>}
          </div>
        )}
      </div>
    </div>
  );
}
