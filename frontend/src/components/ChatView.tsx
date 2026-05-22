import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { Send, Bot, User, Loader2, ChevronDown, ChevronUp, FileText, Network, Layers, Search } from 'lucide-react';
import { api, QueryResponse } from '../lib/api';
import { cn } from '../lib/utils';
import { Message } from '../lib/useChatHistory';

interface ChatViewProps {
  messages: Message[];
  convId: string;
  onAddMessage: (convId: string, message: Message) => void;
}

export function ChatView({ messages, convId, onAddMessage }: ChatViewProps) {
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [useHybrid, setUseHybrid] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Auto-resize textarea
  useEffect(() => {
    const ta = textareaRef.current;
    if (ta) {
      ta.style.height = 'auto';
      ta.style.height = `${Math.min(ta.scrollHeight, 160)}px`;
    }
  }, [input]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading || !convId) return;

    const userMsg: Message = { id: Date.now().toString(), role: 'user', content: input };
    onAddMessage(convId, userMsg);
    setInput('');
    setIsLoading(true);

    try {
      // Build history from existing messages (exclude welcome msg and current user msg)
      const history = messages
        .filter(m => m.id !== 'welcome' && m.id !== userMsg.id)
        .map(m => ({ role: m.role, content: m.content }));

      const response = await api.query({ question: userMsg.content, use_hybrid: useHybrid, history });
      onAddMessage(convId, {
        id: (Date.now() + 1).toString(),
        role: 'bot',
        content: response.answer,
        response: response,
      });
    } catch (error: any) {
      onAddMessage(convId, {
        id: (Date.now() + 1).toString(),
        role: 'bot',
        content: `**错误**：${error.message || '无法连接到API。请确保后端服务运行在 localhost:8000。'}`,
        isError: true,
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e as any);
    }
  };

  return (
    <div className="flex flex-col h-full bg-slate-50">
      {/* Header */}
      <div className="px-6 py-4 bg-white border-b border-slate-200 flex justify-between items-center shadow-sm z-10">
        <div>
          <h2 className="text-lg font-semibold text-slate-800">医疗问答</h2>
          <p className="text-sm text-slate-500">基于医疗知识图谱回答问题</p>
        </div>
        <div className="flex items-center gap-2 bg-slate-100 p-1 rounded-lg border border-slate-200">
          <button
            onClick={() => setUseHybrid(true)}
            className={cn("px-3 py-1.5 text-xs font-medium rounded-md transition-all", useHybrid ? "bg-white shadow-sm text-indigo-600" : "text-slate-500 hover:text-slate-700")}
          >
            混合搜索
          </button>
          <button
            onClick={() => setUseHybrid(false)}
            className={cn("px-3 py-1.5 text-xs font-medium rounded-md transition-all", !useHybrid ? "bg-white shadow-sm text-indigo-600" : "text-slate-500 hover:text-slate-700")}
          >
            仅图谱
          </button>
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {messages.map((msg) => (
          <div key={msg.id} className={cn("flex gap-4 max-w-4xl mx-auto", msg.role === 'user' ? "flex-row-reverse" : "")}>
            <div className={cn(
              "w-8 h-8 rounded-full flex items-center justify-center shrink-0 mt-1",
              msg.role === 'user' ? "bg-indigo-600 text-white" : "bg-emerald-500 text-white"
            )}>
              {msg.role === 'user' ? <User className="w-5 h-5" /> : <Bot className="w-5 h-5" />}
            </div>

            <div className={cn(
              "flex flex-col gap-2 max-w-[80%]",
              msg.role === 'user' ? "items-end" : "items-start"
            )}>
              <div className={cn(
                "px-4 py-3 rounded-2xl text-sm leading-relaxed shadow-sm",
                msg.role === 'user'
                  ? "bg-indigo-600 text-white rounded-tr-sm"
                  : msg.isError
                    ? "bg-rose-50 text-rose-700 border border-rose-200 rounded-tl-sm"
                    : "bg-white text-slate-800 border border-slate-200 rounded-tl-sm"
              )}>
                {msg.role === 'user' ? (
                  // User messages: plain text
                  <span>{msg.content}</span>
                ) : (
                  // Bot messages: render Markdown
                  <MarkdownContent content={msg.content} isError={msg.isError} />
                )}
              </div>

              {/* Sources & Metadata (Bot only) */}
              {msg.response && (
                <MessageMetadata response={msg.response} />
              )}
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="flex gap-4 max-w-4xl mx-auto">
            <div className="w-8 h-8 rounded-full bg-emerald-500 text-white flex items-center justify-center shrink-0 mt-1">
              <Bot className="w-5 h-5" />
            </div>
            <div className="px-4 py-3 rounded-2xl bg-white border border-slate-200 rounded-tl-sm shadow-sm flex items-center gap-2 text-slate-500 text-sm">
              <Loader2 className="w-4 h-4 animate-spin" />
              思考中...
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="p-4 bg-white border-t border-slate-200">
        <form onSubmit={handleSubmit} className="max-w-4xl mx-auto relative flex items-end gap-2">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="输入关于医疗知识的问题... (Enter 发送，Shift+Enter 换行)"
            rows={1}
            className="flex-1 pl-4 pr-4 py-3 bg-slate-50 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500 transition-all text-sm resize-none overflow-hidden"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={!input.trim() || isLoading}
            className="p-3 bg-indigo-600 text-white rounded-xl hover:bg-indigo-700 disabled:opacity-40 disabled:hover:bg-indigo-600 transition-all shrink-0"
          >
            <Send className="w-4 h-4" />
          </button>
        </form>
        <p className="text-center text-xs text-slate-400 mt-2">Enter 发送 · Shift+Enter 换行</p>
      </div>
    </div>
  );
}

// ─── Markdown renderer ────────────────────────────────────────────────────────

interface MarkdownContentProps {
  content: string;
  isError?: boolean;
}

function MarkdownContent({ content, isError }: MarkdownContentProps) {
  return (
    <ReactMarkdown
      components={{
        // Paragraphs — no default margin, keep tight
        p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,

        // Headings
        h1: ({ children }) => <h1 className="text-base font-bold mb-2 mt-3 first:mt-0">{children}</h1>,
        h2: ({ children }) => <h2 className="text-sm font-bold mb-1.5 mt-3 first:mt-0">{children}</h2>,
        h3: ({ children }) => <h3 className="text-sm font-semibold mb-1 mt-2 first:mt-0">{children}</h3>,

        // Lists
        ul: ({ children }) => <ul className="list-disc list-inside space-y-0.5 mb-2 pl-1">{children}</ul>,
        ol: ({ children }) => <ol className="list-decimal list-inside space-y-0.5 mb-2 pl-1">{children}</ol>,
        li: ({ children }) => <li className="text-sm">{children}</li>,

        // Inline code
        code: ({ inline, children, ...props }: any) =>
          inline ? (
            <code
              className={cn(
                'px-1 py-0.5 rounded text-xs font-mono',
                isError ? 'bg-rose-100' : 'bg-slate-100 text-slate-700'
              )}
              {...props}
            >
              {children}
            </code>
          ) : (
            // Code block
            <pre
              className={cn(
                'my-2 p-3 rounded-lg text-xs font-mono overflow-x-auto',
                isError ? 'bg-rose-100' : 'bg-slate-100 text-slate-800'
              )}
            >
              <code {...props}>{children}</code>
            </pre>
          ),

        // Blockquote
        blockquote: ({ children }) => (
          <blockquote className="border-l-2 border-slate-300 pl-3 italic text-slate-600 my-2">
            {children}
          </blockquote>
        ),

        // Horizontal rule
        hr: () => <hr className="my-3 border-slate-200" />,

        // Bold / italic
        strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
        em: ({ children }) => <em className="italic">{children}</em>,

        // Links — open in new tab, prevent XSS
        a: ({ href, children }) => (
          <a
            href={href}
            target="_blank"
            rel="noopener noreferrer"
            className="text-indigo-600 underline underline-offset-2 hover:text-indigo-800"
          >
            {children}
          </a>
        ),

        // Tables
        table: ({ children }) => (
          <div className="overflow-x-auto my-2">
            <table className="text-xs border-collapse w-full">{children}</table>
          </div>
        ),
        thead: ({ children }) => <thead className="bg-slate-100">{children}</thead>,
        th: ({ children }) => (
          <th className="border border-slate-200 px-2 py-1 text-left font-semibold">{children}</th>
        ),
        td: ({ children }) => (
          <td className="border border-slate-200 px-2 py-1">{children}</td>
        ),
      }}
    >
      {content}
    </ReactMarkdown>
  );
}

// ─── Message metadata (routing / sources) ────────────────────────────────────

function MessageMetadata({ response }: { response: QueryResponse }) {
  const [expanded, setExpanded] = useState(false);

  const getRoutingIcon = (routing: string) => {
    switch (routing.toLowerCase()) {
      case 'hybrid': return <Layers className="w-3 h-3" />;
      case 'graph': return <Network className="w-3 h-3" />;
      case 'vector': return <FileText className="w-3 h-3" />;
      default: return <Search className="w-3 h-3" />;
    }
  };

  // 解析来源内容为结构化显示
  const parseSourceContent = (content: string) => {
    // 检查是否是实体摘要格式："实体【xxx】(类型): ..."
    const entityMatch = content.match(/^实体【([^】]+)】\s*\(([^)]+)\):\s*(.*)$/);
    if (entityMatch) {
      return {
        type: "entity",
        name: entityMatch[1],
        entityType: entityMatch[2],
        content: entityMatch[3]
      };
    }
    // 兼容旧格式："实体[xxx] (类型): ..."
    const entityMatchOld = content.match(/^实体\[([^\]]+)\]\s*\(([^)]+)\):\s*(.*)$/);
    if (entityMatchOld) {
      return {
        type: "entity",
        name: entityMatchOld[1],
        entityType: entityMatchOld[2],
        content: entityMatchOld[3]
      };
    }
    return { type: "text", content };
  };

  // 解析摘要内容
  const parseSummary = (summary: string) => {
    const parts = summary.split("|").map(p => p.trim());
    const result: { [key: string]: string } = {};
    
    parts.forEach(part => {
      const colonIndex = part.indexOf(":");
      if (colonIndex > 0) {
        const key = part.substring(0, colonIndex).trim();
        const value = part.substring(colonIndex + 1).trim();
        if (key && value) {
          result[key] = value;
        }
      }
    });
    
    return result;
  };

  return (
    <div className="w-full mt-1">
      <div className="flex items-center gap-3 text-xs text-slate-500 mb-2 px-1">
        <div className="flex items-center gap-1.5 bg-slate-100 px-2 py-1 rounded-md border border-slate-200">
          {getRoutingIcon(response.routing)}
          <span className="font-medium capitalize">{response.routing} 路由</span>
        </div>
        <div className="flex items-center gap-1.5">
          <FileText className="w-3 h-3" />
          <span>{response.documents_count} 篇文档</span>
        </div>
        {response.sources && response.sources.length > 0 && (
          <button
            onClick={() => setExpanded(!expanded)}
            className="ml-auto flex items-center gap-1 hover:text-indigo-600 transition-colors"
          >
            {expanded ? '收起来源' : '查看来源'}
            {expanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
          </button>
        )}
      </div>

      {expanded && response.sources && response.sources.length > 0 && (
        <div className="space-y-2 mt-2">
          {response.sources.map((source, idx) => {
            const parsed = parseSourceContent(source.content);
            return (
              <div key={idx} className="bg-slate-50 border border-slate-200 rounded-lg p-3 text-xs text-slate-600 shadow-sm">
                <div className="flex justify-between items-center mb-2 pb-2 border-b border-slate-200/60">
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-slate-700">来源 {idx + 1}</span>
                    {parsed.type === "entity" && (
                      <span className="px-1.5 py-0.5 bg-indigo-50 text-indigo-600 rounded text-[0.65rem] border border-indigo-100">
                        {parsed.entityType}
                      </span>
                    )}
                  </div>
                  <span className="text-emerald-600 font-mono bg-emerald-50 px-1.5 py-0.5 rounded">
                    分数: {typeof source.score === 'number' ? source.score.toFixed(3) : source.score}
                  </span>
                </div>
                
                {parsed.type === "entity" ? (
                  <div className="space-y-1.5">
                    <div className="font-medium text-slate-700 text-sm">{parsed.name}</div>
                    {(() => {
                      const summary = parseSummary(parsed.content);
                      return (
                        <div className="space-y-1">
                          {summary["类型"] && (
                            <div className="flex items-center gap-1.5">
                              <span className="text-slate-400">类型:</span>
                              <span className="text-slate-600">{summary["类型"]}</span>
                            </div>
                          )}
                          {summary["属性"] && (
                            <div className="flex items-start gap-1.5">
                              <span className="text-slate-400 shrink-0">属性:</span>
                              <span className="text-slate-600">{summary["属性"]}</span>
                            </div>
                          )}
                          {summary["关系"] && (
                            <div className="flex items-start gap-1.5">
                              <span className="text-slate-400 shrink-0">关系:</span>
                              <span className="text-slate-600">{summary["关系"]}</span>
                            </div>
                          )}
                          {/* 显示剩余内容 */}
                          {!summary["类型"] && !summary["属性"] && !summary["关系"] && (
                            <p className="text-slate-600">{parsed.content}</p>
                          )}
                        </div>
                      );
                    })()}
                  </div>
                ) : (
                  <p className="line-clamp-3 hover:line-clamp-none transition-all">{source.content}</p>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
