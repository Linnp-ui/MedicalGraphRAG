import React, { useState, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { ChatView } from './components/ChatView';
import { IngestView } from './components/IngestView';
import { SchemaView } from './components/SchemaView';
import { RetrievalView } from './components/RetrievalView';
import { GraphView } from './components/GraphView';
import { api, HealthResponse } from './lib/api';
import { useChatHistory, Message, Conversation } from './lib/useChatHistory';

export default function App() {
  const [activeTab, setActiveTab] = useState('chat');
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [isCheckingHealth, setIsCheckingHealth] = useState(true);
  const chatHistory = useChatHistory();

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const data = await api.checkHealth();
        setHealth(data);
      } catch (error) {
        setHealth({ status: 'unhealthy', neo4j_connected: false, version: 'unknown' });
      } finally {
        setIsCheckingHealth(false);
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex h-screen w-full bg-white overflow-hidden font-sans text-slate-900">
      <Sidebar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        health={health}
        isCheckingHealth={isCheckingHealth}
        conversations={chatHistory.conversations}
        activeId={chatHistory.activeId}
        onSelectConversation={chatHistory.setActiveConversation}
        onNewConversation={chatHistory.createConversation}
        onDeleteConversation={chatHistory.deleteConversation}
      />

      <main className="flex-1 flex flex-col h-full overflow-hidden relative">
        {activeTab === 'chat' && (
          <ChatView
            messages={chatHistory.activeConversation?.messages || []}
            convId={chatHistory.activeId || ''}
            onAddMessage={chatHistory.addMessage}
          />
        )}
        {activeTab === 'ingest' && <IngestView />}
        {activeTab === 'schema' && <SchemaView />}
        {activeTab === 'graph' && <GraphView />}
        {activeTab === 'retrieval' && <RetrievalView />}
      </main>
    </div>
  );
}
