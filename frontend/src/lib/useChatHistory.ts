import { useState, useEffect, useCallback } from 'react';
import { QueryResponse } from './api';

export interface Message {
  id: string;
  role: 'user' | 'bot';
  content: string;
  response?: QueryResponse;
  isError?: boolean;
}

export interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  createdAt: number;
  updatedAt: number;
}

const STORAGE_KEY = 'graphrag_conversations';
const ACTIVE_KEY = 'graphrag_active_conversation';

function loadConversations(): Conversation[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    return JSON.parse(raw);
  } catch {
    return [];
  }
}

function saveConversations(conversations: Conversation[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
}

function loadActiveId(): string | null {
  return localStorage.getItem(ACTIVE_KEY);
}

function saveActiveId(id: string | null) {
  if (id) {
    localStorage.setItem(ACTIVE_KEY, id);
  } else {
    localStorage.removeItem(ACTIVE_KEY);
  }
}

function generateTitle(messages: Message[]): string {
  const firstUser = messages.find(m => m.role === 'user');
  if (!firstUser) return '新对话';
  return firstUser.content.length > 30
    ? firstUser.content.slice(0, 30) + '...'
    : firstUser.content;
}

const WELCOME_MSG: Message = {
  id: 'welcome',
  role: 'bot',
  content: '你好！我是 GraphRAG 助手。请基于知识图谱向我提问。',
};

export function useChatHistory() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);

  // Load from localStorage on mount
  useEffect(() => {
    const stored = loadConversations();
    setConversations(stored);
    const storedActive = loadActiveId();
    if (storedActive && stored.find(c => c.id === storedActive)) {
      setActiveId(storedActive);
    } else if (stored.length > 0) {
      setActiveId(stored[0].id);
    } else {
      // Create initial conversation
      const initial: Conversation = {
        id: Date.now().toString(),
        title: '新对话',
        messages: [WELCOME_MSG],
        createdAt: Date.now(),
        updatedAt: Date.now(),
      };
      setConversations([initial]);
      setActiveId(initial.id);
      saveConversations([initial]);
      saveActiveId(initial.id);
    }
  }, []);

  // Persist conversations whenever they change
  useEffect(() => {
    if (conversations.length > 0) {
      saveConversations(conversations);
    }
  }, [conversations]);

  // Persist active id
  useEffect(() => {
    saveActiveId(activeId);
  }, [activeId]);

  const activeConversation = conversations.find(c => c.id === activeId) || null;

  const setActiveConversation = useCallback((id: string) => {
    setActiveId(id);
  }, []);

  const createConversation = useCallback(() => {
    const newConv: Conversation = {
      id: Date.now().toString(),
      title: '新对话',
      messages: [WELCOME_MSG],
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };
    setConversations(prev => [newConv, ...prev]);
    setActiveId(newConv.id);
    return newConv.id;
  }, []);

  const deleteConversation = useCallback((id: string) => {
    setConversations(prev => {
      const filtered = prev.filter(c => c.id !== id);
      if (filtered.length === 0) {
        // Create a new empty conversation
        const newConv: Conversation = {
          id: Date.now().toString(),
          title: '新对话',
          messages: [WELCOME_MSG],
          createdAt: Date.now(),
          updatedAt: Date.now(),
        };
        setActiveId(newConv.id);
        return [newConv];
      }
      if (activeId === id) {
        setActiveId(filtered[0].id);
      }
      return filtered;
    });
  }, [activeId]);

  const addMessage = useCallback((convId: string, message: Message) => {
    setConversations(prev => prev.map(c => {
      if (c.id !== convId) return c;
      const updatedMessages = [...c.messages, message];
      return {
        ...c,
        messages: updatedMessages,
        title: c.messages.length <= 1 ? generateTitle(updatedMessages) : c.title,
        updatedAt: Date.now(),
      };
    }));
  }, []);

  const clearAll = useCallback(() => {
    const newConv: Conversation = {
      id: Date.now().toString(),
      title: '新对话',
      messages: [WELCOME_MSG],
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };
    setConversations([newConv]);
    setActiveId(newConv.id);
  }, []);

  return {
    conversations,
    activeId,
    activeConversation,
    setActiveConversation,
    createConversation,
    deleteConversation,
    addMessage,
    clearAll,
  };
}
