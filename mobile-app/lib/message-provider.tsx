import React, { createContext, useContext, useState, useEffect, ReactNode } from "react";
import { Message, MessageStats, Classification } from "@/types/message";
import { loadMessages, saveMessages, addMessage as addMessageToStorage, deleteMessage as deleteMessageFromStorage, updateMessage as updateMessageInStorage, clearAllMessages } from "@/lib/storage";

interface MessageContextType {
  messages: Message[];
  stats: MessageStats;
  loading: boolean;
  addMessage: (message: Message) => Promise<void>;
  deleteMessage: (id: string) => Promise<void>;
  updateMessage: (id: string, updates: Partial<Message>) => Promise<void>;
  clearMessages: () => Promise<void>;
  refreshMessages: () => Promise<void>;
  getFilteredMessages: (filter: "all" | Classification) => Message[];
}

const MessageContext = createContext<MessageContextType | undefined>(undefined);

export function MessageProvider({ children }: { children: ReactNode }) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(true);

  const calculateStats = (msgs: Message[]): MessageStats => {
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    
    const todaySpamCount = msgs.filter((msg) => {
      const msgDate = new Date(msg.createdAt);
      msgDate.setHours(0, 0, 0, 0);
      return msg.classification === "SPAM" && msgDate.getTime() === today.getTime();
    }).length;

    return {
      totalMessages: msgs.length,
      inboxCount: msgs.filter((msg) => msg.classification === "INBOX").length,
      spamCount: msgs.filter((msg) => msg.classification === "SPAM").length,
      todaySpamCount,
    };
  };

  const stats = calculateStats(messages);

  const refreshMessages = async () => {
    try {
      setLoading(true);
      const loadedMessages = await loadMessages();
      setMessages(loadedMessages);
    } catch (error) {
      console.error("Error refreshing messages:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refreshMessages();
  }, []);

  const addMessage = async (message: Message) => {
    await addMessageToStorage(message);
    await refreshMessages();
  };

  const deleteMessage = async (id: string) => {
    await deleteMessageFromStorage(id);
    await refreshMessages();
  };

  const updateMessage = async (id: string, updates: Partial<Message>) => {
    await updateMessageInStorage(id, updates);
    await refreshMessages();
  };

  const clearMessages = async () => {
    await clearAllMessages();
    await refreshMessages();
  };

  const getFilteredMessages = (filter: "all" | Classification): Message[] => {
    if (filter === "all") return messages;
    return messages.filter((msg) => msg.classification === filter);
  };

  return (
    <MessageContext.Provider
      value={{
        messages,
        stats,
        loading,
        addMessage,
        deleteMessage,
        updateMessage,
        clearMessages,
        refreshMessages,
        getFilteredMessages,
      }}
    >
      {children}
    </MessageContext.Provider>
  );
}

export function useMessages() {
  const context = useContext(MessageContext);
  if (context === undefined) {
    throw new Error("useMessages must be used within a MessageProvider");
  }
  return context;
}
