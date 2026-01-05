export type ModelType = "BERT" | "RoBERTa" | "BigBird";

export type Classification = "INBOX" | "SPAM";

export interface Message {
  id: string;
  text: string;
  classification: Classification;
  confidence: number;
  model: ModelType;
  createdAt: string;
  updatedAt: string;
}

export interface MessageStats {
  totalMessages: number;
  inboxCount: number;
  spamCount: number;
  todaySpamCount: number;
}

export interface AnalyzeRequest {
  text: string;
  model: ModelType;
}

export interface AnalyzeResponse {
  classification: Classification;
  confidence: number;
  model: ModelType;
}
