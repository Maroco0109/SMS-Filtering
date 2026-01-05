/**
 * Statistics Utility Functions
 * 메시지 데이터 통계 계산 유틸리티
 */

import { Message } from "@/types/message";

export interface DailyStats {
  date: string; // YYYY-MM-DD
  spam: number;
  inbox: number;
  total: number;
}

export interface ModelStats {
  model: string;
  count: number;
  spamCount: number;
  inboxCount: number;
  avgConfidence: number;
}

export interface HourlyStats {
  hour: number; // 0-23
  spam: number;
  inbox: number;
}

export interface WeeklyTrend {
  weekday: string;
  spam: number;
  inbox: number;
}

/**
 * 일별 통계 계산
 */
export function calculateDailyStats(messages: Message[], days: number = 7): DailyStats[] {
  const now = new Date();
  const stats: Map<string, DailyStats> = new Map();

  // 최근 N일 초기화
  for (let i = 0; i < days; i++) {
    const date = new Date(now);
    date.setDate(date.getDate() - i);
    const dateStr = date.toISOString().split("T")[0];
    stats.set(dateStr, {
      date: dateStr,
      spam: 0,
      inbox: 0,
      total: 0,
    });
  }

  // 메시지 집계
  messages.forEach((msg) => {
    const dateStr = msg.createdAt.split("T")[0];
    const stat = stats.get(dateStr);
    if (stat) {
      stat.total++;
      if (msg.classification === "SPAM") {
        stat.spam++;
      } else {
        stat.inbox++;
      }
    }
  });

  // 날짜 순 정렬
  return Array.from(stats.values()).sort((a, b) => a.date.localeCompare(b.date));
}

/**
 * 모델별 통계 계산
 */
export function calculateModelStats(messages: Message[]): ModelStats[] {
  const stats: Map<string, ModelStats> = new Map();

  messages.forEach((msg) => {
    if (!stats.has(msg.model)) {
      stats.set(msg.model, {
        model: msg.model,
        count: 0,
        spamCount: 0,
        inboxCount: 0,
        avgConfidence: 0,
      });
    }

    const stat = stats.get(msg.model)!;
    stat.count++;
    if (msg.classification === "SPAM") {
      stat.spamCount++;
    } else {
      stat.inboxCount++;
    }
    stat.avgConfidence += msg.confidence;
  });

  // 평균 신뢰도 계산
  stats.forEach((stat) => {
    stat.avgConfidence = stat.avgConfidence / stat.count;
  });

  return Array.from(stats.values()).sort((a, b) => b.count - a.count);
}

/**
 * 시간대별 통계 계산
 */
export function calculateHourlyStats(messages: Message[]): HourlyStats[] {
  const stats: Map<number, HourlyStats> = new Map();

  // 0-23시 초기화
  for (let i = 0; i < 24; i++) {
    stats.set(i, { hour: i, spam: 0, inbox: 0 });
  }

  messages.forEach((msg) => {
    const date = new Date(msg.createdAt);
    const hour = date.getHours();
    const stat = stats.get(hour)!;

    if (msg.classification === "SPAM") {
      stat.spam++;
    } else {
      stat.inbox++;
    }
  });

  return Array.from(stats.values());
}

/**
 * 요일별 통계 계산
 */
export function calculateWeeklyTrend(messages: Message[]): WeeklyTrend[] {
  const weekdays = ["일", "월", "화", "수", "목", "금", "토"];
  const stats: Map<number, WeeklyTrend> = new Map();

  // 요일 초기화
  weekdays.forEach((day, index) => {
    stats.set(index, { weekday: day, spam: 0, inbox: 0 });
  });

  messages.forEach((msg) => {
    const date = new Date(msg.createdAt);
    const dayOfWeek = date.getDay();
    const stat = stats.get(dayOfWeek)!;

    if (msg.classification === "SPAM") {
      stat.spam++;
    } else {
      stat.inbox++;
    }
  });

  return Array.from(stats.values());
}

/**
 * 전체 통계 요약
 */
export interface OverallStats {
  totalMessages: number;
  totalSpam: number;
  totalInbox: number;
  spamRate: number;
  todaySpam: number;
  todayInbox: number;
  avgConfidence: number;
  mostUsedModel: string;
}

export function calculateOverallStats(messages: Message[]): OverallStats {
  const today = new Date().toISOString().split("T")[0];
  
  let totalSpam = 0;
  let totalInbox = 0;
  let todaySpam = 0;
  let todayInbox = 0;
  let totalConfidence = 0;
  const modelCount: Map<string, number> = new Map();

  messages.forEach((msg) => {
    const msgDate = msg.createdAt.split("T")[0];

    if (msg.classification === "SPAM") {
      totalSpam++;
      if (msgDate === today) todaySpam++;
    } else {
      totalInbox++;
      if (msgDate === today) todayInbox++;
    }

    totalConfidence += msg.confidence;
    modelCount.set(msg.model, (modelCount.get(msg.model) || 0) + 1);
  });

  // 가장 많이 사용된 모델
  let mostUsedModel = "BERT";
  let maxCount = 0;
  modelCount.forEach((count, model) => {
    if (count > maxCount) {
      maxCount = count;
      mostUsedModel = model;
    }
  });

  return {
    totalMessages: messages.length,
    totalSpam,
    totalInbox,
    spamRate: messages.length > 0 ? (totalSpam / messages.length) * 100 : 0,
    todaySpam,
    todayInbox,
    avgConfidence: messages.length > 0 ? totalConfidence / messages.length : 0,
    mostUsedModel,
  };
}

/**
 * 날짜 포맷팅 (MM/DD)
 */
export function formatDate(dateStr: string): string {
  const date = new Date(dateStr);
  return `${date.getMonth() + 1}/${date.getDate()}`;
}

/**
 * 시간 포맷팅 (HH시)
 */
export function formatHour(hour: number): string {
  return `${hour}시`;
}
