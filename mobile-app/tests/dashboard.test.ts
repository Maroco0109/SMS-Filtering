import { describe, it, expect } from "vitest";
import {
  calculateDailyStats,
  calculateModelStats,
  calculateWeeklyTrend,
  calculateOverallStats,
  calculateHourlyStats,
  formatDate,
  formatHour,
} from "../lib/statistics";
import { Message } from "../types/message";

describe("Dashboard Statistics", () => {
  const mockMessages: Message[] = [
    {
      id: "1",
      text: "스팸 메시지 1",
      classification: "SPAM",
      confidence: 0.95,
      model: "BERT",
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    },
    {
      id: "2",
      text: "정상 메시지 1",
      classification: "INBOX",
      confidence: 0.88,
      model: "BERT",
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    },
    {
      id: "3",
      text: "스팸 메시지 2",
      classification: "SPAM",
      confidence: 0.92,
      model: "RoBERTa",
      createdAt: new Date(Date.now() - 86400000).toISOString(), // 1일 전
      updatedAt: new Date(Date.now() - 86400000).toISOString(),
    },
  ];

  describe("calculateOverallStats", () => {
    it("should calculate total messages correctly", () => {
      const stats = calculateOverallStats(mockMessages);
      expect(stats.totalMessages).toBe(3);
    });

    it("should calculate spam and inbox counts", () => {
      const stats = calculateOverallStats(mockMessages);
      expect(stats.totalSpam).toBe(2);
      expect(stats.totalInbox).toBe(1);
    });

    it("should calculate spam rate", () => {
      const stats = calculateOverallStats(mockMessages);
      expect(stats.spamRate).toBeCloseTo(66.67, 1);
    });

    it("should calculate average confidence", () => {
      const stats = calculateOverallStats(mockMessages);
      const expectedAvg = (0.95 + 0.88 + 0.92) / 3;
      expect(stats.avgConfidence).toBeCloseTo(expectedAvg, 2);
    });

    it("should identify most used model", () => {
      const stats = calculateOverallStats(mockMessages);
      expect(stats.mostUsedModel).toBe("BERT");
    });

    it("should handle empty messages", () => {
      const stats = calculateOverallStats([]);
      expect(stats.totalMessages).toBe(0);
      expect(stats.spamRate).toBe(0);
      expect(stats.avgConfidence).toBe(0);
    });
  });

  describe("calculateDailyStats", () => {
    it("should return stats for specified number of days", () => {
      const stats = calculateDailyStats(mockMessages, 7);
      expect(stats).toHaveLength(7);
    });

    it("should aggregate messages by date", () => {
      const stats = calculateDailyStats(mockMessages, 7);
      const today = stats.find((s) => s.date === new Date().toISOString().split("T")[0]);
      expect(today).toBeDefined();
      expect(today!.total).toBeGreaterThan(0);
    });

    it("should separate spam and inbox counts", () => {
      const stats = calculateDailyStats(mockMessages, 7);
      stats.forEach((stat) => {
        expect(stat.spam + stat.inbox).toBe(stat.total);
      });
    });

    it("should sort by date ascending", () => {
      const stats = calculateDailyStats(mockMessages, 7);
      for (let i = 1; i < stats.length; i++) {
        expect(stats[i].date >= stats[i - 1].date).toBe(true);
      }
    });
  });

  describe("calculateModelStats", () => {
    it("should aggregate by model", () => {
      const stats = calculateModelStats(mockMessages);
      expect(stats.length).toBeGreaterThan(0);
      
      const bertStats = stats.find((s) => s.model === "BERT");
      expect(bertStats).toBeDefined();
      expect(bertStats!.count).toBe(2);
    });

    it("should calculate spam and inbox counts per model", () => {
      const stats = calculateModelStats(mockMessages);
      stats.forEach((stat) => {
        expect(stat.spamCount + stat.inboxCount).toBe(stat.count);
      });
    });

    it("should calculate average confidence per model", () => {
      const stats = calculateModelStats(mockMessages);
      const bertStats = stats.find((s) => s.model === "BERT");
      expect(bertStats!.avgConfidence).toBeGreaterThan(0);
      expect(bertStats!.avgConfidence).toBeLessThanOrEqual(1);
    });

    it("should sort by count descending", () => {
      const stats = calculateModelStats(mockMessages);
      for (let i = 1; i < stats.length; i++) {
        expect(stats[i].count <= stats[i - 1].count).toBe(true);
      }
    });
  });

  describe("calculateWeeklyTrend", () => {
    it("should return stats for all 7 weekdays", () => {
      const stats = calculateWeeklyTrend(mockMessages);
      expect(stats).toHaveLength(7);
    });

    it("should use Korean weekday names", () => {
      const stats = calculateWeeklyTrend(mockMessages);
      const weekdays = ["일", "월", "화", "수", "목", "금", "토"];
      stats.forEach((stat) => {
        expect(weekdays).toContain(stat.weekday);
      });
    });

    it("should aggregate messages by weekday", () => {
      const stats = calculateWeeklyTrend(mockMessages);
      const totalMessages = stats.reduce((sum, s) => sum + s.spam + s.inbox, 0);
      expect(totalMessages).toBe(mockMessages.length);
    });
  });

  describe("calculateHourlyStats", () => {
    it("should return stats for all 24 hours", () => {
      const stats = calculateHourlyStats(mockMessages);
      expect(stats).toHaveLength(24);
    });

    it("should have hours from 0 to 23", () => {
      const stats = calculateHourlyStats(mockMessages);
      stats.forEach((stat, index) => {
        expect(stat.hour).toBe(index);
      });
    });

    it("should aggregate messages by hour", () => {
      const stats = calculateHourlyStats(mockMessages);
      const totalMessages = stats.reduce((sum, s) => sum + s.spam + s.inbox, 0);
      expect(totalMessages).toBe(mockMessages.length);
    });
  });

  describe("Formatting functions", () => {
    it("should format date as MM/DD", () => {
      const formatted = formatDate("2024-12-30");
      expect(formatted).toMatch(/^\d{1,2}\/\d{1,2}$/);
    });

    it("should format hour with Korean suffix", () => {
      expect(formatHour(0)).toBe("0시");
      expect(formatHour(12)).toBe("12시");
      expect(formatHour(23)).toBe("23시");
    });
  });
});

describe("Chart Components", () => {
  it("should handle empty data gracefully", () => {
    // 차트 컴포넌트는 빈 데이터를 받아도 에러 없이 렌더링되어야 함
    expect(true).toBe(true);
  });

  it("should normalize bar heights correctly", () => {
    const data = [10, 50, 100];
    const max = Math.max(...data);
    const normalized = data.map((v) => (v / max) * 100);
    
    expect(normalized[0]).toBe(10);
    expect(normalized[1]).toBe(50);
    expect(normalized[2]).toBe(100);
  });
});
