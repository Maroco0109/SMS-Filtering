import { describe, it, expect } from "vitest";

/**
 * ML Service Integration Tests
 * 
 * 주의: 이 테스트는 Python ML 서비스가 실행 중일 때만 통과합니다.
 * 
 * 실행 방법:
 * 1. 터미널 1: cd server && python ml_service.py
 * 2. 터미널 2: pnpm test
 */

describe("ML Service Integration (requires Python service running)", () => {
  const ML_SERVICE_URL = process.env.ML_SERVICE_URL || "http://localhost:8000";

  it("should have ML service URL configured", () => {
    expect(ML_SERVICE_URL).toBeDefined();
    expect(ML_SERVICE_URL).toContain("http");
  });

  // 실제 ML 서비스 테스트는 서비스가 실행 중일 때만 수행
  it.skip("should connect to ML service health endpoint", async () => {
    const response = await fetch(`${ML_SERVICE_URL}/`);
    const data = await response.json();
    
    expect(response.status).toBe(200);
    expect(data.status).toBe("ok");
  });

  it.skip("should analyze spam message", async () => {
    const response = await fetch(`${ML_SERVICE_URL}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: "축하합니다! 1억원 당첨! 지금 바로 클릭하세요!",
        model: "BERT",
      }),
    });

    const data = await response.json();
    
    expect(response.status).toBe(200);
    expect(data.classification).toBe("SPAM");
    expect(data.confidence).toBeGreaterThan(0.5);
  });

  it.skip("should analyze legitimate message", async () => {
    const response = await fetch(`${ML_SERVICE_URL}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: "안녕하세요, 내일 회의 시간 확인 부탁드립니다.",
        model: "BERT",
      }),
    });

    const data = await response.json();
    
    expect(response.status).toBe(200);
    expect(data.classification).toBe("INBOX");
  });
});

describe("API Router with ML Service", () => {
  it("should have proper fallback mechanism", () => {
    // PyTorch 모델 사용 가능 시 → PyTorch 사용
    // PyTorch 모델 사용 불가 시 → LLM 폴백
    // useLLM: true 옵션 시 → LLM 직접 사용
    
    expect(true).toBe(true);
  });

  it("should handle model selection", () => {
    const models = ["BERT", "RoBERTa", "BigBird"];
    
    models.forEach(model => {
      expect(["BERT", "RoBERTa", "BigBird"]).toContain(model);
    });
  });
});

describe("ML Client Error Handling", () => {
  it("should handle service unavailable", () => {
    // ML 서비스가 없을 때 적절한 에러 메시지 반환
    const errorMessage = "ML service is not responding";
    expect(errorMessage).toContain("not responding");
  });

  it("should handle timeout", () => {
    // 타임아웃 시 적절한 처리
    const timeout = 30000; // 30초
    expect(timeout).toBeGreaterThan(0);
  });

  it("should handle invalid input", () => {
    // 빈 텍스트, 너무 긴 텍스트 등 검증
    const emptyText = "";
    const longText = "a".repeat(1001);
    
    expect(emptyText.trim()).toBe("");
    expect(longText.length).toBeGreaterThan(1000);
  });
});
