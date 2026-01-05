/**
 * ML Service Client
 * Python FastAPI 모델 서버와 통신하는 클라이언트
 */

import axios, { AxiosError } from "axios";

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || "http://localhost:8000";

export type ModelType = "BERT" | "RoBERTa" | "BigBird";
export type Classification = "SPAM" | "INBOX";

export interface MLAnalyzeRequest {
  text: string;
  model: ModelType;
}

export interface MLAnalyzeResponse {
  classification: Classification;
  confidence: number;
  model: string;
  reasoning: string;
}

export interface MLServiceError {
  detail: string;
}

/**
 * ML 서비스 헬스 체크
 */
export async function checkMLServiceHealth(): Promise<boolean> {
  try {
    const response = await axios.get(`${ML_SERVICE_URL}/`, {
      timeout: 5000,
    });
    return response.status === 200 && response.data.status === "ok";
  } catch (error) {
    console.error("ML service health check failed:", error);
    return false;
  }
}

/**
 * SMS 텍스트 분석 요청
 */
export async function analyzeText(
  text: string,
  model: ModelType
): Promise<MLAnalyzeResponse> {
  try {
    const response = await axios.post<MLAnalyzeResponse>(
      `${ML_SERVICE_URL}/analyze`,
      {
        text,
        model,
      } as MLAnalyzeRequest,
      {
        timeout: 30000, // 30초 타임아웃 (모델 로딩 시간 고려)
        headers: {
          "Content-Type": "application/json",
        },
      }
    );

    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      const axiosError = error as AxiosError<MLServiceError>;

      if (axiosError.response) {
        // 서버가 응답했지만 에러 상태
        const detail =
          axiosError.response.data?.detail || "ML service error";
        throw new Error(`ML Service Error: ${detail}`);
      } else if (axiosError.request) {
        // 요청은 보냈지만 응답 없음
        throw new Error(
          "ML service is not responding. Please ensure the Python service is running."
        );
      }
    }

    // 기타 에러
    throw new Error(`Failed to analyze text: ${(error as Error).message}`);
  }
}

/**
 * 사용 가능한 모델 목록 조회
 */
export async function listModels(): Promise<
  Array<{ name: string; huggingface_id: string; loaded: boolean }>
> {
  try {
    const response = await axios.get(`${ML_SERVICE_URL}/models`, {
      timeout: 5000,
    });
    return response.data.models;
  } catch (error) {
    console.error("Failed to list models:", error);
    return [];
  }
}

/**
 * ML 서비스 초기화 대기
 * 서버 시작 시 모델 로딩을 기다림
 */
export async function waitForMLService(
  maxRetries: number = 30,
  retryDelay: number = 2000
): Promise<boolean> {
  for (let i = 0; i < maxRetries; i++) {
    const isHealthy = await checkMLServiceHealth();
    if (isHealthy) {
      console.log("✅ ML service is ready");
      return true;
    }

    console.log(
      `⏳ Waiting for ML service... (${i + 1}/${maxRetries})`
    );
    await new Promise((resolve) => setTimeout(resolve, retryDelay));
  }

  console.error("❌ ML service failed to start");
  return false;
}
