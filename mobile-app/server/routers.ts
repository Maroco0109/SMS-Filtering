import { COOKIE_NAME } from "../shared/const.js";
import { getSessionCookieOptions } from "./_core/cookies";
import { systemRouter } from "./_core/systemRouter";
import { publicProcedure, router } from "./_core/trpc";
import { invokeLLM } from "./_core/llm";
import { z } from "zod";
import { analyzeText, checkMLServiceHealth } from "./ml_client";

export const appRouter = router({
  // if you need to use socket.io, read and register route in server/_core/index.ts, all api should start with '/api/' so that the gateway can route correctly
  system: systemRouter,
  auth: router({
    me: publicProcedure.query((opts) => opts.ctx.user),
    logout: publicProcedure.mutation(({ ctx }) => {
      const cookieOptions = getSessionCookieOptions(ctx.req);
      ctx.res.clearCookie(COOKIE_NAME, { ...cookieOptions, maxAge: -1 });
      return {
        success: true,
      } as const;
    }),
  }),

  // SMS 분석 API
  sms: router({
    // ML 서비스 헬스 체크
    health: publicProcedure.query(async () => {
      const isHealthy = await checkMLServiceHealth();
      return {
        status: isHealthy ? "ok" : "unavailable",
        service: "ML Service (Python FastAPI)",
      };
    }),

    // SMS 분석 (PyTorch 모델 사용)
    analyze: publicProcedure
      .input(
        z.object({
          text: z.string().min(1, "메시지 내용이 필요합니다").max(1000, "메시지가 너무 깁니다"),
          model: z.enum(["BERT", "RoBERTa", "BigBird"]).default("BERT"),
          useLLM: z.boolean().optional().default(false), // LLM 폴백 옵션
        })
      )
      .mutation(async ({ input }) => {
        const { text, model, useLLM } = input;

        // ML 서비스 사용 가능 여부 확인
        const mlServiceAvailable = !useLLM && (await checkMLServiceHealth());

        if (mlServiceAvailable) {
          // PyTorch 모델 사용
          try {
            console.log(`Using PyTorch model: ${model}`);
            const result = await analyzeText(text, model);
            return {
              classification: result.classification,
              confidence: result.confidence,
              model: result.model,
              reasoning: result.reasoning,
              source: "pytorch" as const,
            };
          } catch (error) {
            console.error("PyTorch 모델 오류, LLM으로 폴백:", error);
            // LLM으로 폴백
          }
        }

        // LLM 폴백 또는 명시적 LLM 사용
        console.log(`Using LLM fallback for model: ${model}`);
        const systemPrompt = `You are an expert SMS spam classifier. Analyze the given SMS message and determine if it is SPAM or INBOX (legitimate).

Classification Guidelines:
- SPAM: Promotional messages, phishing attempts, suspicious links, unsolicited marketing, scams, lottery/prize notifications
- INBOX: Personal messages, legitimate notifications, OTP codes, delivery updates, appointment reminders

Respond ONLY with a JSON object in this exact format:
{
  "classification": "SPAM" or "INBOX",
  "confidence": 0.0 to 1.0,
  "reasoning": "brief explanation"
}`;

        const userPrompt = `Analyze this SMS message using ${model} classification approach:

"${text}"

Classify as SPAM or INBOX and provide confidence score.`;

        try {
          const response = await invokeLLM({
            messages: [
              { role: "system", content: systemPrompt },
              { role: "user", content: userPrompt },
            ],
            response_format: { type: "json_object" },
          });

          const content = response.choices[0].message.content;
          if (typeof content !== "string") {
            throw new Error("잘못된 응답 형식");
          }
          const result = JSON.parse(content);

          return {
            classification: result.classification as "SPAM" | "INBOX",
            confidence: Math.min(Math.max(result.confidence, 0), 1),
            model: model,
            reasoning: result.reasoning,
            source: "llm" as const,
          };
        } catch (error) {
          console.error("분석 오류:", error);
          throw new Error("메시지 분석 중 오류가 발생했습니다");
        }
      }),
  }),
});

export type AppRouter = typeof appRouter;
