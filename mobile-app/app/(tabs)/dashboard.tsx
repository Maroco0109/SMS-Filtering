/**
 * Dashboard Screen
 * SMS 필터링 통계 및 인사이트 대시보드
 */

import { ScrollView, Text, View } from "react-native";
import { ScreenContainer } from "@/components/screen-container";
import { useMessages } from "@/lib/message-provider";
import { useColors } from "@/hooks/use-colors";
import { IconSymbol } from "@/components/ui/icon-symbol";
import {
  calculateDailyStats,
  calculateModelStats,
  calculateWeeklyTrend,
  calculateOverallStats,
  formatDate,
} from "@/lib/statistics";
import { BarChart } from "@/components/charts/bar-chart";
import { StackedBarChart } from "@/components/charts/stacked-bar-chart";
import { useMemo } from "react";

export default function DashboardScreen() {
  const { messages } = useMessages();
  const colors = useColors();

  // 통계 계산 (메모이제이션)
  const stats = useMemo(() => {
    const overall = calculateOverallStats(messages);
    const daily = calculateDailyStats(messages, 7);
    const weekly = calculateWeeklyTrend(messages);
    const modelStats = calculateModelStats(messages);

    return { overall, daily, weekly, modelStats };
  }, [messages]);

  // 차트 데이터 변환
  const dailyChartData = stats.daily.map((d) => ({
    label: formatDate(d.date),
    spam: d.spam,
    inbox: d.inbox,
  }));

  const weeklyChartData = stats.weekly.map((w) => ({
    label: w.weekday,
    spam: w.spam,
    inbox: w.inbox,
  }));

  const modelChartData = stats.modelStats.map((m) => ({
    label: m.model,
    value: m.count,
    color: m.model === "BERT" ? colors.primary : m.model === "RoBERTa" ? colors.success : colors.warning,
  }));

  return (
    <ScreenContainer className="bg-background">
      <ScrollView
        className="flex-1"
        contentContainerStyle={{ paddingHorizontal: 16, paddingVertical: 8 }}
        showsVerticalScrollIndicator={false}
      >
        {/* 헤더 */}
        <View className="mb-6">
          <Text className="text-3xl font-bold text-foreground">대시보드</Text>
          <Text className="text-sm text-muted mt-1">
            SMS 필터링 통계 및 인사이트
          </Text>
        </View>

        {/* 전체 통계 카드 */}
        <View className="flex-row gap-3 mb-6">
          <View className="flex-1 bg-surface rounded-2xl p-4 border border-border">
            <View className="flex-row items-center gap-2 mb-2">
              <IconSymbol name="envelope.fill" size={20} color={colors.primary} />
              <Text className="text-xs text-muted">전체 메시지</Text>
            </View>
            <Text className="text-2xl font-bold text-foreground">
              {stats.overall.totalMessages}
            </Text>
          </View>

          <View className="flex-1 bg-surface rounded-2xl p-4 border border-border">
            <View className="flex-row items-center gap-2 mb-2">
              <IconSymbol name="xmark.shield.fill" size={20} color={colors.error} />
              <Text className="text-xs text-muted">차단된 스팸</Text>
            </View>
            <Text className="text-2xl font-bold text-error">
              {stats.overall.totalSpam}
            </Text>
          </View>
        </View>

        {/* 스팸 비율 카드 */}
        <View className="bg-surface rounded-2xl p-4 border border-border mb-6">
          <Text className="text-sm font-semibold text-foreground mb-3">
            스팸 차단율
          </Text>
          <View className="flex-row items-end gap-2">
            <Text className="text-4xl font-bold text-foreground">
              {stats.overall.spamRate.toFixed(1)}
            </Text>
            <Text className="text-lg text-muted mb-1">%</Text>
          </View>
          <View className="mt-3 h-2 bg-background rounded-full overflow-hidden">
            <View
              className="h-full bg-error"
              style={{ width: `${stats.overall.spamRate}%` }}
            />
          </View>
        </View>

        {/* 일별 통계 차트 */}
        <View className="bg-surface rounded-2xl p-4 border border-border mb-6">
          <View className="flex-row items-center justify-between mb-3">
            <Text className="text-sm font-semibold text-foreground">
              최근 7일 통계
            </Text>
            <IconSymbol name="chart.bar.fill" size={18} color={colors.muted} />
          </View>
          <StackedBarChart data={dailyChartData} height={180} />
        </View>

        {/* 요일별 통계 차트 */}
        <View className="bg-surface rounded-2xl p-4 border border-border mb-6">
          <View className="flex-row items-center justify-between mb-3">
            <Text className="text-sm font-semibold text-foreground">
              요일별 통계
            </Text>
            <IconSymbol name="calendar" size={18} color={colors.muted} />
          </View>
          <StackedBarChart data={weeklyChartData} height={180} />
        </View>

        {/* 모델별 사용 통계 */}
        {modelChartData.length > 0 && (
          <View className="bg-surface rounded-2xl p-4 border border-border mb-6">
            <View className="flex-row items-center justify-between mb-3">
              <Text className="text-sm font-semibold text-foreground">
                모델별 사용 통계
              </Text>
              <IconSymbol name="cpu" size={18} color={colors.muted} />
            </View>
            <BarChart data={modelChartData} height={160} />
          </View>
        )}

        {/* 모델 성능 상세 */}
        {stats.modelStats.length > 0 && (
          <View className="bg-surface rounded-2xl p-4 border border-border mb-6">
            <Text className="text-sm font-semibold text-foreground mb-3">
              모델 성능 상세
            </Text>
            {stats.modelStats.map((model, index) => (
              <View
                key={model.model}
                className={`flex-row items-center justify-between py-3 ${
                  index < stats.modelStats.length - 1 ? "border-b border-border" : ""
                }`}
              >
                <View className="flex-1">
                  <Text className="text-sm font-semibold text-foreground">
                    {model.model}
                  </Text>
                  <Text className="text-xs text-muted mt-1">
                    평균 신뢰도: {(model.avgConfidence * 100).toFixed(1)}%
                  </Text>
                </View>
                <View className="items-end">
                  <Text className="text-sm font-semibold text-foreground">
                    {model.count}건
                  </Text>
                  <Text className="text-xs text-muted mt-1">
                    스팸 {model.spamCount} / 정상 {model.inboxCount}
                  </Text>
                </View>
              </View>
            ))}
          </View>
        )}

        {/* 빈 상태 */}
        {messages.length === 0 && (
          <View className="items-center justify-center py-12">
            <IconSymbol name="chart.bar" size={64} color={colors.muted} />
            <Text className="text-base text-muted mt-4 text-center">
              아직 분석된 메시지가 없습니다{"\n"}
              메시지를 분석하면 통계가 표시됩니다
            </Text>
          </View>
        )}

        {/* 하단 여백 */}
        <View className="h-6" />
      </ScrollView>
    </ScreenContainer>
  );
}
