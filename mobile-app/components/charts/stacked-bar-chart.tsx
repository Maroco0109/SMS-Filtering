/**
 * Stacked Bar Chart Component
 * SPAM/INBOX 비율을 보여주는 스택 바 차트
 */

import { View, Text } from "react-native";
import { useColors } from "@/hooks/use-colors";

export interface StackedBarData {
  label: string;
  spam: number;
  inbox: number;
}

interface StackedBarChartProps {
  data: StackedBarData[];
  height?: number;
}

export function StackedBarChart({ data, height = 200 }: StackedBarChartProps) {
  const colors = useColors();

  if (data.length === 0) {
    return (
      <View className="flex-1 items-center justify-center" style={{ height }}>
        <Text className="text-muted">데이터가 없습니다</Text>
      </View>
    );
  }

  const maxValue = Math.max(...data.map((d) => d.spam + d.inbox), 1);
  const barWidth = 100 / data.length;

  return (
    <View className="flex-1" style={{ height }}>
      {/* 범례 */}
      <View className="flex-row justify-center mb-3 gap-4">
        <View className="flex-row items-center gap-1">
          <View className="w-3 h-3 rounded-sm bg-error" />
          <Text className="text-xs text-foreground">SPAM</Text>
        </View>
        <View className="flex-row items-center gap-1">
          <View className="w-3 h-3 rounded-sm bg-success" />
          <Text className="text-xs text-foreground">INBOX</Text>
        </View>
      </View>

      {/* 차트 영역 */}
      <View className="flex-1 flex-row items-end justify-around px-2">
        {data.map((item, index) => {
          const total = item.spam + item.inbox;
          const totalHeight = (total / maxValue) * (height - 80);
          const spamHeight = total > 0 ? (item.spam / total) * totalHeight : 0;
          const inboxHeight = total > 0 ? (item.inbox / total) * totalHeight : 0;

          return (
            <View
              key={index}
              className="items-center justify-end"
              style={{ width: `${barWidth - 2}%` }}
            >
              {/* 총 개수 표시 */}
              {total > 0 && (
                <Text className="text-xs text-foreground mb-1 font-semibold">
                  {total}
                </Text>
              )}

              {/* 스택 바 */}
              <View className="w-full rounded-t-lg overflow-hidden">
                {/* SPAM 부분 */}
                {spamHeight > 0 && (
                  <View
                    className="w-full bg-error"
                    style={{ height: Math.max(spamHeight, 2) }}
                  />
                )}
                {/* INBOX 부분 */}
                {inboxHeight > 0 && (
                  <View
                    className="w-full bg-success"
                    style={{ height: Math.max(inboxHeight, 2) }}
                  />
                )}
              </View>
            </View>
          );
        })}
      </View>

      {/* 라벨 영역 */}
      <View className="flex-row justify-around mt-2 px-2">
        {data.map((item, index) => (
          <View
            key={index}
            className="items-center"
            style={{ width: `${barWidth - 2}%` }}
          >
            <Text
              className="text-xs text-muted text-center"
              numberOfLines={1}
              ellipsizeMode="tail"
            >
              {item.label}
            </Text>
          </View>
        ))}
      </View>
    </View>
  );
}
