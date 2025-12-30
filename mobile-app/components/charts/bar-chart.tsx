/**
 * Simple Bar Chart Component
 * React Native용 경량 바 차트
 */

import { View, Text } from "react-native";
import { useColors } from "@/hooks/use-colors";

export interface BarChartData {
  label: string;
  value: number;
  color?: string;
}

interface BarChartProps {
  data: BarChartData[];
  height?: number;
  showValues?: boolean;
}

export function BarChart({ data, height = 200, showValues = true }: BarChartProps) {
  const colors = useColors();
  
  if (data.length === 0) {
    return (
      <View className="flex-1 items-center justify-center" style={{ height }}>
        <Text className="text-muted">데이터가 없습니다</Text>
      </View>
    );
  }

  const maxValue = Math.max(...data.map((d) => d.value), 1);
  const barWidth = 100 / data.length;

  return (
    <View className="flex-1" style={{ height }}>
      {/* 차트 영역 */}
      <View className="flex-1 flex-row items-end justify-around px-2">
        {data.map((item, index) => {
          const barHeight = (item.value / maxValue) * (height - 60);
          const barColor = item.color || colors.primary;

          return (
            <View
              key={index}
              className="items-center justify-end"
              style={{ width: `${barWidth - 2}%` }}
            >
              {/* 값 표시 */}
              {showValues && item.value > 0 && (
                <Text className="text-xs text-foreground mb-1 font-semibold">
                  {item.value}
                </Text>
              )}
              
              {/* 바 */}
              <View
                className="w-full rounded-t-lg"
                style={{
                  height: Math.max(barHeight, 2),
                  backgroundColor: barColor,
                  minHeight: item.value > 0 ? 4 : 0,
                }}
              />
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
