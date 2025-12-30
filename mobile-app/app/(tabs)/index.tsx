import { ScrollView, Text, View, TouchableOpacity, FlatList } from "react-native";
import { ScreenContainer } from "@/components/screen-container";
import { useMessages } from "@/lib/message-provider";
import { useColors } from "@/hooks/use-colors";
import { IconSymbol } from "@/components/ui/icon-symbol";
import { router } from "expo-router";

export default function HomeScreen() {
  const { messages, stats, loading } = useMessages();
  const colors = useColors();

  const recentMessages = messages.slice(0, 5);

  const handleAnalyze = () => {
    router.push("/analyze");
  };

  const renderStatCard = (icon: string, label: string, value: number, color: string) => (
    <View className="flex-1 bg-surface rounded-2xl p-4 border border-border">
      <View className="flex-row items-center gap-2 mb-2">
        <IconSymbol name={icon as any} size={24} color={color} />
        <Text className="text-xs text-muted">{label}</Text>
      </View>
      <Text className="text-2xl font-bold text-foreground">{value}</Text>
    </View>
  );

  const renderRecentMessage = ({ item }: { item: typeof messages[0] }) => {
    const isSpam = item.classification === "SPAM";
    const badgeColor = isSpam ? colors.error : colors.success;
    const iconName = isSpam ? "xmark.shield.fill" : "checkmark.shield.fill";

    return (
      <View className="bg-surface rounded-xl p-3 mb-2 border border-border">
        <View className="flex-row items-center gap-2 mb-1">
          <IconSymbol name={iconName as any} size={16} color={badgeColor} />
          <Text className="font-semibold text-foreground text-xs">{item.classification}</Text>
          <Text className="text-xs text-muted">
            {Math.round(item.confidence * 100)}%
          </Text>
        </View>
        <Text className="text-foreground text-sm" numberOfLines={1}>
          {item.text}
        </Text>
      </View>
    );
  };

  return (
    <ScreenContainer className="p-6">
      <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={{ flexGrow: 1 }}>
        <View className="flex-1 gap-6">
          {/* 헤더 */}
          <View>
            <Text className="text-4xl font-bold text-foreground">SMS Shield</Text>
            <Text className="text-base text-muted mt-1">AI 기반 스팸 메시지 필터링</Text>
          </View>

          {/* 통계 카드 */}
          <View className="flex-row gap-3">
            {renderStatCard("xmark.shield.fill", "오늘 차단", stats.todaySpamCount, colors.error)}
            {renderStatCard("tray.fill", "전체 메시지", stats.totalMessages, colors.primary)}
          </View>

          {/* 최근 메시지 */}
          <View>
            <View className="flex-row items-center justify-between mb-3">
              <Text className="text-lg font-semibold text-foreground">최근 메시지</Text>
              <TouchableOpacity onPress={() => router.push("/(tabs)/messages")}>
                <Text className="text-sm text-primary font-medium">전체 보기</Text>
              </TouchableOpacity>
            </View>

            {loading ? (
              <View className="items-center py-8">
                <Text className="text-muted">로딩 중...</Text>
              </View>
            ) : recentMessages.length === 0 ? (
              <View className="items-center py-8 bg-surface rounded-2xl">
                <IconSymbol name="tray.fill" size={40} color={colors.muted} />
                <Text className="text-muted mt-3">아직 분석된 메시지가 없습니다</Text>
              </View>
            ) : (
              <View>
                {recentMessages.map((item) => (
                  <View key={item.id}>{renderRecentMessage({ item })}</View>
                ))}
              </View>
            )}
          </View>

          {/* 새 메시지 분석 버튼 */}
          <TouchableOpacity
            onPress={handleAnalyze}
            className="bg-primary rounded-2xl py-4 flex-row items-center justify-center gap-3 mt-4"
            style={{ opacity: 0.95 }}
          >
            <IconSymbol name="plus.circle.fill" size={24} color="#ffffff" />
            <Text className="text-white font-bold text-lg">새 메시지 분석</Text>
          </TouchableOpacity>
        </View>
      </ScrollView>
    </ScreenContainer>
  );
}
