import { View, Text, FlatList, TouchableOpacity, Alert } from "react-native";
import { useState } from "react";
import { ScreenContainer } from "@/components/screen-container";
import { useMessages } from "@/lib/message-provider";
import { useColors } from "@/hooks/use-colors";
import { IconSymbol } from "@/components/ui/icon-symbol";
import { Classification } from "@/types/message";
import { router } from "expo-router";

type FilterType = "all" | Classification;

export default function MessagesScreen() {
  const { messages, loading, deleteMessage, getFilteredMessages } = useMessages();
  const colors = useColors();
  const [activeFilter, setActiveFilter] = useState<FilterType>("all");

  const filteredMessages = getFilteredMessages(activeFilter);

  const handleDelete = (id: string) => {
    Alert.alert("메시지 삭제", "이 메시지를 삭제하시겠습니까?", [
      { text: "취소", style: "cancel" },
      {
        text: "삭제",
        style: "destructive",
        onPress: async () => {
          await deleteMessage(id);
        },
      },
    ]);
  };

  const renderFilterTab = (filter: FilterType, label: string) => {
    const isActive = activeFilter === filter;
    return (
      <TouchableOpacity
        onPress={() => setActiveFilter(filter)}
        className={`px-6 py-3 rounded-full ${isActive ? "bg-primary" : "bg-surface"}`}
        style={{ opacity: isActive ? 1 : 0.7 }}
      >
        <Text className={`font-semibold ${isActive ? "text-white" : "text-foreground"}`}>
          {label}
        </Text>
      </TouchableOpacity>
    );
  };

  const renderMessage = ({ item }: { item: typeof messages[0] }) => {
    const isSpam = item.classification === "SPAM";
    const badgeColor = isSpam ? colors.error : colors.success;
    const iconName = isSpam ? "xmark.shield.fill" : "checkmark.shield.fill";

    return (
      <TouchableOpacity
        onPress={() => {
          // TODO: 메시지 상세 화면 구현 예정
          Alert.alert("메시지 상세", `ID: ${item.id}\n\n${item.text}`);
        }}
        className="bg-surface rounded-2xl p-4 mb-3 border border-border"
        style={{ opacity: 0.95 }}
      >
        <View className="flex-row items-start justify-between mb-2">
          <View className="flex-row items-center gap-2 flex-1">
            <IconSymbol name={iconName as any} size={20} color={badgeColor} />
            <Text className="font-bold text-foreground">{item.classification}</Text>
            <Text className="text-xs text-muted">
              {Math.round(item.confidence * 100)}%
            </Text>
          </View>
          <TouchableOpacity onPress={() => handleDelete(item.id)}>
            <IconSymbol name="trash.fill" size={18} color={colors.muted} />
          </TouchableOpacity>
        </View>

        <Text className="text-foreground mb-2" numberOfLines={2}>
          {item.text}
        </Text>

        <View className="flex-row items-center justify-between">
          <Text className="text-xs text-muted">{item.model}</Text>
          <Text className="text-xs text-muted">
            {new Date(item.createdAt).toLocaleDateString("ko-KR")}
          </Text>
        </View>
      </TouchableOpacity>
    );
  };

  return (
    <ScreenContainer className="p-4">
      <View className="mb-4">
        <Text className="text-3xl font-bold text-foreground mb-4">Messages</Text>
        <View className="flex-row gap-2">
          {renderFilterTab("all", "전체")}
          {renderFilterTab("INBOX", "INBOX")}
          {renderFilterTab("SPAM", "SPAM")}
        </View>
      </View>

      {loading ? (
        <View className="flex-1 items-center justify-center">
          <Text className="text-muted">로딩 중...</Text>
        </View>
      ) : filteredMessages.length === 0 ? (
        <View className="flex-1 items-center justify-center">
          <IconSymbol name="tray.fill" size={48} color={colors.muted} />
          <Text className="text-muted mt-4">메시지가 없습니다</Text>
        </View>
      ) : (
        <FlatList
          data={filteredMessages}
          renderItem={renderMessage}
          keyExtractor={(item) => item.id}
          showsVerticalScrollIndicator={false}
          contentContainerStyle={{ paddingBottom: 20 }}
        />
      )}
    </ScreenContainer>
  );
}
