import { View, Text, TouchableOpacity, Alert, ScrollView, Switch } from "react-native";
import { ScreenContainer } from "@/components/screen-container";
import { useSettings } from "@/lib/settings-provider";
import { useMessages } from "@/lib/message-provider";
import { useColors } from "@/hooks/use-colors";
import { IconSymbol } from "@/components/ui/icon-symbol";
import { ModelType } from "@/types/message";

export default function SettingsScreen() {
  const { settings, updateDefaultModel, updateNotifications } = useSettings();
  const { clearMessages, stats } = useMessages();
  const colors = useColors();

  const handleClearData = () => {
    Alert.alert(
      "데이터 삭제",
      `모든 메시지(${stats.totalMessages}개)를 삭제하시겠습니까? 이 작업은 되돌릴 수 없습니다.`,
      [
        { text: "취소", style: "cancel" },
        {
          text: "삭제",
          style: "destructive",
          onPress: async () => {
            await clearMessages();
            Alert.alert("완료", "모든 메시지가 삭제되었습니다.");
          },
        },
      ]
    );
  };

  const renderModelOption = (model: ModelType) => {
    const isSelected = settings.defaultModel === model;
    return (
      <TouchableOpacity
        key={model}
        onPress={() => updateDefaultModel(model)}
        className="flex-row items-center justify-between py-4 px-4 bg-surface rounded-xl mb-2"
      >
        <Text className="text-foreground font-medium">{model}</Text>
        {isSelected && <IconSymbol name="checkmark.shield.fill" size={20} color={colors.primary} />}
      </TouchableOpacity>
    );
  };

  return (
    <ScreenContainer className="p-4">
      <ScrollView showsVerticalScrollIndicator={false}>
        <Text className="text-3xl font-bold text-foreground mb-6">Settings</Text>

        {/* 기본 모델 선택 */}
        <View className="mb-6">
          <Text className="text-lg font-semibold text-foreground mb-3">기본 모델</Text>
          <View>
            {renderModelOption("BERT")}
            {renderModelOption("RoBERTa")}
            {renderModelOption("BigBird")}
          </View>
        </View>

        {/* 알림 설정 */}
        <View className="mb-6">
          <Text className="text-lg font-semibold text-foreground mb-3">알림</Text>
          <View className="flex-row items-center justify-between py-4 px-4 bg-surface rounded-xl">
            <Text className="text-foreground font-medium">알림 활성화</Text>
            <Switch
              value={settings.notificationsEnabled}
              onValueChange={updateNotifications}
              trackColor={{ false: colors.border, true: colors.primary }}
              thumbColor="#ffffff"
            />
          </View>
        </View>

        {/* 데이터 관리 */}
        <View className="mb-6">
          <Text className="text-lg font-semibold text-foreground mb-3">데이터 관리</Text>
          <TouchableOpacity
            onPress={handleClearData}
            className="py-4 px-4 bg-surface rounded-xl border border-error"
          >
            <View className="flex-row items-center gap-3">
              <IconSymbol name="trash.fill" size={20} color={colors.error} />
              <Text className="text-error font-medium">모든 메시지 삭제</Text>
            </View>
          </TouchableOpacity>
        </View>

        {/* 앱 정보 */}
        <View className="mb-6">
          <Text className="text-lg font-semibold text-foreground mb-3">앱 정보</Text>
          <View className="py-4 px-4 bg-surface rounded-xl">
            <View className="flex-row justify-between mb-2">
              <Text className="text-muted">버전</Text>
              <Text className="text-foreground font-medium">1.0.0</Text>
            </View>
            <View className="flex-row justify-between mb-2">
              <Text className="text-muted">전체 메시지</Text>
              <Text className="text-foreground font-medium">{stats.totalMessages}개</Text>
            </View>
            <View className="flex-row justify-between">
              <Text className="text-muted">스팸 차단</Text>
              <Text className="text-foreground font-medium">{stats.spamCount}개</Text>
            </View>
          </View>
        </View>

        <View className="items-center py-6">
          <Text className="text-xs text-muted">SMS Shield</Text>
          <Text className="text-xs text-muted mt-1">Powered by BERT, RoBERTa, BigBird</Text>
        </View>
      </ScrollView>
    </ScreenContainer>
  );
}
