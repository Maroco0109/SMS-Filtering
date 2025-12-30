import { View, Text, TextInput, TouchableOpacity, ScrollView, Alert, ActivityIndicator } from "react-native";
import { useState } from "react";
import { ScreenContainer } from "@/components/screen-container";
import { useMessages } from "@/lib/message-provider";
import { useSettings } from "@/lib/settings-provider";
import { useColors } from "@/hooks/use-colors";
import { IconSymbol } from "@/components/ui/icon-symbol";
import { ModelType, AnalyzeResponse } from "@/types/message";
import { router } from "expo-router";
import { trpc } from "@/lib/trpc";

export default function AnalyzeScreen() {
  const { addMessage } = useMessages();
  const { settings } = useSettings();
  const colors = useColors();

  const [text, setText] = useState("");
  const [selectedModel, setSelectedModel] = useState<ModelType>(settings.defaultModel);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);

  const analyzeMutation = trpc.sms.analyze.useMutation();

  const handleAnalyze = async () => {
    if (!text.trim()) {
      Alert.alert("오류", "분석할 메시지를 입력해주세요.");
      return;
    }

    setAnalyzing(true);
    setResult(null);

    try {
      // tRPC API 호출
      const apiResult = await analyzeMutation.mutateAsync({
        text: text.trim(),
        model: selectedModel,
      });

      const apiResponse: AnalyzeResponse = {
        classification: apiResult.classification,
        confidence: apiResult.confidence,
        model: apiResult.model,
      };

      setResult(apiResponse);

      // 메시지 저장
      const newMessage = {
        id: Date.now().toString(),
        text: text.trim(),
        classification: apiResponse.classification,
        confidence: apiResponse.confidence,
        model: apiResponse.model,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      };

      await addMessage(newMessage);
    } catch (error) {
      Alert.alert("오류", "분석 중 오류가 발생했습니다.");
      console.error(error);
    } finally {
      setAnalyzing(false);
    }
  };

  const handleReset = () => {
    setText("");
    setResult(null);
  };

  const handleComplete = () => {
    router.back();
  };

  const renderModelButton = (model: ModelType) => {
    const isSelected = selectedModel === model;
    return (
      <TouchableOpacity
        key={model}
        onPress={() => setSelectedModel(model)}
        className={`px-4 py-2 rounded-full ${isSelected ? "bg-primary" : "bg-surface"}`}
        style={{ opacity: isSelected ? 1 : 0.7 }}
      >
        <Text className={`font-medium ${isSelected ? "text-white" : "text-foreground"}`}>
          {model}
        </Text>
      </TouchableOpacity>
    );
  };

  const renderResult = () => {
    if (!result) return null;

    const isSpam = result.classification === "SPAM";
    const badgeColor = isSpam ? colors.error : colors.success;
    const iconName = isSpam ? "xmark.shield.fill" : "checkmark.shield.fill";

    return (
      <View className="bg-surface rounded-2xl p-6 border-2" style={{ borderColor: badgeColor }}>
        <View className="items-center mb-4">
          <IconSymbol name={iconName as any} size={64} color={badgeColor} />
          <Text className="text-3xl font-bold text-foreground mt-4">{result.classification}</Text>
          <Text className="text-muted mt-2">신뢰도: {Math.round(result.confidence * 100)}%</Text>
        </View>

        <View className="bg-background rounded-xl p-4">
          <View className="flex-row justify-between mb-2">
            <Text className="text-muted">모델</Text>
            <Text className="text-foreground font-medium">{result.model}</Text>
          </View>
          <View className="flex-row justify-between">
            <Text className="text-muted">분류</Text>
            <Text className="text-foreground font-medium">{result.classification}</Text>
          </View>
        </View>

        <View className="flex-row gap-3 mt-4">
          <TouchableOpacity
            onPress={handleReset}
            className="flex-1 bg-surface rounded-xl py-3 items-center"
          >
            <Text className="text-foreground font-semibold">다시 분석</Text>
          </TouchableOpacity>
          <TouchableOpacity
            onPress={handleComplete}
            className="flex-1 bg-primary rounded-xl py-3 items-center"
          >
            <Text className="text-white font-semibold">완료</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  };

  return (
    <ScreenContainer className="p-6">
      <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={{ flexGrow: 1 }}>
        <View className="flex-1 gap-6">
          {/* 헤더 */}
          <View className="flex-row items-center justify-between">
            <View>
              <Text className="text-3xl font-bold text-foreground">메시지 분석</Text>
              <Text className="text-sm text-muted mt-1">AI 모델로 스팸 여부를 확인합니다</Text>
            </View>
            <TouchableOpacity onPress={() => router.back()}>
              <IconSymbol name="chevron.right" size={24} color={colors.muted} />
            </TouchableOpacity>
          </View>

          {/* 모델 선택 */}
          <View>
            <Text className="text-base font-semibold text-foreground mb-3">모델 선택</Text>
            <View className="flex-row gap-2">
              {renderModelButton("BERT")}
              {renderModelButton("RoBERTa")}
              {renderModelButton("BigBird")}
            </View>
          </View>

          {/* 텍스트 입력 */}
          <View>
            <Text className="text-base font-semibold text-foreground mb-3">메시지 내용</Text>
            <TextInput
              value={text}
              onChangeText={setText}
              placeholder="분석할 SMS 메시지를 입력하세요..."
              placeholderTextColor={colors.muted}
              multiline
              numberOfLines={6}
              className="bg-surface rounded-2xl p-4 text-foreground border border-border"
              style={{ minHeight: 120, textAlignVertical: "top" }}
              editable={!analyzing && !result}
            />
          </View>

          {/* 분석 버튼 */}
          {!result && (
            <TouchableOpacity
              onPress={handleAnalyze}
              disabled={analyzing || !text.trim()}
              className="bg-primary rounded-2xl py-4 items-center"
              style={{ opacity: analyzing || !text.trim() ? 0.5 : 1 }}
            >
              {analyzing ? (
                <View className="flex-row items-center gap-2">
                  <ActivityIndicator color="#ffffff" />
                  <Text className="text-white font-bold text-lg">분석 중...</Text>
                </View>
              ) : (
                <Text className="text-white font-bold text-lg">분석하기</Text>
              )}
            </TouchableOpacity>
          )}

          {/* 결과 표시 */}
          {renderResult()}
        </View>
      </ScrollView>
    </ScreenContainer>
  );
}
