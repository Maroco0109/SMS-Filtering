import AsyncStorage from "@react-native-async-storage/async-storage";
import { Message } from "@/types/message";
import { AppSettings, DEFAULT_SETTINGS } from "@/types/settings";

const MESSAGES_KEY = "@sms_shield_messages";
const SETTINGS_KEY = "@sms_shield_settings";

// Message Storage Functions
export async function saveMessages(messages: Message[]): Promise<void> {
  try {
    const jsonValue = JSON.stringify(messages);
    await AsyncStorage.setItem(MESSAGES_KEY, jsonValue);
  } catch (error) {
    console.error("Error saving messages:", error);
    throw error;
  }
}

export async function loadMessages(): Promise<Message[]> {
  try {
    const jsonValue = await AsyncStorage.getItem(MESSAGES_KEY);
    return jsonValue != null ? JSON.parse(jsonValue) : [];
  } catch (error) {
    console.error("Error loading messages:", error);
    return [];
  }
}

export async function addMessage(message: Message): Promise<void> {
  try {
    const messages = await loadMessages();
    messages.unshift(message); // Add to beginning
    await saveMessages(messages);
  } catch (error) {
    console.error("Error adding message:", error);
    throw error;
  }
}

export async function deleteMessage(id: string): Promise<void> {
  try {
    const messages = await loadMessages();
    const filtered = messages.filter((msg) => msg.id !== id);
    await saveMessages(filtered);
  } catch (error) {
    console.error("Error deleting message:", error);
    throw error;
  }
}

export async function updateMessage(id: string, updates: Partial<Message>): Promise<void> {
  try {
    const messages = await loadMessages();
    const index = messages.findIndex((msg) => msg.id === id);
    if (index !== -1) {
      messages[index] = { ...messages[index], ...updates, updatedAt: new Date().toISOString() };
      await saveMessages(messages);
    }
  } catch (error) {
    console.error("Error updating message:", error);
    throw error;
  }
}

export async function clearAllMessages(): Promise<void> {
  try {
    await AsyncStorage.removeItem(MESSAGES_KEY);
  } catch (error) {
    console.error("Error clearing messages:", error);
    throw error;
  }
}

// Settings Storage Functions
export async function saveSettings(settings: AppSettings): Promise<void> {
  try {
    const jsonValue = JSON.stringify(settings);
    await AsyncStorage.setItem(SETTINGS_KEY, jsonValue);
  } catch (error) {
    console.error("Error saving settings:", error);
    throw error;
  }
}

export async function loadSettings(): Promise<AppSettings> {
  try {
    const jsonValue = await AsyncStorage.getItem(SETTINGS_KEY);
    return jsonValue != null ? JSON.parse(jsonValue) : DEFAULT_SETTINGS;
  } catch (error) {
    console.error("Error loading settings:", error);
    return DEFAULT_SETTINGS;
  }
}
