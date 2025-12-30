import React, { createContext, useContext, useState, useEffect, ReactNode } from "react";
import { AppSettings, DEFAULT_SETTINGS } from "@/types/settings";
import { loadSettings, saveSettings as saveSettingsToStorage } from "@/lib/storage";
import { ModelType } from "@/types/message";

interface SettingsContextType {
  settings: AppSettings;
  loading: boolean;
  updateDefaultModel: (model: ModelType) => Promise<void>;
  updateNotifications: (enabled: boolean) => Promise<void>;
  refreshSettings: () => Promise<void>;
}

const SettingsContext = createContext<SettingsContextType | undefined>(undefined);

export function SettingsProvider({ children }: { children: ReactNode }) {
  const [settings, setSettings] = useState<AppSettings>(DEFAULT_SETTINGS);
  const [loading, setLoading] = useState(true);

  const refreshSettings = async () => {
    try {
      setLoading(true);
      const loadedSettings = await loadSettings();
      setSettings(loadedSettings);
    } catch (error) {
      console.error("Error refreshing settings:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refreshSettings();
  }, []);

  const updateDefaultModel = async (model: ModelType) => {
    const newSettings = { ...settings, defaultModel: model };
    await saveSettingsToStorage(newSettings);
    setSettings(newSettings);
  };

  const updateNotifications = async (enabled: boolean) => {
    const newSettings = { ...settings, notificationsEnabled: enabled };
    await saveSettingsToStorage(newSettings);
    setSettings(newSettings);
  };

  return (
    <SettingsContext.Provider
      value={{
        settings,
        loading,
        updateDefaultModel,
        updateNotifications,
        refreshSettings,
      }}
    >
      {children}
    </SettingsContext.Provider>
  );
}

export function useSettings() {
  const context = useContext(SettingsContext);
  if (context === undefined) {
    throw new Error("useSettings must be used within a SettingsProvider");
  }
  return context;
}
