import { ModelType } from "./message";

export interface AppSettings {
  defaultModel: ModelType;
  notificationsEnabled: boolean;
}

export const DEFAULT_SETTINGS: AppSettings = {
  defaultModel: "BERT",
  notificationsEnabled: true,
};
