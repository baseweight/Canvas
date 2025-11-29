// Core type definitions for Baseweight Canvas

export interface MediaItem {
  id: string;
  type: 'image' | 'video' | 'audio';
  url: string;
  filePath?: string; // Original file path for backend operations (audio)
  filename: string;
  size: number;
  dimensions?: {
    width: number;
    height: number;
  };
  thumbnail?: string;
  createdAt: Date;
}

export type ModelTask =
  | 'general-vlm'      // General purpose vision-language model
  | 'audio-llm'        // Audio-capable language model
  | 'ocr'              // Optical character recognition
  | 'classifier'       // Content classification (e.g., NSFW detection)
  | 'captioning'       // Image captioning
  | 'vqa'              // Visual question answering
  | 'detection';       // Object detection

export type InferenceBackend = 'llama.cpp' | 'onnx-runtime' | 'coreml';

export interface Model {
  id: string;
  name: string;
  displayName: string;
  task: ModelTask;
  taskDescription: string;
  backend: InferenceBackend;
  huggingfaceUrl: string;
  size: number;
  downloaded: boolean;
  bundled?: boolean; // True if model ships with Baseweight Canvas
  downloadProgress?: number;
  localPath?: string;
  quantization?: string;
  parameters?: {
    contextLength?: number;
    maxTokens?: number;
  };
}

export interface InferenceJob {
  id: string;
  mediaId: string;
  modelId: string;
  prompt: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress?: number;
  result?: string;
  error?: string;
  startedAt?: Date;
  completedAt?: Date;
  parameters?: {
    temperature?: number;
    maxTokens?: number;
    topP?: number;
  };
}

export interface Project {
  id: string;
  name: string;
  media: MediaItem[];
  inferences: InferenceJob[];
  createdAt: Date;
  updatedAt: Date;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  mediaId?: string;
}

export interface AvailableModel {
  id: string;
  name: string;
  displayName: string;
  task: ModelTask;
  taskDescription: string;
  backend: InferenceBackend;
  huggingfaceUrl: string;
  size: number;
  quantization?: string;
  description?: string;
}
