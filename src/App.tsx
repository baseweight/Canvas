import { useState } from "react";
import { Layout } from "./components/Layout";
import { ImageViewer } from "./components/ImageViewer";
import { ChatPanel } from "./components/ChatPanel";
import { ModelSelectionModal } from "./components/ModelSelectionModal";
import type { MediaItem, Model, AvailableModel, ChatMessage } from "./types";
import "./App.css";

// Bundled model that ships with Baseweight Canvas
const BUNDLED_MODEL: Model = {
  id: 'smolvlm2-2.2b-instruct',
  name: 'SmolVLM2-2.2B-Instruct-GGUF',
  displayName: 'SmolVLM2 2.2B Instruct',
  task: 'general-vlm',
  taskDescription: 'General Vision-Language Model',
  backend: 'llama.cpp',
  huggingfaceUrl: 'https://huggingface.co/ggml-org/SmolVLM2-2.2B-Instruct-GGUF',
  size: 1.3 * 1024 * 1024 * 1024,
  downloaded: true,
  bundled: true, // Ships with Baseweight Canvas out of the box
  quantization: 'Q4_K_M',
  localPath: '/bundled/smolvlm2-2.2b-instruct-q4_k_m.gguf',
};

// Mock downloaded models (user-downloaded models would go here)
const MOCK_DOWNLOADED_MODELS: Model[] = [
  BUNDLED_MODEL, // Always include the bundled model
];

// Mock available models (for download) - Real models from HuggingFace collections
const MOCK_AVAILABLE_MODELS: AvailableModel[] = [
  // Popular SmolVLM2 models (HuggingFace)
  {
    id: 'smolvlm2-2.2b-instruct',
    name: 'SmolVLM2-2.2B-Instruct-GGUF',
    displayName: 'SmolVLM2 2.2B Instruct',
    task: 'general-vlm',
    taskDescription: 'General Vision-Language Model',
    backend: 'llama.cpp',
    huggingfaceUrl: 'https://huggingface.co/ggml-org/SmolVLM2-2.2B-Instruct-GGUF',
    size: 1.3 * 1024 * 1024 * 1024,
    quantization: 'Q4_K_M',
    description: 'Popular compact vision-language model from HuggingFace, excellent quality-to-size ratio',
  },
  {
    id: 'smolvlm2-500m-video',
    name: 'SmolVLM2-500M-Video-Instruct-GGUF',
    displayName: 'SmolVLM2 500M Video',
    task: 'general-vlm',
    taskDescription: 'General Vision-Language Model',
    backend: 'llama.cpp',
    huggingfaceUrl: 'https://huggingface.co/ggml-org/SmolVLM2-500M-Video-Instruct-GGUF',
    size: 0.3 * 1024 * 1024 * 1024,
    quantization: 'Q4_K_M',
    description: 'Tiny video-capable vision model, supports both images and video frames',
  },

  // PaliGemma (Google vision-language)
  {
    id: 'paligemma-3b',
    name: 'paligemma-3b-mix-224-gguf',
    displayName: 'PaliGemma 3B',
    task: 'general-vlm',
    taskDescription: 'General Vision-Language Model',
    backend: 'llama.cpp',
    huggingfaceUrl: 'https://huggingface.co/abetlen/paligemma-3b-mix-224-gguf',
    size: 2.5 * 1024 * 1024 * 1024,
    quantization: 'Q4_K_M',
    description: 'Google\'s vision-language model based on Gemma, strong OCR and visual reasoning',
  },

  // ggml-org multimodal models
  {
    id: 'mistral-small-3.1-24b',
    name: 'Mistral-Small-3.1-24B-Instruct-2503-GGUF',
    displayName: 'Mistral Small 3.1 24B Instruct',
    task: 'general-vlm',
    taskDescription: 'General Vision-Language Model',
    backend: 'llama.cpp',
    huggingfaceUrl: 'https://huggingface.co/ggml-org/Mistral-Small-3.1-24B-Instruct-2503-GGUF',
    size: 14 * 1024 * 1024 * 1024,
    quantization: 'Q4_K_M',
    description: 'Powerful 24B parameter vision-language model with strong instruction following',
  },
  {
    id: 'pixtral-12b',
    name: 'pixtral-12b-GGUF',
    displayName: 'Pixtral 12B',
    task: 'general-vlm',
    taskDescription: 'General Vision-Language Model',
    backend: 'llama.cpp',
    huggingfaceUrl: 'https://huggingface.co/ggml-org/pixtral-12b-GGUF',
    size: 7 * 1024 * 1024 * 1024,
    quantization: 'Q4_K_M',
    description: 'High-performance 12B vision model from Mistral AI',
  },
  {
    id: 'moondream2-2025',
    name: 'moondream2-20250414-GGUF',
    displayName: 'Moondream 2 (2025)',
    task: 'general-vlm',
    taskDescription: 'General Vision-Language Model',
    backend: 'llama.cpp',
    huggingfaceUrl: 'https://huggingface.co/ggml-org/moondream2-20250414-GGUF',
    size: 0.3 * 1024 * 1024 * 1024,
    quantization: 'Q4_K_M',
    description: 'Ultra-compact 500M parameter vision model, perfect for edge devices',
  },

  // LiquidAI LFM2-VL models
  {
    id: 'lfm2-vl-3b',
    name: 'LFM2-VL-3B-GGUF',
    displayName: 'LFM2-VL 3B',
    task: 'general-vlm',
    taskDescription: 'General Vision-Language Model',
    backend: 'llama.cpp',
    huggingfaceUrl: 'https://huggingface.co/LiquidAI/LFM2-VL-3B-GGUF',
    size: 2 * 1024 * 1024 * 1024,
    quantization: 'Q4_K_M',
    description: 'On-device optimized 3B vision-language model from Liquid AI',
  },
  {
    id: 'lfm2-vl-1.6b',
    name: 'LFM2-VL-1.6B-GGUF',
    displayName: 'LFM2-VL 1.6B',
    task: 'general-vlm',
    taskDescription: 'General Vision-Language Model',
    backend: 'llama.cpp',
    huggingfaceUrl: 'https://huggingface.co/LiquidAI/LFM2-VL-1.6B-GGUF',
    size: 1 * 1024 * 1024 * 1024,
    quantization: 'Q4_K_M',
    description: 'Efficient 1.6B model designed for on-device deployment',
  },
  {
    id: 'lfm2-vl-450m',
    name: 'LFM2-VL-450M-GGUF',
    displayName: 'LFM2-VL 450M',
    task: 'general-vlm',
    taskDescription: 'General Vision-Language Model',
    backend: 'llama.cpp',
    huggingfaceUrl: 'https://huggingface.co/LiquidAI/LFM2-VL-450M-GGUF',
    size: 0.3 * 1024 * 1024 * 1024,
    quantization: 'Q4_K_M',
    description: 'Ultra-lightweight 450M model for resource-constrained environments',
  },

];

function App() {
  const [currentMedia, setCurrentMedia] = useState<MediaItem | undefined>(undefined);
  const [currentModel, setCurrentModel] = useState<Model | undefined>(BUNDLED_MODEL); // SmolVLM2 ships with the app
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isModelModalOpen, setIsModelModalOpen] = useState(false);

  const handleMediaDrop = (files: File[]) => {
    if (files.length === 0) return;

    const file = files[0];

    // Create image element to get dimensions
    const img = new Image();
    const url = URL.createObjectURL(file);

    img.onload = () => {
      const mediaItem: MediaItem = {
        id: crypto.randomUUID(),
        type: file.type.startsWith('image/') ? 'image' : 'video',
        url,
        filename: file.name,
        size: file.size,
        dimensions: {
          width: img.width,
          height: img.height,
        },
        createdAt: new Date(),
      };

      setCurrentMedia(mediaItem);
      // Clear messages when new image is loaded
      setMessages([]);
    };

    img.src = url;
  };

  const handleSendMessage = (content: string) => {
    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content,
      timestamp: new Date(),
      mediaId: currentMedia?.id,
    };

    setMessages(prev => [...prev, userMessage]);

    // Simulate assistant response (replace with actual inference later)
    setTimeout(() => {
      const assistantMessage: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: 'This is a mock response. The llama.cpp inference backend will be integrated next.',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, assistantMessage]);
    }, 1000);
  };

  const handleSelectModel = (modelId: string) => {
    const model = MOCK_DOWNLOADED_MODELS.find(m => m.id === modelId);
    if (model) {
      setCurrentModel(model);
      // Clear messages when switching models
      setMessages([]);
    }
  };

  const handleDownloadModel = (modelId: string) => {
    console.log('Download model:', modelId);
    // TODO: Implement actual download logic
    alert(`Download functionality will be implemented with the Rust backend.\nModel ID: ${modelId}`);
  };

  const handleAddModel = (repo: string, quantization: string) => {
    console.log('Add model from HuggingFace:', repo, quantization);
    // TODO: Implement actual HuggingFace download logic
    alert(
      `Add Model from HuggingFace\n\n` +
      `Repository: ${repo}\n` +
      `Quantization: ${quantization}\n\n` +
      `This will be implemented with the Rust backend:\n` +
      `1. Fetch model card from HuggingFace\n` +
      `2. Locate mmproj-*.gguf and model-${quantization.toLowerCase()}.gguf\n` +
      `3. Download both files\n` +
      `4. Add to downloaded models list`
    );
  };

  return (
    <Layout>
      <ImageViewer
        mediaItem={currentMedia}
        onMediaDrop={handleMediaDrop}
      />
      <ChatPanel
        currentModel={currentModel}
        messages={messages}
        onChangeModel={() => setIsModelModalOpen(true)}
        onSendMessage={handleSendMessage}
      />
      <ModelSelectionModal
        isOpen={isModelModalOpen}
        onClose={() => setIsModelModalOpen(false)}
        currentModel={currentModel}
        downloadedModels={MOCK_DOWNLOADED_MODELS}
        availableModels={MOCK_AVAILABLE_MODELS}
        onSelectModel={handleSelectModel}
        onDownloadModel={handleDownloadModel}
        onAddModel={handleAddModel}
      />
    </Layout>
  );
}

export default App;
