import { useState, useEffect } from "react";
import { invoke, convertFileSrc } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open, ask } from "@tauri-apps/plugin-dialog";
import { Layout } from "./components/Layout";
import { ImageViewer } from "./components/ImageViewer";
import { ChatPanel } from "./components/ChatPanel";
import { ModelSelectionModal } from "./components/ModelSelectionModal";
import { DownloadModelDialog } from "./components/DownloadModelDialog";
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
    id: 'ggml_org_smolvlm2_500m_video_instruct_gguf',
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

  // Audio models
  {
    id: 'ggml_org_ultravox_v0_5_llama_3_2_1b_gguf',
    name: 'ultravox-v0_5-llama-3_2-1b-GGUF',
    displayName: 'Ultravox v0.5 Llama 3.2 1B',
    task: 'audio-llm',
    taskDescription: 'Audio Capable Language Model',
    backend: 'llama.cpp',
    huggingfaceUrl: 'https://huggingface.co/ggml-org/ultravox-v0_5-llama-3_2-1b-GGUF',
    size: 2.2 * 1024 * 1024 * 1024,
    quantization: 'Q4_K_M',
    description: 'Audio-capable language model based on Llama 3.2, supports speech understanding',
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

interface DownloadProgress {
  current: number;
  total: number;
  percentage: number;
  file: string;
}

function App() {
  const [currentMedia, setCurrentMedia] = useState<MediaItem | undefined>(undefined);
  const [currentModel, setCurrentModel] = useState<Model | undefined>(undefined);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isModelModalOpen, setIsModelModalOpen] = useState(false);
  const [isDownloadDialogOpen, setIsDownloadDialogOpen] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState<DownloadProgress | undefined>(undefined);
  const [_isModelLoading, setIsModelLoading] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isAudioCapable, setIsAudioCapable] = useState(false);
  const [downloadedModels, setDownloadedModels] = useState<Model[]>([]);

  // Check if bundled model is downloaded on startup
  useEffect(() => {
    const checkBundledModel = async () => {
      try {
        const isDownloaded = await invoke<boolean>('is_bundled_model_downloaded');

        if (isDownloaded) {
          // Model is downloaded, set it as current
          const updatedModel = { ...BUNDLED_MODEL, downloaded: true };
          setCurrentModel(updatedModel);
        } else {
          // Model not downloaded, show download dialog
          setIsDownloadDialogOpen(true);
        }
      } catch (error) {
        console.error('Failed to check bundled model:', error);
      }
    };

    checkBundledModel();

    // Listen for download progress events
    const unlisten = listen<DownloadProgress>('download-progress', (event) => {
      setDownloadProgress(event.payload);
    });

    return () => {
      unlisten.then(fn => fn());
    };
  }, []);

  // Load all downloaded models on startup
  useEffect(() => {
    const loadDownloadedModels = async () => {
      try {
        const modelIds = await invoke<string[]>('list_downloaded_models');

        // Convert model IDs to full Model objects
        const models: Model[] = modelIds.map(modelId => {
          // Check if it's the bundled model
          if (modelId === BUNDLED_MODEL.id) {
            return BUNDLED_MODEL;
          }

          // Find in available models
          const availableModel = MOCK_AVAILABLE_MODELS.find(m => m.id === modelId);
          if (availableModel) {
            return {
              ...availableModel,
              localPath: `/models/${modelId}`,
              downloaded: true,
            } as Model;
          }

          // Fallback for unknown models
          return {
            id: modelId,
            name: modelId,
            displayName: modelId,
            task: 'general-vlm',
            taskDescription: 'General Vision-Language Model',
            backend: 'llama.cpp',
            huggingfaceUrl: `https://huggingface.co/${modelId}`,
            size: 0,
            localPath: `/models/${modelId}`,
            downloaded: true,
          } as Model;
        });

        setDownloadedModels(models);
      } catch (error) {
        console.error('Failed to load downloaded models:', error);
      }
    };

    loadDownloadedModels();
  }, []);

  // Listen for file-opened events from the File menu
  useEffect(() => {
    console.log('Setting up file-opened event listener');
    const unlisten = listen<string>('file-opened', async (event) => {
      console.log('file-opened event received:', event);

      // If there's already an image loaded, confirm before resetting session
      if (currentMedia) {
        const confirmed = await ask(
          'Loading a new image will reset the current session and clear the chat history. Do you want to continue?',
          {
            title: 'Reset Session?',
            kind: 'warning',
          }
        );

        if (!confirmed) {
          return; // User cancelled
        }
      }

      const filePath = event.payload;
      console.log('File path:', filePath);
      const url = convertFileSrc(filePath);
      console.log('Converted URL:', url);
      const filename = filePath.split('/').pop() || filePath.split('\\').pop() || 'file';
      console.log('Filename:', filename);

      // Determine file type based on extension
      const extension = filename.split('.').pop()?.toLowerCase() || '';
      const audioExtensions = ['wav', 'mp3', 'flac'];
      const isAudio = audioExtensions.includes(extension);

      if (isAudio) {
        // Handle audio file
        const mediaItem: MediaItem = {
          id: crypto.randomUUID(),
          type: 'audio',
          url: filePath, // Store the file path for audio files
          filename,
          size: 0, // Size not available from file path
          createdAt: new Date(),
        };

        setCurrentMedia(mediaItem);
        setMessages([]);
      } else {
        // Handle image file
        const img = new Image();

        img.onload = () => {
          const mediaItem: MediaItem = {
            id: crypto.randomUUID(),
            type: 'image',
            url,
            filename,
            size: 0, // Size not available from file path
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

        img.onerror = () => {
          console.error('Failed to load image from:', filePath);
          alert(`Failed to load image: ${filename}`);
        };

        img.src = url;
      }
    });

    return () => {
      unlisten.then(fn => fn());
    };
  }, [currentMedia]);

  // Load model when currentModel changes
  useEffect(() => {
    const loadModel = async () => {
      if (!currentModel || !currentModel.downloaded) {
        return;
      }

      try {
        setIsModelLoading(true);
        console.log('Loading model:', currentModel.id);

        await invoke('load_model', {
          modelId: currentModel.id,
          nGpuLayers: 999, // Use all available GPU layers
        });

        console.log('Model loaded successfully');

        // Check if model supports audio
        try {
          const supportsAudio = await invoke<boolean>('check_audio_support');
          setIsAudioCapable(supportsAudio);
          console.log('Model audio support:', supportsAudio);
        } catch (error) {
          console.error('Failed to check audio support:', error);
          setIsAudioCapable(false);
        }

        setIsModelLoading(false);
      } catch (error) {
        console.error('Failed to load model:', error);
        alert(`Failed to load model: ${error}`);
        setIsModelLoading(false);
        setIsAudioCapable(false);
      }
    };

    loadModel();
  }, [currentModel]);

  const handleMediaDrop = (files: File[]) => {
    if (files.length === 0) return;

    const file = files[0];

    // Check if it's an audio file
    if (file.type.startsWith('audio/')) {
      alert('Audio files must be opened via File → Open menu (Cmd/Ctrl+O)');
      return;
    }

    // Handle image/video files
    const url = URL.createObjectURL(file);
    const img = new Image();

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

  const handleImageCrop = (croppedImageUrl: string) => {
    // Create a new image to get dimensions
    const img = new Image();
    img.onload = () => {
      setCurrentMedia(prev => prev ? {
        ...prev,
        url: croppedImageUrl,
        dimensions: {
          width: img.width,
          height: img.height,
        },
      } : prev);
    };
    img.src = croppedImageUrl;
  };

  const handleLoadImage = async () => {
    try {
      // If there's already an image loaded, confirm before resetting session
      if (currentMedia) {
        const confirmed = await ask(
          'Loading a new image will reset the current session and clear the chat history. Do you want to continue?',
          {
            title: 'Reset Session?',
            kind: 'warning',
          }
        );

        if (!confirmed) {
          return; // User cancelled
        }
      }

      const filePath = await open({
        multiple: false,
        directory: false,
        filters: [
          {
            name: 'Images',
            extensions: ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'],
          },
        ],
      });

      if (!filePath) {
        return; // User cancelled
      }

      const url = convertFileSrc(filePath);
      const filename = filePath.split('/').pop() || filePath.split('\\').pop() || 'image';

      // Create image element to get dimensions
      const img = new Image();

      img.onload = () => {
        const mediaItem: MediaItem = {
          id: crypto.randomUUID(),
          type: 'image',
          url,
          filename,
          size: 0,
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

      img.onerror = () => {
        console.error('Failed to load image from:', filePath);
        alert(`Failed to load image: ${filename}`);
      };

      img.src = url;
    } catch (error) {
      console.error('Failed to open file dialog:', error);
    }
  };

  const handleSendMessage = async (content: string) => {
    if (!currentMedia || !currentModel) {
      return;
    }

    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content,
      timestamp: new Date(),
      mediaId: currentMedia.id,
    };

    setMessages(prev => [...prev, userMessage]);
    setIsGenerating(true);

    try {
      let response: string;

      if (currentMedia.type === 'audio') {
        // Handle audio file
        console.log('Calling inference with audio file:', currentMedia.url);
        response = await invoke<string>('generate_response_audio', {
          prompt: content,
          audioPath: currentMedia.url,
        });
      } else {
        // Handle image file
        const img = new Image();
        img.crossOrigin = 'anonymous';

        await new Promise((resolve, reject) => {
          img.onload = resolve;
          img.onerror = reject;
          img.src = currentMedia.url;
        });

        // Create canvas and get RGB data
        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d');

        if (!ctx) {
          throw new Error('Failed to get canvas context');
        }

        ctx.drawImage(img, 0, 0);
        const imageData = ctx.getImageData(0, 0, img.width, img.height);

        // Convert RGBA to RGB
        const rgbData: number[] = [];
        for (let i = 0; i < imageData.data.length; i += 4) {
          rgbData.push(imageData.data[i]);     // R
          rgbData.push(imageData.data[i + 1]); // G
          rgbData.push(imageData.data[i + 2]); // B
        }

        console.log('Calling inference with image:', img.width, 'x', img.height);

        // Build conversation history for backend (all messages + new user message)
        const allMessages = [...messages, userMessage];
        const conversation = allMessages.map(msg => ({
          role: msg.role,
          content: msg.content,
        }));

        console.log('Sending conversation with', conversation.length, 'messages');

        // Call inference with full conversation
        response = await invoke<string>('generate_response', {
          conversation,
          imageData: rgbData,
          imageWidth: img.width,
          imageHeight: img.height,
        });
      }

      const assistantMessage: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: response,
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, assistantMessage]);
      setIsGenerating(false);
    } catch (error) {
      console.error('Failed to generate response:', error);
      const errorMessage: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: `Error: ${error}`,
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
      setIsGenerating(false);
    }
  };

  const handleSelectModel = (modelId: string) => {
    const model = downloadedModels.find(m => m.id === modelId);
    if (model) {
      setCurrentModel(model);
      // Clear messages when switching models
      setMessages([]);
    }
  };

  const handleDownloadModel = async (modelId: string) => {
    console.log('Download model:', modelId);

    // Find the model in available models to get the repo
    const model = MOCK_AVAILABLE_MODELS.find(m => m.id === modelId);
    if (!model) {
      alert('Model not found');
      return;
    }

    // Extract repo from huggingface URL (e.g., "ggml-org/ultravox-v0_5-llama-3_2-1b-GGUF")
    const url = new URL(model.huggingfaceUrl);
    const pathParts = url.pathname.split('/');
    const repo = `${pathParts[1]}/${pathParts[2]}`; // org/repo

    try {
      setIsDownloading(true);
      setDownloadProgress(undefined);

      // Call the download_model command
      const downloadedModelId = await invoke<string>('download_model', { repo });

      // Model download complete
      setIsDownloading(false);
      setIsModelModalOpen(false);

      // Create the downloaded model object
      const downloadedModel: Model = {
        ...model,
        id: downloadedModelId,
        downloaded: true,
      };

      // Add to downloaded models list
      setDownloadedModels(prev => [...prev, downloadedModel]);

      // Load the downloaded model
      setCurrentModel(downloadedModel);

      alert(`Model downloaded successfully!\nModel ID: ${downloadedModelId}`);
    } catch (error) {
      console.error('Failed to download model:', error);
      alert(`Failed to download model: ${error}`);
      setIsDownloading(false);
    }
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

  const handleDownloadBundledModel = async () => {
    try {
      setIsDownloading(true);
      setDownloadProgress(undefined);

      await invoke('download_bundled_model');

      // Download complete, update model state
      const updatedModel = { ...BUNDLED_MODEL, downloaded: true };
      setCurrentModel(updatedModel);
      setIsDownloadDialogOpen(false);
      setIsDownloading(false);
    } catch (error) {
      console.error('Failed to download bundled model:', error);
      alert(`Failed to download model: ${error}`);
      setIsDownloading(false);
    }
  };

  const handleCancelDownload = () => {
    // User canceled download, close dialog and leave currentModel as undefined
    setIsDownloadDialogOpen(false);
  };

  return (
    <Layout>
      <ImageViewer
        mediaItem={currentMedia}
        onMediaDrop={handleMediaDrop}
        onLoadImage={handleLoadImage}
        onImageCrop={handleImageCrop}
      />
      <ChatPanel
        currentModel={currentModel}
        messages={messages}
        onChangeModel={() => setIsModelModalOpen(true)}
        onSendMessage={handleSendMessage}
        isAudioCapable={isAudioCapable}
        isGenerating={isGenerating}
      />
      <ModelSelectionModal
        isOpen={isModelModalOpen}
        onClose={() => setIsModelModalOpen(false)}
        currentModel={currentModel}
        downloadedModels={downloadedModels}
        availableModels={MOCK_AVAILABLE_MODELS}
        onSelectModel={handleSelectModel}
        onDownloadModel={handleDownloadModel}
        onAddModel={handleAddModel}
      />
      <DownloadModelDialog
        isOpen={isDownloadDialogOpen}
        onDownload={handleDownloadBundledModel}
        onCancel={handleCancelDownload}
        progress={downloadProgress}
        isDownloading={isDownloading}
      />
    </Layout>
  );
}

export default App;
