import { useState, useEffect, useRef } from "react";
import { invoke, convertFileSrc } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open, ask } from "@tauri-apps/plugin-dialog";
import { Layout } from "./components/Layout";
import { ImageViewer } from "./components/ImageViewer";
import { ChatPanel } from "./components/ChatPanel";
import { ModelSelectionModal } from "./components/ModelSelectionModal";
import { DownloadModelDialog } from "./components/DownloadModelDialog";
import type { MediaItem, Model, AvailableModel, OnnxModel, ChatMessage } from "./types";
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
// Commented out to avoid unused variable error - can be re-enabled for testing
// const MOCK_DOWNLOADED_MODELS: Model[] = [
//   BUNDLED_MODEL, // Always include the bundled model
// ];

// Mock available models (for download) - Real models from HuggingFace collections
const MOCK_AVAILABLE_MODELS: AvailableModel[] = [
  // SmolVLM2 2.2B Instruct - Best compact vision-language model
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

  // Ultravox - Audio-capable language model
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

  // LFM2-VL 450M - Ultra-lightweight vision model
  {
    id: 'lfm2-vl-450m',
    name: 'LFM2-VL-450M-GGUF',
    displayName: 'LFM2-VL 450M',
    task: 'general-vlm',
    taskDescription: 'General Vision-Language Model',
    backend: 'llama.cpp',
    huggingfaceUrl: 'https://huggingface.co/LiquidAI/LFM2-VL-450M-GGUF',
    size: 0.3 * 1024 * 1024 * 1024,
    quantization: 'Q4_0',
    description: 'Ultra-lightweight 450M model for resource-constrained environments',
  },

  // Ministral 3 14B - Mistral's latest vision model
  {
    id: 'mistralai_ministral_3_14b_instruct_2512_gguf',
    name: 'Ministral-3-14B-Instruct-2512-GGUF',
    displayName: 'Ministral 3 14B Instruct',
    task: 'general-vlm',
    taskDescription: 'General Vision-Language Model',
    backend: 'llama.cpp',
    huggingfaceUrl: 'https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512-GGUF',
    size: 8.5 * 1024 * 1024 * 1024,
    quantization: 'Q4_K_M (BF16 mmproj)',
    description: 'Mistral\'s latest 14B vision-language model with unified architecture and strong performance',
  },

];

// ONNX models for download
const ONNX_MODELS: OnnxModel[] = [
  {
    id: 'smolvlm2-256m-video-instruct-onnx',
    name: 'SmolVLM2-256M-Video-Instruct-ONNX',
    displayName: 'SmolVLM2 256M Video Instruct (ONNX)',
    task: 'general-vlm',
    taskDescription: 'General Vision-Language Model',
    huggingfaceRepo: 'HuggingFaceTB/SmolVLM2-256M-Video-Instruct',
    huggingfaceUrl: 'https://huggingface.co/HuggingFaceTB/SmolVLM2-256M-Video-Instruct',
    quantizations: ['Q4', 'Q8', 'FP16'],
    estimatedSizes: {
      'Q4': 0.3 * 1024 * 1024 * 1024,
      'Q8': 0.5 * 1024 * 1024 * 1024,
      'FP16': 0.8 * 1024 * 1024 * 1024,
    },
    description: 'SmolVLM2 256M ONNX Runtime backend - compact model optimized for GPU acceleration',
  },
];

interface DownloadProgress {
  current: number;
  total: number;
  percentage: number;
  file: string;
}

interface FileProgress {
  [filename: string]: DownloadProgress;
}

function App() {
  const [currentMedia, setCurrentMedia] = useState<MediaItem | undefined>(undefined);
  const [currentModel, setCurrentModel] = useState<Model | undefined>(undefined);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isModelModalOpen, setIsModelModalOpen] = useState(false);
  const [isDownloadDialogOpen, setIsDownloadDialogOpen] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);
  const [fileProgress, setFileProgress] = useState<FileProgress>({});
  const [_isModelLoading, setIsModelLoading] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isAudioCapable, setIsAudioCapable] = useState(false);
  const [downloadedModels, setDownloadedModels] = useState<Model[]>([]);
  const downloadCancelledRef = useRef(false);

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
      const progress = event.payload;
      setFileProgress(prev => ({
        ...prev,
        [progress.file]: progress
      }));
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
        console.log('Downloaded model IDs:', modelIds);

        // Convert model IDs to full Model objects
        const models: Model[] = modelIds.map(modelId => {
          // Check if it's the bundled model
          if (modelId === BUNDLED_MODEL.id) {
            return BUNDLED_MODEL;
          }

          // Check if it's an ONNX model
          if (modelId.includes('_onnx_')) {
            const baseName = modelId.split('_onnx_')[0];
            const quantFromId = modelId.split('_onnx_')[1];
            console.log(`ONNX model detected: ${modelId}, baseName: ${baseName}, quant: ${quantFromId}`);

            const onnxModel = ONNX_MODELS.find(m => {
              const normalizedRepo = m.huggingfaceRepo.replace('/', '_').replace(/-/g, '_').toLowerCase();
              console.log(`  Comparing baseName "${baseName}" with normalized "${normalizedRepo}"`);
              return normalizedRepo === baseName;
            });

            console.log(`  Found ONNX model match:`, onnxModel ? onnxModel.displayName : 'NO MATCH');

            if (onnxModel) {
              return {
                id: modelId,
                name: onnxModel.name,
                displayName: `${onnxModel.displayName} (${quantFromId.toUpperCase()})`,
                task: onnxModel.task,
                taskDescription: onnxModel.taskDescription,
                backend: 'onnx-runtime' as const,
                huggingfaceUrl: onnxModel.huggingfaceUrl,
                size: onnxModel.estimatedSizes[quantFromId.toUpperCase()] || 0,
                downloaded: true,
                quantization: quantFromId.toUpperCase(),
                localPath: `/models/${modelId}`,
              } as Model;
            }
          }

          // Find in available models (GGUF models)
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
          url: url, // Converted URL for display
          filePath: filePath, // Original file path for backend inference
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

  // Listen for file-save events from the File menu
  useEffect(() => {
    console.log('Setting up file-save event listener');
    const unlisten = listen<string>('file-save', async (event) => {
      console.log('file-save event received:', event);

      if (!currentMedia || currentMedia.type !== 'image') {
        alert('No image to save. Please load an image first.');
        return;
      }

      const filePath = event.payload;
      console.log('Save file path:', filePath);

      // Determine format from file extension
      const extension = filePath.split('.').pop()?.toLowerCase() || 'png';
      const format = extension === 'jpeg' ? 'jpg' : extension;
      console.log('Save format:', format);

      try {
        // Load the current image and extract RGB data
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

        console.log('Saving image:', img.width, 'x', img.height, 'to', filePath);

        // Call the save_image command
        await invoke('save_image', {
          imageData: rgbData,
          width: img.width,
          height: img.height,
          filePath: filePath,
          format: format,
        });

        console.log('Image saved successfully');
      } catch (error) {
        console.error('Failed to save image:', error);
        alert(`Failed to save image: ${error}`);
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
        console.log('Loading model:', currentModel.id, 'backend:', currentModel.backend);

        // Load based on backend type
        if (currentModel.backend === 'onnx-runtime') {
          console.log('Loading ONNX model');
          await invoke('load_onnx_model', {
            modelId: currentModel.id,
          });
          console.log('ONNX model loaded successfully');

          // ONNX models don't support audio yet
          setIsAudioCapable(false);
        } else {
          // llama.cpp backend
          console.log('Loading llama.cpp model');
          await invoke('load_model', {
            modelId: currentModel.id,
            nGpuLayers: 999, // Use all available GPU layers
          });
          console.log('llama.cpp model loaded successfully');

          // Check if model supports audio (llama.cpp only)
          try {
            const supportsAudio = await invoke<boolean>('check_audio_support');
            setIsAudioCapable(supportsAudio);
            console.log('Model audio support:', supportsAudio);
          } catch (error) {
            console.error('Failed to check audio support:', error);
            setIsAudioCapable(false);
          }
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
        const audioPath = currentMedia.filePath || currentMedia.url;
        console.log('Calling inference with audio file:', audioPath);
        response = await invoke<string>('generate_response_audio', {
          prompt: content,
          audioPath: audioPath,
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

        // Call inference based on backend type
        if (currentModel.backend === 'onnx-runtime') {
          // ONNX Runtime - simpler single-turn for now
          // Convert RGBA to raw bytes for ONNX
          const canvas2 = document.createElement('canvas');
          canvas2.width = img.width;
          canvas2.height = img.height;
          const ctx2 = canvas2.getContext('2d');
          if (!ctx2) throw new Error('Failed to get canvas context');
          ctx2.drawImage(img, 0, 0);

          // Get image as JPEG bytes
          const blob = await new Promise<Blob>((resolve) => {
            canvas2.toBlob((b) => resolve(b!), 'image/jpeg', 0.95);
          });
          const arrayBuffer = await blob.arrayBuffer();
          const imageBytes = Array.from(new Uint8Array(arrayBuffer));

          console.log('Calling ONNX inference with image:', img.width, 'x', img.height);
          response = await invoke<string>('generate_onnx_response', {
            prompt: content,
            imageData: imageBytes,
            imageWidth: img.width,
            imageHeight: img.height,
          });
        } else {
          // llama.cpp backend - full conversation support
          response = await invoke<string>('generate_response', {
            conversation,
            imageData: rgbData,
            imageWidth: img.width,
            imageHeight: img.height,
          });
        }
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
    const quantization = model.quantization || 'Q4_K_M'; // Default to Q4_K_M if not specified

    try {
      setIsDownloading(true);
      setFileProgress({}); // Clear previous download progress
      setIsDownloadDialogOpen(true); // Show download dialog
      setIsModelModalOpen(false); // Close model selection modal
      downloadCancelledRef.current = false; // Reset cancel flag

      // Call the download_model command with quantization
      const downloadedModelId = await invoke<string>('download_model', { repo, quantization });

      // Model download complete
      setIsDownloading(false);
      setIsDownloadDialogOpen(false);

      // Refresh the entire downloaded models list from backend
      const modelIds = await invoke<string[]>('list_downloaded_models');
      const models: Model[] = modelIds.map(modelId => {
        // Check if it's the bundled model
        if (modelId === BUNDLED_MODEL.id) {
          return BUNDLED_MODEL;
        }

        // Check if it's an ONNX model
        if (modelId.includes('_onnx_')) {
          const baseName = modelId.split('_onnx_')[0];
          const quantFromId = modelId.split('_onnx_')[1];

          const onnxModel = ONNX_MODELS.find(m => {
            const normalizedRepo = m.huggingfaceRepo.replace('/', '_').replace(/-/g, '_').toLowerCase();
            return normalizedRepo === baseName;
          });

          if (onnxModel) {
            return {
              id: modelId,
              name: onnxModel.name,
              displayName: `${onnxModel.displayName} (${quantFromId.toUpperCase()})`,
              task: onnxModel.task,
              taskDescription: onnxModel.taskDescription,
              backend: 'onnx-runtime' as const,
              huggingfaceUrl: onnxModel.huggingfaceUrl,
              size: onnxModel.estimatedSizes[quantFromId.toUpperCase()] || 0,
              downloaded: true,
              quantization: quantFromId.toUpperCase(),
              localPath: `/models/${modelId}`,
            } as Model;
          }
        }

        // Find in available models (GGUF models)
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

      // Only load the model if download wasn't cancelled
      if (!downloadCancelledRef.current) {
        // Find and load the downloaded model
        const downloadedModel = models.find(m => m.id === downloadedModelId);
        if (downloadedModel) {
          setCurrentModel(downloadedModel);
        }

        alert(`Model downloaded successfully!\nModel ID: ${downloadedModelId}`);
      }
    } catch (error) {
      console.error('Failed to download model:', error);
      alert(`Failed to download model: ${error}`);
      setIsDownloading(false);
      setIsDownloadDialogOpen(false);
    }
  };

  const handleDownloadOnnxModel = async (repo: string, quantization: string) => {
    console.log('Download ONNX model:', repo, quantization);

    setIsDownloading(true);
    setIsDownloadDialogOpen(true);
    setFileProgress({});

    try {
      // Start the ONNX model download
      const modelId = await invoke<string>('download_onnx_model', {
        repo,
        quantization,
      });

      console.log('ONNX model download completed, model ID:', modelId);

      // Refresh downloaded models list
      const downloadedModelIds = await invoke<string[]>('list_downloaded_models');
      const newDownloadedModels = downloadedModelIds.map(id => {
        // Check if it's a ONNX model
        if (id.includes('_onnx_')) {
         const onnxModel = ONNX_MODELS.find(m => modelId === id || m.huggingfaceRepo.replace('/', '_').toLowerCase().includes(id.split('_onnx_')[0]));
          if (onnxModel) {
            return {
              id: modelId,
              name: onnxModel.name,
              displayName: onnxModel.displayName,
             task: onnxModel.task,
              taskDescription: onnxModel.taskDescription,
              backend: 'onnx-runtime' as const,
              huggingfaceUrl: onnxModel.huggingfaceUrl,
              size: onnxModel.estimatedSizes[quantization] || 0,
              downloaded: true,
              quantization,
              localPath: `/models/${modelId}`,
            };
          }
        }

        // Otherwise handle as GGUF model
        if (id === BUNDLED_MODEL.id) {
          return BUNDLED_MODEL;
        }

        const availableModel = MOCK_AVAILABLE_MODELS.find(m => m.id === id);
        if (availableModel) {
          return {
            ...availableModel,
            localPath: `/models/${id}`,
            downloaded: true,
          };
        }

        return null;
      }).filter((m): m is Model => m !== null);

      setDownloadedModels(newDownloadedModels);

      setIsDownloading(false);
      setIsDownloadDialogOpen(false);

      if (modelId) {
        alert(`ONNX model downloaded successfully!\nModel ID: ${modelId}`);
      }
    } catch (error) {
      console.error('Failed to download ONNX model:', error);
      alert(`Failed to download ONNX model: ${error}`);
      setIsDownloading(false);
      setIsDownloadDialogOpen(false);
    }
  };

  const handleAddModel = async (repo: string, quantization: string) => {
    console.log('Add model from HuggingFace:', repo, quantization);

    try {
      setIsDownloading(true);
      setFileProgress({}); // Clear previous download progress
      setIsDownloadDialogOpen(true); // Show download dialog
      setIsModelModalOpen(false); // Close model selection modal
      downloadCancelledRef.current = false; // Reset cancel flag

      // Call the download_model command with quantization
      const downloadedModelId = await invoke<string>('download_model', { repo, quantization });

      // Model download complete
      setIsDownloading(false);
      setIsDownloadDialogOpen(false);

      // Refresh the entire downloaded models list from backend
      const modelIds = await invoke<string[]>('list_downloaded_models');
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

        // Fallback for unknown models (custom models added via "Add Model")
        return {
          id: modelId,
          name: modelId,
          displayName: modelId.split('_').map(word =>
            word.charAt(0).toUpperCase() + word.slice(1)
          ).join(' '),
          task: 'general-vlm',
          taskDescription: 'General Vision-Language Model',
          backend: 'llama.cpp',
          huggingfaceUrl: `https://huggingface.co/${repo}`,
          size: 0,
          localPath: `/models/${modelId}`,
          downloaded: true,
          quantization: quantization,
        } as Model;
      });

      setDownloadedModels(models);

      // Only load the model if download wasn't cancelled
      if (!downloadCancelledRef.current) {
        // Find and load the downloaded model
        const downloadedModel = models.find(m => m.id === downloadedModelId);
        if (downloadedModel) {
          setCurrentModel(downloadedModel);
        }

        alert(`Model downloaded successfully!\nModel ID: ${downloadedModelId}`);
      }
    } catch (error) {
      console.error('Failed to download model:', error);
      alert(`Failed to download model: ${error}`);
      setIsDownloading(false);
      setIsDownloadDialogOpen(false);
    }
  };

  const handleDownloadBundledModel = async () => {
    try {
      setIsDownloading(true);
      setFileProgress({}); // Clear previous download progress

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

  const handleCancelDownload = async () => {
    // User canceled download, close dialog and keep current model loaded
    downloadCancelledRef.current = true;

    try {
      // Signal backend to cancel download
      await invoke('cancel_download');
    } catch (error) {
      console.error('Failed to cancel download:', error);
    }

    setIsDownloadDialogOpen(false);
    setIsDownloading(false);
    setFileProgress({});
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
        onnxModels={ONNX_MODELS}
        onSelectModel={handleSelectModel}
        onDownloadModel={handleDownloadModel}
        onDownloadOnnxModel={handleDownloadOnnxModel}
        onAddModel={handleAddModel}
      />
      <DownloadModelDialog
        isOpen={isDownloadDialogOpen}
        onDownload={handleDownloadBundledModel}
        onCancel={handleCancelDownload}
        fileProgress={fileProgress}
        isDownloading={isDownloading}
      />
    </Layout>
  );
}

export default App;
