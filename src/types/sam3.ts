// SAM3 (Segment Anything 3) type definitions

// Point prompt for SAM3 segmentation
export interface Sam3Point {
  x: number;
  y: number;
  label: 0 | 1; // 0 = background (exclude), 1 = foreground (include)
}

// Bounding box prompt for SAM3 segmentation
export interface Sam3Box {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

// Collection of prompts for SAM3
export interface Sam3Prompts {
  points: Sam3Point[];
  box: Sam3Box | null;
}

// Result from SAM3 segmentation (matches Rust Sam3SegmentResult)
export interface Sam3SegmentResult {
  masks: string[]; // 3 base64-encoded PNG masks
  iou_scores: number[]; // IoU confidence scores for each mask
  object_score: number; // Object presence probability (0-1)
  width: number;
  height: number;
}

// SAM3 UI state
export interface Sam3State {
  // Loading states
  isEnabled: boolean;
  isLoading: boolean; // Model loading
  isEncoding: boolean; // Vision encoder running
  isSegmenting: boolean; // Decoder running

  // Model state
  isModelDownloaded: boolean;
  isModelLoaded: boolean;

  // Prompts and results
  prompts: Sam3Prompts;
  segmentResult: Sam3SegmentResult | null;

  // UI configuration
  selectedMaskIndex: number; // 0, 1, or 2
  overlayOpacity: number; // 0-100
  overlayColor: string; // Hex color for mask overlay
}

// Initial SAM3 state
export const initialSam3State: Sam3State = {
  isEnabled: false,
  isLoading: false,
  isEncoding: false,
  isSegmenting: false,
  isModelDownloaded: false,
  isModelLoaded: false,
  prompts: {
    points: [],
    box: null,
  },
  segmentResult: null,
  selectedMaskIndex: 0,
  overlayOpacity: 50,
  overlayColor: '#00ff00', // Green
};

// SAM3 overlay props
export interface Sam3OverlayProps {
  imageElement: HTMLImageElement | null;
  imageWidth: number;
  imageHeight: number;
  state: Sam3State;
  onAddPoint: (point: Sam3Point) => void;
  onSetBox: (box: Sam3Box | null) => void;
  onClearPrompts: () => void;
}

// SAM3 controls props
export interface Sam3ControlsProps {
  state: Sam3State;
  onToggle: () => void;
  onMaskIndexChange: (index: number) => void;
  onOpacityChange: (opacity: number) => void;
  onClearPrompts: () => void;
  onExportMask: () => void;
  onCropToMask: () => void;
  onDownloadModel: () => void;
}
