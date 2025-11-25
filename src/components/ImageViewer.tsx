import React from 'react';
import type { MediaItem } from '../types';
import type { NormalizedSelection } from '../types/selection';
import { DropZone } from './DropZone';
import { Toolbar, type ToolType } from './Toolbar';
import { SelectionOverlay } from './SelectionOverlay';
import './ImageViewer.css';

interface ImageViewerProps {
  mediaItem?: MediaItem;
  onMediaDrop: (files: File[]) => void;
  onLoadImage?: () => void;
  onImageCrop?: (croppedImageUrl: string) => void;
}

export const ImageViewer: React.FC<ImageViewerProps> = ({ mediaItem, onMediaDrop, onLoadImage, onImageCrop }) => {
  const [activeTool, setActiveTool] = React.useState<ToolType | undefined>(undefined);
  const [selection, setSelection] = React.useState<NormalizedSelection | null>(null);
  const imageRef = React.useRef<HTMLImageElement>(null);

  const handleToolSelect = (tool: ToolType) => {
    console.log('Tool selected:', tool, { hasSelection: !!selection, hasImageRef: !!imageRef.current });
    setActiveTool(tool);

    // Handle tool actions
    if (tool === 'load' && onLoadImage) {
      onLoadImage();
    } else if (tool === 'crop' && selection && imageRef.current) {
      console.log('Starting crop with selection:', selection);
      handleCrop();
    } else if (tool === 'crop') {
      console.log('Crop clicked but conditions not met:', {
        hasSelection: !!selection,
        hasImageRef: !!imageRef.current,
        hasCallback: !!onImageCrop
      });
    }
  };

  const handleCrop = () => {
    if (!selection || !imageRef.current || !onImageCrop) {
      return;
    }

    const img = imageRef.current;

    // Calculate the scale factor between displayed size and actual image size
    const scaleX = img.naturalWidth / img.width;
    const scaleY = img.naturalHeight / img.height;

    // Create canvas for cropping
    const canvas = document.createElement('canvas');
    canvas.width = selection.width * scaleX;
    canvas.height = selection.height * scaleY;
    const ctx = canvas.getContext('2d');

    if (!ctx) {
      console.error('Failed to get canvas context');
      return;
    }

    // Draw the cropped portion
    ctx.drawImage(
      img,
      selection.x * scaleX,
      selection.y * scaleY,
      selection.width * scaleX,
      selection.height * scaleY,
      0,
      0,
      canvas.width,
      canvas.height
    );

    // Convert to blob and create URL
    canvas.toBlob((blob) => {
      if (blob) {
        const croppedUrl = URL.createObjectURL(blob);
        onImageCrop(croppedUrl);
        // Clear selection after crop
        setSelection(null);
        setActiveTool(undefined);
      }
    }, 'image/png');
  };

  const handleSelectionChange = (newSelection: NormalizedSelection | null) => {
    setSelection(newSelection);
  };

  if (!mediaItem) {
    return (
      <div className="bw-image-viewer bw-image-viewer--empty">
        <DropZone onMediaDrop={onMediaDrop} />
      </div>
    );
  }

  return (
    <div className="bw-image-viewer">
      <Toolbar activeTool={activeTool} hasSelection={!!selection} onToolSelect={handleToolSelect} />
      <div className="bw-image-viewer-content">
        <div className="bw-image-container">
          {mediaItem.type === 'image' ? (
            <div className="bw-image-wrapper">
              <img
                ref={imageRef}
                src={mediaItem.url}
                alt={mediaItem.filename}
                className="bw-image"
                crossOrigin="anonymous"
              />
              <SelectionOverlay
                imageElement={imageRef.current}
                isActive={activeTool === 'select'}
                onSelectionChange={handleSelectionChange}
              />
            </div>
          ) : (
            <video
              src={mediaItem.url}
              controls
              className="bw-image"
            />
          )}
        </div>
        <div className="bw-image-info">
          <span className="bw-image-filename">{mediaItem.filename}</span>
          {mediaItem.dimensions && (
            <span className="bw-image-dimensions">
              {mediaItem.dimensions.width} × {mediaItem.dimensions.height}
            </span>
          )}
        </div>
      </div>
    </div>
  );
};
