import React from 'react';
import type { MediaItem } from '../types';
import { DropZone } from './DropZone';
import './ImageViewer.css';

interface ImageViewerProps {
  mediaItem?: MediaItem;
  onMediaDrop: (files: File[]) => void;
}

export const ImageViewer: React.FC<ImageViewerProps> = ({ mediaItem, onMediaDrop }) => {
  if (!mediaItem) {
    return (
      <div className="bw-image-viewer bw-image-viewer--empty">
        <DropZone onMediaDrop={onMediaDrop} />
      </div>
    );
  }

  return (
    <div className="bw-image-viewer">
      <div className="bw-image-container">
        {mediaItem.type === 'image' ? (
          <img
            src={mediaItem.url}
            alt={mediaItem.filename}
            className="bw-image"
          />
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
  );
};
