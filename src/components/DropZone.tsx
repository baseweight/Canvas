import React, { useState, useCallback } from 'react';
import './DropZone.css';

interface DropZoneProps {
  onMediaDrop: (files: File[]) => void;
}

export const DropZone: React.FC<DropZoneProps> = ({ onMediaDrop }) => {
  const [isDragging, setIsDragging] = useState(false);

  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();

    // Only set dragging to false if we're leaving the drop zone entirely
    if (e.currentTarget === e.target) {
      setIsDragging(false);
    }
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = Array.from(e.dataTransfer.files);

    // Filter for images and videos only
    const mediaFiles = files.filter(file =>
      file.type.startsWith('image/') || file.type.startsWith('video/')
    );

    if (mediaFiles.length > 0) {
      onMediaDrop(mediaFiles);
    }
  }, [onMediaDrop]);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      onMediaDrop(Array.from(files));
    }
  }, [onMediaDrop]);

  return (
    <div
      className={`bw-dropzone ${isDragging ? 'bw-dropzone--dragging' : ''}`}
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
    >
      <div className="bw-dropzone-content">
        <svg className="bw-dropzone-icon" viewBox="0 0 36 36" aria-hidden="true">
          <path d="M34 12v-2c0-2.2-1.8-4-4-4h-8V2h-8v4h-8C3.8 6 2 7.8 2 10v2h32zm0 4H2v14c0 2.2 1.8 4 4 4h24c2.2 0 4-1.8 4-4V16zm-16 8v-4h-4v-2h4v-4h2v4h4v2h-4v4h-2z"></path>
        </svg>

        <h2 className="bw-dropzone-heading">
          Drag and drop images or videos
        </h2>

        <p className="bw-dropzone-text">
          or click to browse
        </p>

        <input
          type="file"
          className="bw-dropzone-input"
          accept="image/*,video/*"
          multiple
          onChange={handleFileInput}
          aria-label="Upload images or videos"
        />
      </div>
    </div>
  );
};
