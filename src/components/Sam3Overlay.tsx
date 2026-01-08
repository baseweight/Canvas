import React, { useCallback, useRef, useState } from 'react';
import type { Sam3Point, Sam3Box, Sam3SegmentResult, Sam3Prompts } from '../types';
import './Sam3Overlay.css';

interface Sam3OverlayProps {
  imageElement: HTMLImageElement | null;
  imageWidth: number;
  imageHeight: number;
  isEnabled: boolean;
  isEncoding: boolean;
  isSegmenting: boolean;
  segmentResult: Sam3SegmentResult | null;
  selectedMaskIndex: number;
  overlayOpacity: number;
  overlayColor: string;
  prompts: Sam3Prompts;
  onAddPoint: (point: Sam3Point) => void;
  onSetBox: (box: Sam3Box | null) => void;
  onClearPrompts: () => void;
}

export const Sam3Overlay: React.FC<Sam3OverlayProps> = ({
  imageElement,
  imageWidth,
  imageHeight,
  isEnabled,
  isEncoding,
  isSegmenting,
  segmentResult,
  selectedMaskIndex,
  overlayOpacity,
  overlayColor,
  prompts,
  onAddPoint,
  onSetBox,
}) => {
  const overlayRef = useRef<HTMLDivElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState<{ x: number; y: number } | null>(null);
  const [currentBox, setCurrentBox] = useState<Sam3Box | null>(null);

  // Convert screen coordinates to image coordinates
  const screenToImageCoords = useCallback((clientX: number, clientY: number): { x: number; y: number } | null => {
    if (!imageElement || !overlayRef.current) return null;

    const rect = overlayRef.current.getBoundingClientRect();
    const scaleX = imageWidth / rect.width;
    const scaleY = imageHeight / rect.height;

    const x = (clientX - rect.left) * scaleX;
    const y = (clientY - rect.top) * scaleY;

    // Clamp to image bounds
    return {
      x: Math.max(0, Math.min(imageWidth, x)),
      y: Math.max(0, Math.min(imageHeight, y)),
    };
  }, [imageElement, imageWidth, imageHeight]);

  // Handle mouse down - start potential drag
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (!isEnabled || isEncoding || isSegmenting) return;

    const coords = screenToImageCoords(e.clientX, e.clientY);
    if (!coords) return;

    setIsDragging(true);
    setDragStart(coords);
    setCurrentBox(null);
  }, [isEnabled, isEncoding, isSegmenting, screenToImageCoords]);

  // Handle mouse move - update drag box
  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!isDragging || !dragStart) return;

    const coords = screenToImageCoords(e.clientX, e.clientY);
    if (!coords) return;

    // If we've moved more than 10px, start drawing a box
    const dx = Math.abs(coords.x - dragStart.x);
    const dy = Math.abs(coords.y - dragStart.y);

    if (dx > 10 || dy > 10) {
      setCurrentBox({
        x1: Math.min(dragStart.x, coords.x),
        y1: Math.min(dragStart.y, coords.y),
        x2: Math.max(dragStart.x, coords.x),
        y2: Math.max(dragStart.y, coords.y),
      });
    }
  }, [isDragging, dragStart, screenToImageCoords]);

  // Handle mouse up - finalize action
  const handleMouseUp = useCallback((e: React.MouseEvent) => {
    if (!isEnabled || !isDragging || !dragStart) {
      setIsDragging(false);
      setDragStart(null);
      return;
    }

    const coords = screenToImageCoords(e.clientX, e.clientY);
    if (!coords) {
      setIsDragging(false);
      setDragStart(null);
      setCurrentBox(null);
      return;
    }

    // Check if this was a click (no significant movement) or a drag
    const dx = Math.abs(coords.x - dragStart.x);
    const dy = Math.abs(coords.y - dragStart.y);

    if (dx < 10 && dy < 10) {
      // Click - add a point
      const isRightClick = e.button === 2;
      onAddPoint({
        x: coords.x,
        y: coords.y,
        label: isRightClick ? 0 : 1, // 0 = background, 1 = foreground
      });
    } else if (currentBox) {
      // Drag - set bounding box
      onSetBox(currentBox);
    }

    setIsDragging(false);
    setDragStart(null);
    setCurrentBox(null);
  }, [isEnabled, isDragging, dragStart, currentBox, screenToImageCoords, onAddPoint, onSetBox]);

  // Handle right-click for background points
  const handleContextMenu = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    if (!isEnabled || isEncoding || isSegmenting) return;

    const coords = screenToImageCoords(e.clientX, e.clientY);
    if (!coords) return;

    onAddPoint({
      x: coords.x,
      y: coords.y,
      label: 0, // background point
    });
  }, [isEnabled, isEncoding, isSegmenting, screenToImageCoords, onAddPoint]);

  // Convert image coordinates to display coordinates
  const imageToDisplayCoords = useCallback((x: number, y: number): { left: string; top: string } => {
    if (!overlayRef.current) return { left: '0%', top: '0%' };

    const leftPercent = (x / imageWidth) * 100;
    const topPercent = (y / imageHeight) * 100;

    return {
      left: `${leftPercent}%`,
      top: `${topPercent}%`,
    };
  }, [imageWidth, imageHeight]);

  // Get the selected mask as a data URL
  const maskDataUrl = segmentResult?.masks[selectedMaskIndex];

  if (!isEnabled || !imageElement) {
    return null;
  }

  return (
    <div
      ref={overlayRef}
      className="bw-sam3-overlay"
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      onContextMenu={handleContextMenu}
    >
      {/* Mask overlay */}
      {maskDataUrl && (
        <img
          src={`data:image/png;base64,${maskDataUrl}`}
          alt="Segmentation mask"
          className="bw-sam3-mask"
          style={{
            opacity: overlayOpacity / 100,
            filter: `drop-shadow(0 0 0 ${overlayColor})`,
            mixBlendMode: 'multiply',
          }}
        />
      )}

      {/* Point markers */}
      {prompts.points.map((point, index) => {
        const pos = imageToDisplayCoords(point.x, point.y);
        return (
          <div
            key={`point-${index}`}
            className={`bw-sam3-point ${point.label === 1 ? 'bw-sam3-point--foreground' : 'bw-sam3-point--background'}`}
            style={{ left: pos.left, top: pos.top }}
          />
        );
      })}

      {/* Bounding box */}
      {(prompts.box || currentBox) && (() => {
        const box = currentBox || prompts.box!;
        const topLeft = imageToDisplayCoords(box.x1, box.y1);
        return (
          <div
            className="bw-sam3-box"
            style={{
              left: topLeft.left,
              top: topLeft.top,
              width: `${((box.x2 - box.x1) / imageWidth) * 100}%`,
              height: `${((box.y2 - box.y1) / imageHeight) * 100}%`,
            }}
          />
        );
      })()}

      {/* Loading indicator */}
      {(isEncoding || isSegmenting) && (
        <div className="bw-sam3-loading">
          <span>{isEncoding ? 'Encoding image...' : 'Segmenting...'}</span>
        </div>
      )}
    </div>
  );
};
