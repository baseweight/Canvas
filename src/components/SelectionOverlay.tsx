import React from 'react';
import type { SelectionRect, NormalizedSelection } from '../types/selection';
import { normalizeSelection } from '../types/selection';
import './SelectionOverlay.css';

interface SelectionOverlayProps {
  imageElement: HTMLImageElement | null;
  isActive: boolean;
  onSelectionChange?: (selection: NormalizedSelection | null) => void;
}

export const SelectionOverlay: React.FC<SelectionOverlayProps> = ({
  imageElement,
  isActive,
  onSelectionChange,
}) => {
  const [isSelecting, setIsSelecting] = React.useState(false);
  const [selection, setSelection] = React.useState<SelectionRect | null>(null);
  const overlayRef = React.useRef<HTMLDivElement>(null);

  const getMousePosition = (e: React.MouseEvent<HTMLDivElement>): { x: number; y: number } | null => {
    if (!overlayRef.current) return null;

    const rect = overlayRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Clamp coordinates to overlay bounds (which matches image bounds)
    const clampedX = Math.max(0, Math.min(x, rect.width));
    const clampedY = Math.max(0, Math.min(y, rect.height));

    return { x: clampedX, y: clampedY };
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!isActive) return;

    const pos = getMousePosition(e);
    if (!pos) return;

    setIsSelecting(true);
    setSelection({
      startX: pos.x,
      startY: pos.y,
      endX: pos.x,
      endY: pos.y,
    });
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!isSelecting || !selection) return;

    const pos = getMousePosition(e);
    if (!pos) return;

    setSelection({
      ...selection,
      endX: pos.x,
      endY: pos.y,
    });
  };

  const handleMouseUp = () => {
    if (!isSelecting || !selection) return;

    setIsSelecting(false);

    const normalized = normalizeSelection(selection);

    // Only keep selection if it has meaningful size
    if (normalized.width > 5 && normalized.height > 5) {
      if (onSelectionChange) {
        onSelectionChange(normalized);
      }
    } else {
      // Clear selection if too small
      setSelection(null);
      if (onSelectionChange) {
        onSelectionChange(null);
      }
    }
  };

  const handleMouseLeave = () => {
    if (isSelecting) {
      handleMouseUp();
    }
  };

  // Clear selection when tool becomes inactive
  React.useEffect(() => {
    if (!isActive) {
      setSelection(null);
      setIsSelecting(false);
      if (onSelectionChange) {
        onSelectionChange(null);
      }
    }
  }, [isActive, onSelectionChange]);

  // Handle Escape key to clear selection
  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && selection) {
        setSelection(null);
        if (onSelectionChange) {
          onSelectionChange(null);
        }
      }
    };

    if (isActive) {
      window.addEventListener('keydown', handleKeyDown);
      return () => window.removeEventListener('keydown', handleKeyDown);
    }
  }, [isActive, selection, onSelectionChange]);

  if (!isActive || !imageElement) {
    return null;
  }

  const normalizedSelection = selection ? normalizeSelection(selection) : null;

  return (
    <div
      ref={overlayRef}
      className="bw-selection-overlay"
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseLeave}
    >
      {normalizedSelection && (
        <div
          className="bw-selection-rect"
          style={{
            left: `${normalizedSelection.x}px`,
            top: `${normalizedSelection.y}px`,
            width: `${normalizedSelection.width}px`,
            height: `${normalizedSelection.height}px`,
          }}
        />
      )}
    </div>
  );
};
