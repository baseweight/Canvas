import React from 'react';
import './DrawingOverlay.css';

interface Point {
  x: number;
  y: number;
}

interface DrawingOverlayProps {
  imageElement: HTMLImageElement | null;
  isActive: boolean;
  brushColor?: string;
  brushSize?: number;
  onApplyDrawing?: (canvasElement: HTMLCanvasElement) => void;
}

export interface DrawingOverlayRef {
  applyDrawing: () => void;
  hasDrawing: () => boolean;
  clearDrawing: () => void;
}

export const DrawingOverlay = React.forwardRef<DrawingOverlayRef, DrawingOverlayProps>(({
  imageElement,
  isActive,
  brushColor = '#FF0000',
  brushSize = 5,
  onApplyDrawing,
}, ref) => {
  const [isDrawing, setIsDrawing] = React.useState(false);
  const [currentPath, setCurrentPath] = React.useState<Point[]>([]);
  const [hasAnyDrawing, setHasAnyDrawing] = React.useState(false);
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const overlayRef = React.useRef<HTMLDivElement>(null);

  // Expose methods via ref
  React.useImperativeHandle(ref, () => ({
    applyDrawing: () => {
      if (canvasRef.current && onApplyDrawing && hasAnyDrawing) {
        onApplyDrawing(canvasRef.current);
        setHasAnyDrawing(false);
      }
    },
    hasDrawing: () => hasAnyDrawing,
    clearDrawing: () => {
      if (canvasRef.current) {
        const ctx = canvasRef.current.getContext('2d');
        if (ctx) {
          ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
        }
        setHasAnyDrawing(false);
      }
    },
  }));

  // Initialize canvas when image element changes
  React.useEffect(() => {
    if (!imageElement || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const rect = imageElement.getBoundingClientRect();

    // Set canvas size to match image display size
    canvas.width = rect.width;
    canvas.height = rect.height;

    // Clear canvas
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  }, [imageElement]);

  const getMousePosition = (e: React.MouseEvent<HTMLCanvasElement>): Point | null => {
    if (!canvasRef.current) return null;

    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    return { x, y };
  };

  const drawLine = (ctx: CanvasRenderingContext2D, from: Point, to: Point) => {
    ctx.strokeStyle = brushColor;
    ctx.lineWidth = brushSize;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    ctx.beginPath();
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(to.x, to.y);
    ctx.stroke();
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isActive) return;

    const pos = getMousePosition(e);
    if (!pos) return;

    setIsDrawing(true);
    setCurrentPath([pos]);
    setHasAnyDrawing(true);

    // Draw initial point
    const ctx = canvasRef.current?.getContext('2d');
    if (ctx) {
      ctx.fillStyle = brushColor;
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, brushSize / 2, 0, 2 * Math.PI);
      ctx.fill();
    }
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing || !isActive) return;

    const pos = getMousePosition(e);
    if (!pos) return;

    const ctx = canvasRef.current?.getContext('2d');
    if (!ctx || currentPath.length === 0) return;

    const lastPoint = currentPath[currentPath.length - 1];
    drawLine(ctx, lastPoint, pos);

    setCurrentPath([...currentPath, pos]);
  };

  const handleMouseUp = () => {
    if (!isDrawing) return;
    setIsDrawing(false);
    setCurrentPath([]);
  };

  const handleMouseLeave = () => {
    if (isDrawing) {
      handleMouseUp();
    }
  };

  // Handle Escape key to clear drawing
  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && canvasRef.current) {
        const ctx = canvasRef.current.getContext('2d');
        if (ctx) {
          ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
        }
      }
    };

    if (isActive) {
      window.addEventListener('keydown', handleKeyDown);
      return () => window.removeEventListener('keydown', handleKeyDown);
    }
  }, [isActive]);

  if (!isActive || !imageElement) {
    return null;
  }

  return (
    <div ref={overlayRef} className="bw-drawing-overlay">
      <canvas
        ref={canvasRef}
        className="bw-drawing-canvas"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
      />
    </div>
  );
});
