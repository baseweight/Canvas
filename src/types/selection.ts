export interface SelectionRect {
  startX: number;
  startY: number;
  endX: number;
  endY: number;
}

export interface NormalizedSelection {
  x: number;
  y: number;
  width: number;
  height: number;
}

export function normalizeSelection(rect: SelectionRect): NormalizedSelection {
  const x = Math.min(rect.startX, rect.endX);
  const y = Math.min(rect.startY, rect.endY);
  const width = Math.abs(rect.endX - rect.startX);
  const height = Math.abs(rect.endY - rect.startY);

  return { x, y, width, height };
}
