import React from 'react';
import type { Sam3SegmentResult } from '../types';
import './Sam3Controls.css';

interface Sam3ControlsProps {
  isEnabled: boolean;
  isLoading: boolean;
  isEncoding: boolean;
  isSegmenting: boolean;
  segmentResult: Sam3SegmentResult | null;
  selectedMaskIndex: number;
  overlayOpacity: number;
  onMaskIndexChange: (index: number) => void;
  onOpacityChange: (opacity: number) => void;
  onClearPrompts: () => void;
  onExportMask: () => void;
  onCropToMask: () => void;
}

export const Sam3Controls: React.FC<Sam3ControlsProps> = ({
  isEnabled,
  isLoading,
  isEncoding,
  isSegmenting,
  segmentResult,
  selectedMaskIndex,
  overlayOpacity,
  onMaskIndexChange,
  onOpacityChange,
  onClearPrompts,
  onExportMask,
  onCropToMask,
}) => {
  if (!isEnabled) {
    return null;
  }

  const isProcessing = isLoading || isEncoding || isSegmenting;

  return (
    <div className="bw-sam3-controls">
      <div className="bw-sam3-controls-header">
        <span className="bw-sam3-controls-title">Segment Anything</span>
        {isProcessing && (
          <span className="bw-sam3-status">
            {isLoading ? 'Loading...' : isEncoding ? 'Encoding...' : 'Segmenting...'}
          </span>
        )}
      </div>

      {segmentResult && (
        <>
          {/* Mask selector */}
          <div className="bw-sam3-section">
            <span className="bw-sam3-section-label">Mask Quality</span>
            <div className="bw-sam3-mask-buttons">
              {[0, 1, 2].map((index) => (
                <button
                  key={index}
                  className={`bw-sam3-mask-btn ${selectedMaskIndex === index ? 'bw-sam3-mask-btn--active' : ''}`}
                  onClick={() => onMaskIndexChange(index)}
                  title={`IoU: ${(segmentResult.iou_scores[index] * 100).toFixed(1)}%`}
                >
                  <span className="bw-sam3-mask-label">
                    {index === 0 ? 'Fine' : index === 1 ? 'Medium' : 'Coarse'}
                  </span>
                  <span className="bw-sam3-mask-iou">
                    {(segmentResult.iou_scores[index] * 100).toFixed(0)}%
                  </span>
                </button>
              ))}
            </div>
          </div>

          {/* Object score */}
          <div className="bw-sam3-section">
            <span className="bw-sam3-section-label">Object Confidence</span>
            <div className="bw-sam3-object-score">
              <div
                className="bw-sam3-object-score-bar"
                style={{ width: `${segmentResult.object_score * 100}%` }}
              />
              <span className="bw-sam3-object-score-text">
                {(segmentResult.object_score * 100).toFixed(0)}%
              </span>
            </div>
          </div>
        </>
      )}

      {/* Opacity slider */}
      <div className="bw-sam3-section">
        <span className="bw-sam3-section-label">Overlay Opacity</span>
        <div className="bw-sam3-slider-row">
          <input
            type="range"
            min="0"
            max="100"
            value={overlayOpacity}
            onChange={(e) => onOpacityChange(parseInt(e.target.value))}
            className="bw-sam3-slider"
          />
          <span className="bw-sam3-slider-value">{overlayOpacity}%</span>
        </div>
      </div>

      {/* Action buttons */}
      <div className="bw-sam3-actions">
        <button
          className="bw-sam3-action-btn"
          onClick={onClearPrompts}
          disabled={isProcessing}
        >
          Clear
        </button>
        <button
          className="bw-sam3-action-btn"
          onClick={onExportMask}
          disabled={!segmentResult || isProcessing}
        >
          Export
        </button>
        <button
          className="bw-sam3-action-btn bw-sam3-action-btn--primary"
          onClick={onCropToMask}
          disabled={!segmentResult || isProcessing}
        >
          Crop
        </button>
      </div>

      {/* Instructions */}
      <div className="bw-sam3-instructions">
        <span>Left-click: add point</span>
        <span>Right-click: exclude point</span>
        <span>Drag: draw box</span>
      </div>
    </div>
  );
};
