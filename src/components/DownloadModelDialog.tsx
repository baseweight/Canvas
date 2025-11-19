import React from 'react';
import './DownloadModelDialog.css';

interface DownloadProgress {
  current: number;
  total: number;
  percentage: number;
  file: string;
}

interface DownloadModelDialogProps {
  isOpen: boolean;
  onDownload: () => void;
  onCancel: () => void;
  progress?: DownloadProgress;
  isDownloading: boolean;
}

export const DownloadModelDialog: React.FC<DownloadModelDialogProps> = ({
  isOpen,
  onDownload,
  onCancel,
  progress,
  isDownloading,
}) => {
  if (!isOpen) return null;

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  };

  return (
    <>
      <div className="bw-modal-backdrop" />
      <div className="bw-modal bw-download-modal">
        <div className="bw-modal-header">
          <h2 className="bw-modal-title">
            {isDownloading ? 'Downloading Model' : 'Download SmolVLM2 2.2B Instruct'}
          </h2>
        </div>

        <div className="bw-modal-content">
          {!isDownloading ? (
            <div className="bw-download-info">
              <p className="bw-download-description">
                Baseweight Canvas ships with SmolVLM2 2.2B Instruct, a compact and efficient vision-language model.
              </p>

              <div className="bw-download-details">
                <div className="bw-download-detail-row">
                  <span className="bw-download-label">Model:</span>
                  <span className="bw-download-value">SmolVLM2 2.2B Instruct</span>
                </div>
                <div className="bw-download-detail-row">
                  <span className="bw-download-label">Quantization:</span>
                  <span className="bw-download-value">Q4_K_M</span>
                </div>
                <div className="bw-download-detail-row">
                  <span className="bw-download-label">Download Size:</span>
                  <span className="bw-download-value">~1.3 GB</span>
                </div>
                <div className="bw-download-detail-row">
                  <span className="bw-download-label">Source:</span>
                  <span className="bw-download-value">HuggingFace (ggml-org)</span>
                </div>
              </div>

              <p className="bw-download-note">
                This download is required for first-time setup. The model will be saved locally for future use.
              </p>
            </div>
          ) : (
            <div className="bw-download-progress-container">
              {progress && (
                <>
                  <div className="bw-download-file-info">
                    <span className="bw-download-file-name">{progress.file}</span>
                    <span className="bw-download-file-size">
                      {formatBytes(progress.current)} / {formatBytes(progress.total)}
                    </span>
                  </div>

                  <div className="bw-download-progress-bar">
                    <div
                      className="bw-download-progress-fill"
                      style={{ width: `${progress.percentage}%` }}
                    />
                  </div>

                  <div className="bw-download-percentage">
                    {progress.percentage.toFixed(1)}%
                  </div>
                </>
              )}

              {!progress && (
                <div className="bw-download-spinner">
                  <div className="bw-spinner"></div>
                  <p>Initializing download...</p>
                </div>
              )}
            </div>
          )}
        </div>

        <div className="bw-modal-footer">
          {!isDownloading ? (
            <>
              <button
                className="bw-button-secondary"
                onClick={onCancel}
              >
                Skip for Now
              </button>
              <button
                className="bw-button-primary"
                onClick={onDownload}
              >
                Download Model
              </button>
            </>
          ) : (
            <button
              className="bw-button-secondary"
              disabled
            >
              Downloading...
            </button>
          )}
        </div>
      </div>
    </>
  );
};
