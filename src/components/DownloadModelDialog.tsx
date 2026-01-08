import React from 'react';
import './DownloadModelDialog.css';

interface DownloadProgress {
  current: number;
  total: number;
  percentage: number;
  file: string;
}

interface FileProgress {
  [filename: string]: DownloadProgress;
}

interface DownloadModelDialogProps {
  isOpen: boolean;
  onDownload: () => void;
  onCancel: () => void;
  fileProgress: FileProgress;
  isDownloading: boolean;
  // Optional customization for different model types
  modelName?: string;
  modelDescription?: string;
  downloadSize?: string;
  source?: string;
  sourceRepo?: string;
}

export const DownloadModelDialog: React.FC<DownloadModelDialogProps> = ({
  isOpen,
  onDownload,
  onCancel,
  fileProgress,
  isDownloading,
  modelName = 'SmolVLM2 2.2B Instruct',
  modelDescription = 'Baseweight Canvas ships with SmolVLM2 2.2B Instruct, a compact and efficient vision-language model.',
  downloadSize = '~1.3 GB',
  source = 'HuggingFace',
  sourceRepo = 'ggml-org',
}) => {
  if (!isOpen) return null;

  const files = Object.values(fileProgress);
  const hasFiles = files.length > 0;

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
            {isDownloading ? 'Downloading Model...' : `Download ${modelName}`}
          </h2>
        </div>

        <div className="bw-modal-content">
          {!isDownloading ? (
            <div className="bw-download-info">
              <p className="bw-download-description">
                {modelDescription}
              </p>

              <div className="bw-download-details">
                <div className="bw-download-detail-row">
                  <span className="bw-download-label">Model:</span>
                  <span className="bw-download-value">{modelName}</span>
                </div>
                <div className="bw-download-detail-row">
                  <span className="bw-download-label">Download Size:</span>
                  <span className="bw-download-value">{downloadSize}</span>
                </div>
                <div className="bw-download-detail-row">
                  <span className="bw-download-label">Source:</span>
                  <span className="bw-download-value">{source} ({sourceRepo})</span>
                </div>
              </div>

              <p className="bw-download-note">
                This download is required for first-time setup. The model will be saved locally for future use.
              </p>
            </div>
          ) : (
            <div className="bw-download-progress-container">
              {hasFiles ? (
                <div className="bw-download-files-list">
                  {files.map((progress) => (
                    <div key={progress.file} className="bw-download-file-item">
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
                    </div>
                  ))}
                </div>
              ) : (
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
              onClick={onCancel}
            >
              Cancel Download
            </button>
          )}
        </div>
      </div>
    </>
  );
};
