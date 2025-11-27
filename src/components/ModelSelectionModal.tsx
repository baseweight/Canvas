import React, { useState } from 'react';
import type { Model, AvailableModel } from '../types';
import { AddModelDialog } from './AddModelDialog';
import './ModelSelectionModal.css';

interface ModelSelectionModalProps {
  isOpen: boolean;
  onClose: () => void;
  currentModel?: Model;
  downloadedModels: Model[];
  availableModels: AvailableModel[];
  onSelectModel: (modelId: string) => void;
  onDownloadModel: (modelId: string) => void;
  onAddModel: (repo: string, quantization: string) => void;
}

export const ModelSelectionModal: React.FC<ModelSelectionModalProps> = ({
  isOpen,
  onClose,
  currentModel,
  downloadedModels,
  availableModels,
  onSelectModel,
  onDownloadModel,
  onAddModel,
}) => {
  const [activeTab, setActiveTab] = useState<'downloaded' | 'available'>('downloaded');
  const [isAddDialogOpen, setIsAddDialogOpen] = useState(false);

  if (!isOpen) return null;

  const getTaskBadgeClass = (task: string) => {
    const taskMap: Record<string, string> = {
      'general-vlm': 'general',
      'ocr': 'ocr',
      'classifier': 'classifier',
      'captioning': 'captioning',
      'vqa': 'vqa',
      'detection': 'detection',
    };
    return taskMap[task] || 'general';
  };

  const formatSize = (bytes: number) => {
    const gb = bytes / (1024 * 1024 * 1024);
    return `${gb.toFixed(1)} GB`;
  };

  return (
    <>
      <div className="bw-modal-backdrop" onClick={onClose} />
      <div className="bw-modal">
        <div className="bw-modal-header">
          <h2 className="bw-modal-title">Select Model</h2>
          <div className="bw-modal-header-actions">
            <button
              className="bw-button-primary bw-button-small"
              onClick={() => setIsAddDialogOpen(true)}
            >
              Add Model
            </button>
            <button className="bw-modal-close" onClick={onClose} aria-label="Close">
              <svg viewBox="0 0 24 24" width="20" height="20">
                <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z" fill="currentColor"/>
              </svg>
            </button>
          </div>
        </div>

        <div className="bw-modal-tabs">
          <button
            className={`bw-modal-tab ${activeTab === 'downloaded' ? 'bw-modal-tab--active' : ''}`}
            onClick={() => setActiveTab('downloaded')}
          >
            Downloaded Models ({downloadedModels.length})
          </button>
          <button
            className={`bw-modal-tab ${activeTab === 'available' ? 'bw-modal-tab--active' : ''}`}
            onClick={() => setActiveTab('available')}
          >
            Recommended Models ({availableModels.length})
          </button>
        </div>

        <div className="bw-modal-content">
          {activeTab === 'downloaded' ? (
            <div className="bw-model-list">
              {downloadedModels.length === 0 ? (
                <div className="bw-model-empty">
                  <p>No models downloaded yet</p>
                  <button
                    className="bw-button-secondary"
                    onClick={() => setActiveTab('available')}
                  >
                    Browse Recommended Models
                  </button>
                </div>
              ) : (
                downloadedModels.map(model => (
                  <div
                    key={model.id}
                    className={`bw-model-card ${currentModel?.id === model.id ? 'bw-model-card--selected' : ''}`}
                  >
                    <div className="bw-model-card-header">
                      <div>
                        <h3 className="bw-model-name">
                          {model.displayName}
                          {model.bundled && (
                            <span className="bw-model-bundled-badge">Bundled</span>
                          )}
                        </h3>
                        <div className="bw-model-meta">
                          <span className={`bw-model-task bw-model-task--${getTaskBadgeClass(model.task)}`}>
                            {model.taskDescription}
                          </span>
                          <span className="bw-model-backend">{model.backend}</span>
                          {model.quantization && (
                            <span className="bw-model-quant">{model.quantization}</span>
                          )}
                        </div>
                      </div>
                      <div className="bw-model-size">{formatSize(model.size)}</div>
                    </div>

                    <div className="bw-model-card-footer">
                      <a
                        href={model.huggingfaceUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="bw-model-link"
                      >
                        View on HuggingFace →
                      </a>
                      {currentModel?.id === model.id ? (
                        <span className="bw-model-current">Currently Loaded</span>
                      ) : (
                        <button
                          className="bw-button-primary"
                          onClick={() => {
                            onSelectModel(model.id);
                            onClose();
                          }}
                        >
                          Load Model
                        </button>
                      )}
                    </div>
                  </div>
                ))
              )}
            </div>
          ) : (
            <div className="bw-model-list">
              {availableModels.map(model => {
                const isDownloaded = downloadedModels.some(m => m.id === model.id);
                return (
                  <div key={model.id} className="bw-model-card">
                    <div className="bw-model-card-header">
                      <div>
                        <h3 className="bw-model-name">{model.displayName}</h3>
                        <div className="bw-model-meta">
                          <span className={`bw-model-task bw-model-task--${getTaskBadgeClass(model.task)}`}>
                            {model.taskDescription}
                          </span>
                          <span className="bw-model-backend">{model.backend}</span>
                          {model.quantization && (
                            <span className="bw-model-quant">{model.quantization}</span>
                          )}
                        </div>
                        {model.description && (
                          <p className="bw-model-description">{model.description}</p>
                        )}
                      </div>
                      <div className="bw-model-size">{formatSize(model.size)}</div>
                    </div>

                    <div className="bw-model-card-footer">
                      <a
                        href={model.huggingfaceUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="bw-model-link"
                      >
                        View on HuggingFace →
                      </a>
                      {isDownloaded ? (
                        <button
                          className="bw-button-success"
                          onClick={() => {
                            onSelectModel(model.id);
                            onClose();
                          }}
                        >
                          Load Model
                        </button>
                      ) : (
                        <button
                          className="bw-button-primary"
                          onClick={() => onDownloadModel(model.id)}
                        >
                          Download Model
                        </button>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>

      <AddModelDialog
        isOpen={isAddDialogOpen}
        onClose={() => setIsAddDialogOpen(false)}
        onAdd={(repo, quantization) => {
          onAddModel(repo, quantization);
          setIsAddDialogOpen(false);
        }}
      />
    </>
  );
};
