import React, { useState } from 'react';
import type { Model } from '../types';
import './ModelSelector.css';

interface ModelSelectorProps {
  models: Model[];
  selectedModelId?: string;
  onModelSelect: (modelId: string) => void;
}

export const ModelSelector: React.FC<ModelSelectorProps> = ({
  models,
  selectedModelId,
  onModelSelect,
}) => {
  const [isOpen, setIsOpen] = useState(false);

  const selectedModel = models.find(m => m.id === selectedModelId);

  return (
    <div className="bw-model-selector">
      <button
        className="bw-model-selector-button"
        onClick={() => setIsOpen(!isOpen)}
        aria-label="Select model"
      >
        <span className="bw-model-selector-label">
          {selectedModel ? selectedModel.displayName : 'Select a model'}
        </span>
        <svg
          className={`bw-model-selector-icon ${isOpen ? 'bw-model-selector-icon--open' : ''}`}
          viewBox="0 0 24 24"
          aria-hidden="true"
        >
          <path d="M7 10l5 5 5-5z" />
        </svg>
      </button>

      {isOpen && (
        <>
          <div
            className="bw-model-selector-backdrop"
            onClick={() => setIsOpen(false)}
          />
          <div className="bw-model-selector-dropdown">
            {models.length === 0 ? (
              <div className="bw-model-selector-empty">
                No models available
              </div>
            ) : (
              models.map(model => (
                <button
                  key={model.id}
                  className={`bw-model-selector-option ${
                    model.id === selectedModelId ? 'bw-model-selector-option--selected' : ''
                  } ${!model.downloaded ? 'bw-model-selector-option--disabled' : ''}`}
                  onClick={() => {
                    if (model.downloaded) {
                      onModelSelect(model.id);
                      setIsOpen(false);
                    }
                  }}
                  disabled={!model.downloaded}
                >
                  <div className="bw-model-selector-option-content">
                    <div className="bw-model-selector-option-name">
                      {model.displayName}
                    </div>
                    <div className="bw-model-selector-option-meta">
                      {model.backend} • {(model.size / 1024 / 1024 / 1024).toFixed(1)}GB
                      {!model.downloaded && ' • Not downloaded'}
                    </div>
                  </div>
                </button>
              ))
            )}
          </div>
        </>
      )}
    </div>
  );
};
