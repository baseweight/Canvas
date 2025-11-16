import React, { useState } from 'react';
import './AddModelDialog.css';

interface AddModelDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onAdd: (repo: string, quantization: string) => void;
}

const QUANTIZATION_OPTIONS = [
  { value: 'Q4_K_M', label: 'Q4_K_M (Recommended - Balanced)', description: 'Good quality, reasonable size' },
  { value: 'Q4_K_S', label: 'Q4_K_S (Small)', description: 'Smaller size, slightly lower quality' },
  { value: 'Q5_K_M', label: 'Q5_K_M (High Quality)', description: 'Better quality, larger size' },
  { value: 'Q5_K_S', label: 'Q5_K_S (Medium-High)', description: 'Good balance of quality and size' },
  { value: 'Q6_K', label: 'Q6_K (Very High Quality)', description: 'Highest quality, largest size' },
  { value: 'Q8_0', label: 'Q8_0 (Near-Original)', description: 'Almost original quality, very large' },
];

export const AddModelDialog: React.FC<AddModelDialogProps> = ({
  isOpen,
  onClose,
  onAdd,
}) => {
  const [repo, setRepo] = useState('');
  const [quantization, setQuantization] = useState('Q4_K_M');
  const [error, setError] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    // Basic validation
    if (!repo.trim()) {
      setError('Please enter a HuggingFace repository');
      return;
    }

    // Check if it looks like a HF repo (owner/repo format)
    const repoPattern = /^[\w-]+\/[\w.-]+$/;
    if (!repoPattern.test(repo.trim())) {
      setError('Repository must be in format: username/model-name');
      return;
    }

    setError('');
    onAdd(repo.trim(), quantization);

    // Reset form
    setRepo('');
    setQuantization('Q4_K_M');
  };

  const handleClose = () => {
    setRepo('');
    setQuantization('Q4_K_M');
    setError('');
    onClose();
  };

  if (!isOpen) return null;

  return (
    <>
      <div className="bw-dialog-backdrop" onClick={handleClose} />
      <div className="bw-dialog">
        <div className="bw-dialog-header">
          <h2 className="bw-dialog-title">Add Model from HuggingFace</h2>
          <button className="bw-dialog-close" onClick={handleClose} aria-label="Close">
            <svg viewBox="0 0 24 24" width="20" height="20">
              <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z" fill="currentColor"/>
            </svg>
          </button>
        </div>

        <form className="bw-dialog-content" onSubmit={handleSubmit}>
          <div className="bw-dialog-info">
            <h3 className="bw-dialog-info-title">Requirements for Vision-Language Models</h3>
            <p className="bw-dialog-info-text">
              The HuggingFace repository must contain both:
            </p>
            <ul className="bw-dialog-requirements">
              <li>
                <strong>mmproj GGUF</strong> - Multimodal projector file
                <span className="bw-dialog-hint">(e.g., mmproj-model-f16.gguf)</span>
              </li>
              <li>
                <strong>Language model GGUF</strong> - The base language model
                <span className="bw-dialog-hint">(e.g., model-q4_k_m.gguf)</span>
              </li>
            </ul>
          </div>

          <div className="bw-dialog-field">
            <label htmlFor="repo" className="bw-dialog-label">
              HuggingFace Repository
            </label>
            <input
              id="repo"
              type="text"
              className="bw-dialog-input"
              placeholder="username/model-name"
              value={repo}
              onChange={(e) => {
                setRepo(e.target.value);
                setError('');
              }}
              autoFocus
            />
            <p className="bw-dialog-field-hint">
              Example: liuhaotian/llava-v1.6-34b-gguf
            </p>
            {error && <p className="bw-dialog-error">{error}</p>}
          </div>

          <div className="bw-dialog-field">
            <label htmlFor="quantization" className="bw-dialog-label">
              Quantization Level
            </label>
            <select
              id="quantization"
              className="bw-dialog-select"
              value={quantization}
              onChange={(e) => setQuantization(e.target.value)}
            >
              {QUANTIZATION_OPTIONS.map(opt => (
                <option key={opt.value} value={opt.value}>
                  {opt.label} - {opt.description}
                </option>
              ))}
            </select>
            <p className="bw-dialog-field-hint">
              Lower quantization = smaller file size but slightly reduced quality
            </p>
          </div>

          <div className="bw-dialog-footer">
            <button
              type="button"
              className="bw-button-secondary"
              onClick={handleClose}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="bw-button-primary"
            >
              Download Model
            </button>
          </div>
        </form>
      </div>
    </>
  );
};
