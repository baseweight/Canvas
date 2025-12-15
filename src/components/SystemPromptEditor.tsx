import { useState } from 'react';

interface SystemPromptEditorProps {
  modelId?: string;
  systemPrompt: string;
  onSystemPromptChange: (prompt: string) => void;
  supportsSystemPrompt?: boolean;
}

export function SystemPromptEditor({
  modelId,
  systemPrompt,
  onSystemPromptChange,
  supportsSystemPrompt = false,
}: SystemPromptEditorProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  if (!modelId) {
    return null;
  }

  return (
    <div className="system-prompt-editor">
      <button
        className="system-prompt-toggle"
        onClick={() => setIsExpanded(!isExpanded)}
        type="button"
      >
        <span className="system-prompt-icon">⚙️</span>
        <span className="system-prompt-label">
          System Prompt
          {!supportsSystemPrompt && <span className="system-prompt-warning">⚠️</span>}
        </span>
        <span className="system-prompt-chevron">{isExpanded ? '▼' : '▶'}</span>
      </button>

      {isExpanded && (
        <div className="system-prompt-content">
          {!supportsSystemPrompt && (
            <div className="system-prompt-notice">
              ⚠️ This model has limited system prompt support. Results may vary.
            </div>
          )}
          <textarea
            className="system-prompt-textarea"
            value={systemPrompt}
            onChange={(e) => onSystemPromptChange(e.target.value)}
            placeholder="Enter a system prompt to guide the model's behavior (optional)..."
            rows={4}
          />
          <div className="system-prompt-footer">
            <span className="system-prompt-chars">{systemPrompt.length} characters</span>
            <button
              className="system-prompt-clear"
              onClick={() => onSystemPromptChange('')}
              disabled={!systemPrompt}
              type="button"
            >
              Clear
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
