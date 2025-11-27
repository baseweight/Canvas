import React, { useState } from 'react';
import type { Model, ChatMessage } from '../types';
import './ChatPanel.css';

interface ChatPanelProps {
  currentModel?: Model;
  messages: ChatMessage[];
  onChangeModel: () => void;
  onSendMessage: (message: string) => void;
  isAudioCapable?: boolean;
}

export const ChatPanel: React.FC<ChatPanelProps> = ({
  currentModel,
  messages,
  onChangeModel,
  onSendMessage,
  isAudioCapable = false,
}) => {
  const [inputValue, setInputValue] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputValue.trim() && currentModel) {
      onSendMessage(inputValue);
      setInputValue('');
    }
  };

  return (
    <div className="bw-chat-panel">
      <div className="bw-chat-header">
        <div className="bw-chat-info">
          <div className="bw-chat-info-row">
            <label className="bw-chat-label">Model</label>
            <button className="bw-chat-change-button" onClick={onChangeModel}>
              Change
            </button>
          </div>
          {currentModel ? (
            <div className="bw-chat-model-info">
              <span className="bw-chat-model-name">{currentModel.displayName}</span>
              <span className="bw-chat-model-meta">{currentModel.backend}</span>
            </div>
          ) : (
            <div className="bw-chat-model-empty">No model loaded</div>
          )}
        </div>

        <div className="bw-chat-info">
          <label className="bw-chat-label">Task</label>
          {currentModel ? (
            <div className="bw-chat-task-info">
              <span className={`bw-chat-task-badge ${isAudioCapable ? 'bw-chat-task-badge--audio' : 'bw-chat-task-badge--vision'}`}>
                {isAudioCapable ? 'Audio Capable Language Model' : 'General Vision-Language Model'}
              </span>
            </div>
          ) : (
            <div className="bw-chat-model-empty">-</div>
          )}
        </div>
      </div>

      <div className="bw-chat-messages">
        {messages.length === 0 ? (
          <div className="bw-chat-empty">
            {currentModel ? (
              <>
                <p>No messages yet</p>
                <p className="bw-chat-empty-hint">
                  {isAudioCapable ? 'Audio Capable Language Model' : 'General Vision-Language Model'}
                </p>
              </>
            ) : (
              <>
                <p>No model loaded</p>
                <p className="bw-chat-empty-hint">
                  Load a model to begin
                </p>
              </>
            )}
          </div>
        ) : (
          messages.map(message => (
            <div
              key={message.id}
              className={`bw-chat-message bw-chat-message--${message.role}`}
            >
              <div className="bw-chat-message-role">
                {message.role === 'user' ? 'You' : currentModel?.displayName || 'Assistant'}
              </div>
              <div className="bw-chat-message-content">
                {message.content}
              </div>
            </div>
          ))
        )}
      </div>

      <form className="bw-chat-input-container" onSubmit={handleSubmit}>
        <input
          type="text"
          className="bw-chat-input"
          placeholder={currentModel ? "Ask about the image..." : "Load a model first..."}
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          disabled={!currentModel}
        />
        <button
          type="submit"
          className="bw-chat-send"
          disabled={!inputValue.trim() || !currentModel}
        >
          Send
        </button>
      </form>
    </div>
  );
};
