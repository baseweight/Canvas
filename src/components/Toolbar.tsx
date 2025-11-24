import React from 'react';
import './Toolbar.css';

export type ToolType = 'load' | 'select' | 'crop' | 'brush';

interface ToolbarProps {
  activeTool?: ToolType;
  onToolSelect: (tool: ToolType) => void;
}

interface ToolButtonProps {
  tool: ToolType;
  label: string;
  icon: string;
  active: boolean;
  disabled?: boolean;
  onClick: () => void;
}

const ToolButton: React.FC<ToolButtonProps> = ({ label, icon, active, disabled, onClick }) => {
  return (
    <button
      className={`bw-tool-button ${active ? 'bw-tool-button--active' : ''} ${disabled ? 'bw-tool-button--disabled' : ''}`}
      onClick={onClick}
      disabled={disabled}
      title={disabled ? `${label} (Coming soon)` : label}
    >
      <span className="bw-tool-icon">{icon}</span>
    </button>
  );
};

export const Toolbar: React.FC<ToolbarProps> = ({ activeTool, onToolSelect }) => {
  const tools: Array<{ tool: ToolType; label: string; icon: string; disabled?: boolean }> = [
    { tool: 'load', label: 'Load', icon: '📁' },
    { tool: 'select', label: 'Select', icon: '⬚', disabled: true },
    { tool: 'crop', label: 'Crop', icon: '✂', disabled: true },
    { tool: 'brush', label: 'Brush', icon: '🖌', disabled: true },
  ];

  return (
    <div className="bw-toolbar">
      <div className="bw-toolbar-inner">
        {tools.map(({ tool, label, icon, disabled }) => (
          <ToolButton
            key={tool}
            tool={tool}
            label={label}
            icon={icon}
            active={activeTool === tool}
            disabled={disabled}
            onClick={() => onToolSelect(tool)}
          />
        ))}
      </div>
    </div>
  );
};
