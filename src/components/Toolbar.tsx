import React from 'react';
import './Toolbar.css';

export type ToolType = 'load' | 'select' | 'crop' | 'brush';

interface ToolbarProps {
  activeTool?: ToolType;
  hasSelection?: boolean;
  onToolSelect: (tool: ToolType) => void;
}

interface ToolButtonProps {
  tool: ToolType;
  label: string;
  icon: string;
  active: boolean;
  disabled?: boolean;
  disabledReason?: string;
  onClick: () => void;
}

const ToolButton: React.FC<ToolButtonProps> = ({ label, icon, active, disabled, disabledReason, onClick }) => {
  return (
    <button
      className={`bw-tool-button ${active ? 'bw-tool-button--active' : ''} ${disabled ? 'bw-tool-button--disabled' : ''}`}
      onClick={onClick}
      disabled={disabled}
      title={disabled ? `${label} (${disabledReason || 'Coming soon'})` : label}
    >
      <span className="bw-tool-icon">{icon}</span>
    </button>
  );
};

export const Toolbar: React.FC<ToolbarProps> = ({ activeTool, hasSelection, onToolSelect }) => {
  const tools: Array<{ tool: ToolType; label: string; icon: string; disabled?: boolean; disabledReason?: string }> = [
    { tool: 'load', label: 'Load', icon: '📁' },
    { tool: 'select', label: 'Select', icon: '⬚' },
    { tool: 'crop', label: 'Crop', icon: '✂', disabled: !hasSelection, disabledReason: !hasSelection ? 'Select an area first' : undefined },
    { tool: 'brush', label: 'Brush', icon: '🖌' },
  ];

  return (
    <div className="bw-toolbar">
      <div className="bw-toolbar-inner">
        {tools.map(({ tool, label, icon, disabled, disabledReason }) => (
          <ToolButton
            key={tool}
            tool={tool}
            label={label}
            icon={icon}
            active={activeTool === tool}
            disabled={disabled}
            disabledReason={disabledReason}
            onClick={() => onToolSelect(tool)}
          />
        ))}
      </div>
    </div>
  );
};
