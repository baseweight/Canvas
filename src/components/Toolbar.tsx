import React from 'react';
import './Toolbar.css';

export type ToolType = 'load' | 'select' | 'crop' | 'brush';

interface ToolbarProps {
  activeTool?: ToolType;
  hasSelection?: boolean;
  onToolSelect: (tool: ToolType) => void;
  // SAM3 props
  sam3Enabled?: boolean;
  sam3Loading?: boolean;
  sam3ModelDownloaded?: boolean;
  onSam3Toggle?: () => void;
  onSam3Download?: () => void;
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

export const Toolbar: React.FC<ToolbarProps> = ({
  activeTool,
  hasSelection,
  onToolSelect,
  sam3Enabled,
  sam3Loading,
  sam3ModelDownloaded,
  onSam3Toggle,
  onSam3Download,
}) => {
  const tools: Array<{ tool: ToolType; label: string; icon: string; disabled?: boolean; disabledReason?: string }> = [
    { tool: 'load', label: 'Load', icon: '📁' },
    { tool: 'select', label: 'Select', icon: '⬚' },
    { tool: 'crop', label: 'Crop', icon: '✂', disabled: !hasSelection, disabledReason: !hasSelection ? 'Select an area first' : undefined },
  ];

  const handleSam3Click = () => {
    if (!sam3ModelDownloaded && onSam3Download) {
      onSam3Download();
    } else if (onSam3Toggle) {
      onSam3Toggle();
    }
  };

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
        {/* SAM3 Toggle Button */}
        <div className="bw-toolbar-divider" />
        <button
          className={`bw-tool-button bw-sam3-toggle ${sam3Enabled ? 'bw-tool-button--active' : ''} ${sam3Loading ? 'bw-tool-button--loading' : ''}`}
          onClick={handleSam3Click}
          disabled={sam3Loading}
          title={
            sam3Loading ? 'Loading SAM3...' :
            !sam3ModelDownloaded ? 'Download SAM3 Model' :
            sam3Enabled ? 'Disable Segment Anything' :
            'Enable Segment Anything'
          }
        >
          <span className="bw-tool-icon">
            {sam3Loading ? '⏳' : '✨'}
          </span>
          {!sam3ModelDownloaded && !sam3Loading && (
            <span className="bw-sam3-download-badge">↓</span>
          )}
        </button>
      </div>
    </div>
  );
};
