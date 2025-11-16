import React from 'react';
import './Layout.css';

interface LayoutProps {
  children: React.ReactNode;
}

export const Layout: React.FC<LayoutProps> = ({ children }) => {
  return (
    <div className="bw-layout">
      <div className="bw-toolbar">
        <div className="bw-toolbar-left">
          <div className="bw-logo">
            <h1>Baseweight Canvas</h1>
          </div>
        </div>
        <div className="bw-toolbar-right">
          {/* Settings, help icons will go here */}
        </div>
      </div>

      <div className="bw-main">
        {children}
      </div>
    </div>
  );
};
