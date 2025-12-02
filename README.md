# Baseweight Canvas

Baseweight Canvas is a frontend to llama.cpp/mtmd that puts Vision and Audio modalities first, before chat. Built with Tauri and React, it provides a native desktop experience for multimodal AI interactions.

## Features

- **Vision-First Interface**: Prioritizes image and visual content processing
- **Audio Integration**: Built-in audio handling with waveform visualization using WaveSurfer.js
- **Native Performance**: Powered by Tauri for lightweight, fast desktop application
- **Modern UI**: Responsive interface built with React and Adobe Spectrum CSS
- **Cross-Platform**: Runs on Linux, macOS, and Windows

## Tech Stack

**Frontend:**
- React 19
- TypeScript
- Vite
- Adobe Spectrum CSS
- WaveSurfer.js

**Backend:**
- Tauri 2
- Rust
- Tokio (async runtime)
- Symphonia (audio decoding)

## Prerequisites

Before you begin, ensure you have the following installed:

- [Node.js](https://nodejs.org/) (v18 or higher)
- [npm](https://www.npmjs.com/) or [yarn](https://yarnpkg.com/)
- [Rust](https://www.rust-lang.org/tools/install) (latest stable)
- Platform-specific dependencies for Tauri:
  - **Linux**: `libwebkit2gtk-4.1-dev`, `build-essential`, `curl`, `wget`, `file`, `libssl-dev`, `libayatana-appindicator3-dev`, `librsvg2-dev`
  - **macOS**: Xcode Command Line Tools
  - **Windows**: Microsoft C++ Build Tools

## Installation

1. Clone the repository:
```bash
git clone https://github.com/baseweight/BaseweightCanvas.git
cd BaseweightCanvas
```

2. Install dependencies:
```bash
npm install
```

## Development

To run the application in development mode:

```bash
npm run tauri dev
```

This will start the Vite dev server and launch the Tauri application with hot-reload enabled.

## Building

To create a production build:

```bash
npm run build
npm run tauri build
```

The compiled application will be available in `src-tauri/target/release/`.

## Project Structure

```
BaseweightCanvas/
├── src/                    # React frontend source
│   ├── components/        # React components
│   ├── hooks/            # Custom React hooks
│   ├── styles/           # CSS styles
│   ├── types/            # TypeScript type definitions
│   ├── utils/            # Utility functions
│   └── App.tsx           # Main application component
├── src-tauri/            # Tauri/Rust backend
│   ├── src/              # Rust source code
│   ├── icons/            # Application icons
│   └── Cargo.toml        # Rust dependencies
├── public/               # Static assets
└── package.json          # Node.js dependencies

```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

