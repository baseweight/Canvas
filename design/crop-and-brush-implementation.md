# Crop and Brush Tool Implementation

## Date: 2025-11-24
## Status: Implemented (with known issues)

---

## Overview

This document summarizes the implementation of the crop and brush tools for Baseweight Canvas, completed in preparation for the December 2nd public demo.

---

## What Was Implemented

### 1. Rectangular Marquee Crop Tool ✅

**Components Created:**
- `src/components/SelectionOverlay.tsx` - Handles rectangular selection with mouse drag
- `src/components/SelectionOverlay.css` - Visual styling for selection rectangle
- `src/types/selection.ts` - TypeScript interfaces for selection data

**Features:**
- Click and drag to create rectangular selection
- Visual overlay showing selected area with dashed border
- Crop button only enabled when selection exists
- Press Escape to clear selection
- Automatic coordinate normalization (handles dragging in any direction)
- Minimum selection size (5x5 pixels) to prevent accidental tiny crops

**Implementation Details:**
- Selection coordinates are tracked relative to displayed image size
- When cropping, coordinates are scaled to match natural image dimensions
- Uses HTML5 Canvas API to extract the cropped region
- Creates blob URL for the cropped image
- Updates the current media state with cropped image
- Preserves chat history (doesn't reset conversation)

**Files Modified:**
- `src/components/ImageViewer.tsx` - Integrated SelectionOverlay and crop handler
- `src/components/Toolbar.tsx` - Added conditional enabling of crop button
- `src/App.tsx` - Added handleImageCrop callback

**Key Code:**
```typescript
const handleCrop = () => {
  // Scale selection from display size to natural size
  const scaleX = img.naturalWidth / img.width;
  const scaleY = img.naturalHeight / img.height;

  // Extract cropped region using canvas
  ctx.drawImage(
    img,
    selection.x * scaleX, selection.y * scaleY,
    selection.width * scaleX, selection.height * scaleY,
    0, 0, canvas.width, canvas.height
  );

  // Create blob URL and update state
  canvas.toBlob((blob) => {
    const croppedUrl = URL.createObjectURL(blob);
    onImageCrop(croppedUrl);
  }, 'image/png');
};
```

---

### 2. Brush Drawing Tool ✅

**Components Created:**
- `src/components/DrawingOverlay.tsx` - Handles freehand drawing with mouse
- `src/components/DrawingOverlay.css` - Styling for drawing canvas
- Exposed `DrawingOverlayRef` interface for imperative control

**Features:**
- Freehand drawing with mouse on transparent overlay
- Red brush color (hardcoded for now)
- 5px brush size (hardcoded for now)
- Smooth line drawing between mouse positions
- Press Escape to clear current drawing
- Automatic application when switching away from brush tool
- Automatic application before sending message to VLM

**Implementation Details:**
- Uses separate transparent canvas overlay for drawing
- Drawing is composited with base image when applied
- Uses `React.forwardRef` and `useImperativeHandle` for parent control
- Tracks whether any drawing exists via `hasAnyDrawing` state
- Canvas dimensions match displayed image size
- Drawing is scaled when compositing to match natural image dimensions

**Files Modified:**
- `src/components/ImageViewer.tsx` - Integrated DrawingOverlay, added ref management
- `src/components/Toolbar.tsx` - Enabled brush button
- `src/App.tsx` - Added ref to ImageViewer, auto-apply before VLM inference

**Key Code:**
```typescript
const handleApplyDrawing = (drawingCanvas: HTMLCanvasElement) => {
  // Create composite canvas
  const compositeCanvas = document.createElement('canvas');
  compositeCanvas.width = img.naturalWidth;
  compositeCanvas.height = img.naturalHeight;

  // Draw original image
  ctx.drawImage(img, 0, 0, img.naturalWidth, img.naturalHeight);

  // Draw overlay scaled to natural size
  ctx.drawImage(
    drawingCanvas,
    0, 0, drawingCanvas.width, drawingCanvas.height,
    0, 0, img.naturalWidth, img.naturalHeight
  );

  // Create blob and update state
  compositeCanvas.toBlob((blob) => {
    const compositeUrl = URL.createObjectURL(blob);
    onImageCrop(compositeUrl); // Reuses crop callback
  }, 'image/png');
};
```

**Auto-Apply Integration:**
```typescript
// Before sending message to VLM
imageViewerRef.current?.applyPendingEdits();
await new Promise(resolve => setTimeout(resolve, 100)); // Wait for state update
```

---

## Toolbar Integration

**Updated Toolbar:**
1. Load (📁) - Opens file dialog
2. Select (⬚) - Activates rectangular marquee
3. Crop (✂) - Applies crop (disabled until selection exists)
4. Brush (🖌) - Activates brush drawing mode

**Tool State Management:**
- Only one tool active at a time
- Switching tools auto-applies pending edits
- Clear visual feedback for active tool
- Disabled state shown with tooltip explanation

---

## Known Issues & Limitations

### ⚠️ CRITICAL: Image Merge Performance Issue

**Problem:**
The image compositing operation (merging drawing overlay with base image) is computationally expensive and blocks the UI thread. This causes:
- Noticeable delay when switching tools
- Chat pane freezes during merge
- User message appears delayed
- Poor UX when drawing and immediately sending message

**Why It Happens:**
- `canvas.toBlob()` is synchronous and blocking
- Large images (e.g., 5712x4284) take significant time to composite
- React state updates happen after blob creation completes
- 100ms timeout is insufficient for large images

**Impact:**
- User sees frozen UI for 500-2000ms on large images
- Chat messages feel laggy
- Drawing feels unresponsive

**Potential Solutions (Not Implemented):**
1. Use Web Workers for image compositing (offload to background thread)
2. Debounce the apply operation (only merge when idle for 500ms)
3. Show loading spinner during merge
4. Use `requestIdleCallback` to defer merge
5. Reduce image resolution before compositing (lossy but fast)

**Recommendation for Dec 2nd Demo:**
- Document as known limitation
- Advise users to use smaller images (<2MP)
- Consider as P0 fix for production release

---

### Other Known Issues

**1. No Undo/Redo**
- Edits are permanent once applied
- No way to revert crop or brush strokes
- Workaround: Reload original image (resets session)

**2. No Brush Controls**
- Color is hardcoded to red (#FF0000)
- Size is hardcoded to 5px
- No opacity control
- No eraser tool

**3. No Visual Feedback During Apply**
- User doesn't know if drawing is being applied
- No progress indicator
- Could confuse users on slow devices

**4. Drawing Only Visible in Brush Mode**
- Once you switch tools, can't see your drawing until applied
- Should show preview of pending drawing

**5. Blob URL Memory Leaks**
- Old blob URLs are not revoked
- Could cause memory issues with many edits
- Should call `URL.revokeObjectURL()` on old URLs

---

## Chat State Management

**Current Behavior:**
- Chat is completely stateless from VLM perspective
- Each message sends only the current image + current prompt
- No conversation history sent to backend
- Frontend displays message history for UX only

**What This Means:**
- Cropping/drawing preserves frontend chat UI
- But VLM has no memory of previous exchanges
- User might think model "remembers" but it doesn't
- Session preservation only affects frontend display

**Future Improvement:**
- Add proper conversation history to VLM context
- Send all previous messages + images in each request
- Requires backend changes to support multi-turn context

---

## Demo Readiness Assessment

### ✅ Ready for Demo
- Basic crop functionality works
- Basic brush functionality works
- UI is functional and discoverable
- Tools work together (can crop then draw, or draw then crop)
- Image updates correctly after edits
- Chat history preserved visually

### ⚠️ Needs Disclaimer
- "Image processing may take a moment on large images"
- "Chat history is for reference only - model doesn't retain context"
- "These tools are in alpha - expect rough edges"

### ❌ Not Demo Ready (Future Work)
- Performance optimization for large images
- Brush customization controls
- Undo/redo functionality
- Visual feedback during processing
- Memory leak fixes
- Proper multi-turn conversation context

---

## Files Created

**New Files:**
- `src/components/SelectionOverlay.tsx` (147 lines)
- `src/components/SelectionOverlay.css` (35 lines)
- `src/components/DrawingOverlay.tsx` (177 lines)
- `src/components/DrawingOverlay.css` (14 lines)
- `src/types/selection.ts` (23 lines)

**Modified Files:**
- `src/components/ImageViewer.tsx` - Added crop + brush integration
- `src/components/Toolbar.tsx` - Updated tool states
- `src/App.tsx` - Added refs and auto-apply logic

**Total LOC Added:** ~400 lines

---

## Architecture Decisions

### Why Separate Overlays?
- Selection and Drawing are different interaction modes
- Each needs its own canvas layer
- Separation of concerns makes code maintainable
- Allows independent enable/disable

### Why `forwardRef` for ImageViewer?
- App needs to trigger `applyPendingEdits()` before VLM inference
- Maintains uni-directional data flow
- Avoids prop drilling through multiple layers

### Why Blob URLs Instead of Data URLs?
- Data URLs encode entire image as base64 (33% larger)
- Blob URLs just reference memory (more efficient)
- Base64 encoding is slow for large images
- Blob URLs have browser size limits that are higher

### Why Composite on Apply Rather Than Continuously?
- Compositing is expensive
- User might draw multiple strokes
- Wait until they're done before merging
- Reduces CPU usage during drawing

---

## Testing Notes

**Tested Scenarios:**
- ✅ Crop a region → send message → VLM sees cropped image
- ✅ Draw on image → send message → VLM sees drawing
- ✅ Crop then draw → both edits applied
- ✅ Draw then crop → both edits applied
- ✅ Multiple crops in sequence → each replaces previous
- ✅ Escape key clears selection
- ✅ Escape key clears drawing
- ✅ Switching tools auto-applies drawing

**Not Tested:**
- ❌ Very large images (>10MP)
- ❌ Memory usage over extended editing session
- ❌ Touch input on tablets
- ❌ High-DPI displays
- ❌ Rapid tool switching
- ❌ Network interruption during blob creation

---

## Post-Demo TODO

### High Priority (Before Beta)
1. **Fix image merge performance**
   - Move compositing to Web Worker
   - Add loading indicator
   - Profile and optimize

2. **Add brush controls**
   - Color picker
   - Size slider
   - Opacity control

3. **Memory leak fixes**
   - Revoke old blob URLs
   - Clean up canvas references

### Medium Priority
4. **Add undo/redo**
   - Keep edit history stack
   - Allow reverting changes
   - Max 10 undo levels

5. **Visual feedback**
   - Show "Applying edits..." spinner
   - Preview pending drawing in other tools
   - Progress bar for large images

6. **Multi-turn conversation**
   - Send conversation history to backend
   - Proper VLM context management
   - Token budget tracking

### Low Priority
7. **Keyboard shortcuts**
   - Ctrl+Z for undo
   - Delete/Backspace to clear
   - Number keys for tool selection

8. **Crop improvements**
   - Aspect ratio lock
   - Crop presets (square, 16:9, etc.)
   - Resize handles on selection

---

## Related Issues

**GitHub Issues to Create:**
- Performance: Image merge blocks UI thread
- Feature: Add brush color and size controls
- Bug: Blob URLs not revoked causing memory leak
- Feature: Add undo/redo for edits
- Feature: Multi-turn conversation context
- UX: Show loading indicator during image processing

---

## Revision History

| Date | Author | Changes |
|------|--------|---------|
| 2025-11-24 | Claude | Initial implementation summary |
| TBD | TBD | Performance fixes |
| TBD | TBD | Brush controls added |
