# Crop Tool Implementation

## Date: 2025-11-24
## Status: Crop ✅ Implemented | Brush ❌ Removed

---

## Overview

This document summarizes the implementation of image editing tools for Baseweight Canvas, completed in preparation for the December 2nd public demo.

**TL;DR:** Crop tool works great. Brush tool was attempted but removed due to persistent canvas resizing bugs. Focus on reliable crop-only workflow for demo.

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

### 2. Brush Drawing Tool ❌ REMOVED

**Status: Attempted but removed before demo due to bugs**

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

## Brush Tool: Why It Was Removed

**Decision:** Removed brush tool before demo (2025-11-24)

**Reason:** Persistent canvas resizing bugs made it unreliable:
- Canvas wouldn't resize properly when loading new images
- Drawing became disabled after image reload
- Multiple competing `useEffect` hooks caused race conditions
- Attempted refactor using single effect still exhibited bugs
- Time pressure before Dec 2nd demo meant focusing on reliable features only

**Files Removed:**
- `src/components/DrawingOverlay.tsx` (189 lines)
- `src/components/DrawingOverlay.css` (15 lines)

**Code Still Documented Here For Future Reference**

---

## Toolbar Integration

**Final Toolbar (Brush Removed):**
1. Load (📁) - Opens file dialog
2. Select (⬚) - Activates rectangular marquee
3. Crop (✂) - Applies crop (disabled until selection exists)

**Tool State Management:**
- Only one tool active at a time
- Clear visual feedback for active tool
- Disabled state shown with tooltip explanation

---

## Known Issues & Limitations

### Crop Tool Issues

**1. No Undo/Redo**
- Crops are permanent once applied
- No way to revert crop
- Workaround: Reload original image (resets session)

**2. Blob URL Memory Leaks**
- Old blob URLs from crops are not revoked
- Could cause memory issues with many crops
- Should call `URL.revokeObjectURL()` on old URLs

**3. No Crop Refinement**
- Can't adjust selection after starting
- Must clear and redraw if wrong
- No resize handles on selection box

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
- ✅ Crop functionality works reliably
- ✅ UI is functional and discoverable
- ✅ Image updates correctly after crop
- ✅ Chat history preserved visually
- ✅ Selection tool intuitive and responsive

### ⚠️ Needs Disclaimer
- "Chat history is for reference only - model doesn't retain context yet"
- "Crop cannot be undone - reload image to start over"

### ❌ Not Demo Ready (Future Work)
- Brush tool (attempted but buggy - see above)
- Undo/redo functionality
- Memory leak fixes for blob URLs
- Proper multi-turn conversation context
- Selection refinement (resize handles)

---

## Files Created

**New Files (Final):**
- `src/components/SelectionOverlay.tsx` (147 lines) ✅
- `src/components/SelectionOverlay.css` (35 lines) ✅
- `src/types/selection.ts` (23 lines) ✅

**Files Created Then Removed:**
- `src/components/DrawingOverlay.tsx` (189 lines) ❌ DELETED
- `src/components/DrawingOverlay.css` (15 lines) ❌ DELETED

**Modified Files (Final):**
- `src/components/ImageViewer.tsx` - Added crop integration
- `src/components/Toolbar.tsx` - Added Load/Select/Crop tools
- `src/App.tsx` - Added crop callback

**Total LOC (Final):** ~200 lines

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
