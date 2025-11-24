# Image Editing and Session Management Design

## Overview

This document outlines the behavior and implementation strategy for image editing features and session management in Baseweight Canvas. The core principle is: **loading new images resets the conversation, but editing the current image preserves context**.

---

## Current State

### What Works
- ✅ Load initial image via File→Open menu (Linux/Windows)
- ✅ Chat with image using VLM model
- ✅ Multi-turn conversation about the same image
- ✅ Session resets when new image is loaded via File→Open

### Current Problems
- ❌ **No obvious way to load a new image after the first one**
  - Users must use File→Open menu (not discoverable)
  - No visual "Load Image" button in the UI
  - Mac users don't have File menu override yet
- ❌ **No in-app image editing capabilities**
  - Users cannot markup/annotate images
  - Users cannot crop images
  - Must use external tools and reload
- ❌ **Unclear session management**
  - No visual indication of when session resets
  - No explicit "New Session" action

---

## Session Management Rules

### When Session MUST Reset (Clear KV Cache)

| Action | Behavior | Reason |
|--------|----------|--------|
| Load new image from filesystem | **RESET** | Different image = different context |
| File→Open menu selection | **RESET** | User explicitly loading new file |
| "New Image" button click | **RESET** | Explicit new session action |
| Model change | **RESET** | Different model = incompatible cache |
| Manual "Clear Session" | **RESET** | User-requested reset |

**Implementation:**
```rust
// Clear KV cache
llama_memory_clear(mem, true);
self.n_past = 0;
self.cached_image_hash = None; // or Some(new_hash)
// Clear frontend message history
setMessages([]);
```

### When Session MUST Preserve (Keep KV Cache)

| Action | Behavior | Reason |
|--------|----------|--------|
| Brush/markup annotation | **PRESERVE** | Same image with visual edits |
| Crop with rectangular marquee | **PRESERVE** | Derivative of current image |
| Image adjustments (future: brightness, contrast, etc.) | **PRESERVE** | Transformations of current image |
| Continued chat | **PRESERVE** | Multi-turn conversation |

**Implementation:**
```rust
// Append new image + text to KV cache
// DON'T clear KV cache
// DON'T reset n_past
// DO update cached_image_hash to new edited version
// Call mtmd_tokenize with new bitmap from self.n_past position
```

---

## Feature Specifications

### 1. Load New Image (High Priority)

**User Story:** As a user, I want an obvious way to load a new image into the canvas so I can start a new conversation.

**UI Components:**
- **"Load Image" button** - Prominent button in toolbar/header
- **Drag & drop support** - Drop image files onto canvas
- **File→Open menu** - Existing functionality (keep as secondary option)

**Behavior:**
1. User clicks "Load Image" button or drops file
2. **Confirmation dialog** if active conversation exists:
   ```
   Loading a new image will clear the current conversation.
   Continue?
   [Cancel] [Load New Image]
   ```
3. On confirm:
   - Clear chat message history (frontend)
   - Clear KV cache (backend)
   - Load new image into canvas
   - Reset session state

**Technical Implementation:**
- Frontend: Add `LoadImageButton.tsx` component
- Frontend: Add drag-drop event handlers to canvas area
- Backend: Existing `generate_with_image()` already handles image changes
- State management: `setMessages([])` before loading

**File Locations:**
- Component: `src/components/LoadImageButton.tsx`
- Handler: `src/App.tsx` - `handleLoadNewImage()`
- Backend: No changes needed (current hash-based detection works)

---

### 2. Brush/Markup Tool (High Priority)

**User Story:** As a user, I want to draw/annotate on the image so I can highlight areas of interest and ask questions about specific regions.

**UI Components:**
- **"Brush" tool button** - Toggle brush mode on/off
- **Color picker** - Select brush color
- **Brush size slider** - Adjust brush thickness (e.g., 2-20px)
- **Clear annotations button** - Remove all markup without resetting session
- **Canvas overlay** - Transparent layer for drawing

**Behavior:**
1. User clicks "Brush" tool → enters brush mode
2. User selects color and size
3. User draws on canvas with mouse/touch
4. Annotations are rendered on transparent overlay above image
5. When user sends next chat message:
   - Composite base image + annotations into single image
   - Send combined image to backend with `preserve_context = true`
   - **Session preserved** - model sees edited image in context
   - User can ask: "What's in the red circle I drew?"

**Session Management:**
- ✅ **Preserve conversation context**
- ✅ Update `cached_image_hash` to include annotations
- ✅ Append new image state to KV cache
- ❌ Do NOT clear message history

**Technical Implementation:**

**Frontend:**
```typescript
// Canvas structure
<div className="canvas-container">
  <canvas ref={baseImageCanvas} />      {/* Original image */}
  <canvas ref={annotationCanvas} />     {/* Transparent overlay */}
</div>

// Compositing function
function getCompositeImage(): Uint8Array {
  const composite = document.createElement('canvas');
  const ctx = composite.getContext('2d');
  ctx.drawImage(baseImageCanvas, 0, 0);
  ctx.drawImage(annotationCanvas, 0, 0);
  return getImageData(composite);
}
```

**Backend:**
```rust
// Add preserve_context parameter
pub fn generate_with_image(
    &mut self,
    prompt: &str,
    image_rgb: &[u8],
    image_width: u32,
    image_height: u32,
    preserve_context: bool,  // NEW
) -> Result<String> {
    let current_image_hash = compute_image_hash(image_rgb, image_width, image_height);
    let image_changed = match self.cached_image_hash {
        Some(cached_hash) => cached_hash != current_image_hash,
        None => true,
    };

    if image_changed && !preserve_context {
        // Clear cache for NEW image
        llama_memory_clear(mem, true);
        self.n_past = 0;
    } else if image_changed && preserve_context {
        // APPEND edited image to cache
        // Keep self.n_past intact
        // Process new image from current position
    }

    self.cached_image_hash = Some(current_image_hash);
    // ... rest of processing
}
```

**File Locations:**
- Component: `src/components/BrushTool.tsx`
- Tool state: `src/App.tsx` - `useState<BrushState>`
- Canvas drawing: `src/hooks/useCanvasDrawing.ts`
- Backend: `src-tauri/src/inference_engine.rs:385` - modify `generate_with_image()`
- Tauri command: `src-tauri/src/lib.rs:107` - add `preserve_context` parameter

**Brush State Interface:**
```typescript
interface BrushState {
  enabled: boolean;
  color: string;        // hex color
  size: number;         // 2-20
  opacity: number;      // 0.0-1.0
}
```

---

### 3. Rectangular Marquee Crop Tool (Medium Priority)

**User Story:** As a user, I want to crop the image to a specific region so I can focus the conversation on a particular area.

**UI Components:**
- **"Crop" tool button** - Toggle crop mode
- **Rectangular selection overlay** - Click and drag to define crop region
- **Apply/Cancel buttons** - Confirm or abandon crop
- **Aspect ratio lock** (optional) - Maintain proportions

**Behavior:**
1. User clicks "Crop" tool → enters crop mode
2. User clicks and drags to draw rectangle
3. Rectangle is resizable/draggable before confirming
4. User clicks "Apply Crop"
5. Canvas shows cropped region
6. When user sends next chat message:
   - Send cropped image to backend with `preserve_context = true`
   - **Session preserved** - model sees cropped image as evolution
   - User can ask: "What do you see in this zoomed area?"

**Session Management:**
- ✅ **Preserve conversation context**
- ✅ Update `cached_image_hash` to cropped version
- ✅ Append cropped image to KV cache
- ❌ Do NOT clear message history
- ✅ Allow "undo crop" to restore original (keep in history)

**Technical Implementation:**

**Frontend:**
```typescript
interface CropRegion {
  x: number;
  y: number;
  width: number;
  height: number;
}

function applyCrop(region: CropRegion) {
  const croppedCanvas = document.createElement('canvas');
  croppedCanvas.width = region.width;
  croppedCanvas.height = region.height;

  const ctx = croppedCanvas.getContext('2d');
  ctx.drawImage(
    baseImageCanvas,
    region.x, region.y, region.width, region.height,  // source
    0, 0, region.width, region.height                  // dest
  );

  // Replace base image with cropped version
  setCurrentImage(croppedCanvas.toDataURL());

  // Next message will use preserve_context=true
}
```

**Backend:**
- Same `preserve_context = true` logic as brush tool
- No special handling needed (just a different image)

**File Locations:**
- Component: `src/components/CropTool.tsx`
- Selection UI: `src/components/RectangleSelector.tsx`
- Tool state: `src/App.tsx` - `useState<CropState>`
- Canvas manipulation: `src/utils/imageProcessing.ts`

---

### 4. Visual Session Indicators (Low Priority, UX Enhancement)

**User Story:** As a user, I want to know when my session has been reset so I understand the model's context.

**UI Components:**
- **Session badge** - Shows "Session 1", "Session 2", etc.
- **Visual divider** in chat history - Horizontal line with "New Session" label
- **Toast notification** - "Session reset - conversation cleared"

**Behavior:**
- Session counter increments on reset
- Chat history shows divider when session resets
- Brief notification appears on reset actions

**File Locations:**
- Component: `src/components/SessionIndicator.tsx`
- State: `src/App.tsx` - `useState<sessionId>`

---

## Implementation Priority

### Phase 1: Foundation (Week 1)
1. ✅ Backend: Add `preserve_context` parameter to `generate_with_image()`
2. ✅ Backend: Implement context preservation logic
3. ✅ Frontend: Add "Load Image" button
4. ✅ Frontend: Add drag-drop support
5. ✅ Frontend: Add confirmation dialog for new image

### Phase 2: Brush Tool (Week 2)
1. ✅ Create canvas overlay architecture
2. ✅ Implement brush drawing
3. ✅ Add color/size controls
4. ✅ Implement image compositing
5. ✅ Test context preservation with annotations

### Phase 3: Crop Tool (Week 3)
1. ✅ Implement rectangle selection UI
2. ✅ Add crop apply/cancel logic
3. ✅ Test context preservation with crops
4. ✅ Add undo crop functionality

### Phase 4: Polish (Week 4)
1. ✅ Add session indicators
2. ✅ Improve visual feedback
3. ✅ Add keyboard shortcuts
4. ✅ Write user documentation

---

## Context Window Considerations

### Problem: Image Tokens Are Expensive

Vision models consume significant context:
- **Typical image**: 256-1024 tokens (model dependent)
- **Context limit**: 4096 tokens (current setting in `inference_engine.rs:329`)
- **With 3 images + conversation**: Could exceed limit

### Solution: Context Management Strategy

**Option A: Sliding Window (Recommended)**
- Keep most recent 2-3 images in context
- Drop oldest image when limit approached
- Inform user: "Early conversation history removed to fit new image"

**Option B: Image Replacement**
- Each edit replaces previous image in cache
- Only 1 image in context at a time
- Loses ability to reference "before" state

**Option C: Increase Context Window**
- Bump `n_ctx` to 8192 or higher
- Requires more VRAM
- Add as user setting

**Recommended**: Start with **Option B** (simplest), offer **Option C** as setting, implement **Option A** if users request it.

---

## Edge Cases and Error Handling

### 1. Image Too Large After Crop/Edit
**Problem:** Edited image exceeds model's max resolution
**Solution:** Auto-resize to model's max dimensions, show warning

### 2. Context Window Full
**Problem:** Cannot fit new image + conversation history
**Solution:** Offer choice:
```
Context limit reached. Choose an action:
[ ] Clear conversation and continue
[ ] Remove oldest images
[Cancel]
```

### 3. Brush Annotations Too Complex
**Problem:** Heavy annotation increases image size
**Solution:** Compress annotations to PNG before compositing

### 4. Crop to Tiny Region
**Problem:** Cropped image too small for model
**Solution:** Minimum crop size of 64x64, show validation error

---

## API Changes Summary

### Backend API Changes

**Modified Function:**
```rust
// src-tauri/src/inference_engine.rs
pub fn generate_with_image(
    &mut self,
    prompt: &str,
    image_rgb: &[u8],
    image_width: u32,
    image_height: u32,
    preserve_context: bool,  // NEW PARAMETER
) -> Result<String>
```

**Modified Tauri Command:**
```rust
// src-tauri/src/lib.rs
#[tauri::command]
async fn generate_response(
    prompt: String,
    image_data: Vec<u8>,
    image_width: u32,
    image_height: u32,
    preserve_context: bool,  // NEW PARAMETER
    engine: State<'_, SharedInferenceEngine>,
) -> Result<String, String>
```

### Frontend API Changes

**New Message Handler:**
```typescript
// src/App.tsx
const handleSendMessage = async (message: string) => {
  const preserveContext = !isNewImageLoad; // Preserve for edits, clear for new loads

  const response = await invoke('generate_response', {
    prompt: message,
    imageData: rgbBytes,
    imageWidth: width,
    imageHeight: height,
    preserveContext,  // NEW PARAMETER
  });
};
```

---

## Testing Plan

### Unit Tests
- ✅ Image hash computation
- ✅ Context preservation logic
- ✅ Canvas compositing
- ✅ Crop region calculation

### Integration Tests
- ✅ Load new image → verify session reset
- ✅ Brush annotation → verify session preserved
- ✅ Crop → verify session preserved
- ✅ Multiple edits → verify cumulative context

### User Acceptance Tests
1. Load image → chat → load new image → verify old chat cleared
2. Load image → chat → annotate → chat → verify model sees annotations
3. Load image → chat → crop → chat → verify model sees cropped region
4. Load image → annotate → crop → chat → verify both edits visible

---

## Open Questions

1. **Should we support undo/redo for edits?**
   - Undo would need to roll back KV cache state
   - Complex to implement with llama.cpp
   - **Recommendation:** Phase 2 feature, not MVP

2. **Should brush strokes be added incrementally or on commit?**
   - Incremental: Each stroke = new image in context (expensive)
   - On commit: Wait for user to finish drawing (recommended)
   - **Recommendation:** Buffer strokes, send on next chat message

3. **Should we show image thumbnails in chat history?**
   - Useful for multi-image conversations
   - Increases UI complexity
   - **Recommendation:** Phase 2 feature

4. **Mac menu override timeline?**
   - Needed for File→Open to work on Mac
   - Blocks Mac release
   - **Recommendation:** High priority if targeting Mac users

---

## Success Metrics

### User Experience
- ✅ Users can load new images in ≤2 clicks
- ✅ 80%+ of users discover image loading without documentation
- ✅ Users successfully annotate and ask follow-up questions

### Technical Performance
- ✅ Context preservation works without KV cache corruption
- ✅ Image edits maintain conversation history
- ✅ Context window management prevents overflow errors
- ✅ No memory leaks with repeated edits

### Quality
- ✅ Model correctly references annotated regions
- ✅ Model correctly interprets cropped areas
- ✅ Session resets only when expected
- ✅ Zero data loss during normal operation

---

## Related Documentation

- llama.cpp multimodal support: https://github.com/ggml-org/llama.cpp/pull/12898
- Context management in VLMs: (TBD)
- Canvas API reference: https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API

---

## Revision History

| Date | Version | Author | Changes |
|------|---------|--------|---------|
| 2025-11-23 | 1.0 | Design Team | Initial specification |

