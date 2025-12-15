# Implementation Plan: System Prompts and Tool Use for Baseweight Canvas

## Overview

Add two major features to Baseweight Canvas:
1. **System Prompts**: Per-model system prompts that persist across sessions
2. **Tool Use**: Multi-format tool calling with configurable per-tool approval

## User Requirements

- **Tool Format**: Multi-format support (Functionary XML, Command-R JSON, LLaMA 3.1+)
- **Security**: Configurable per-tool approval (some auto-approved, others require confirmation)
- **UI Display**: Expandable sections in messages for tool execution details
- **Persistence**: Per-model system prompt persistence (saved to disk, different prompts per model)

## Architecture Decisions

### System Prompts
- System prompts stored in a configuration file per model
- Prepended to conversation array before sending to inference engine
- Included in KV cache (only reprocessed when changed)

### Tool Use
- Model metadata specifies tool call format (XML, JSON, etc.)
- Tool registry using Rust trait pattern for extensibility
- Tools execute in Rust backend (Tauri) for security and system access
- Tool results injected back into conversation for iterative generation
- Configurable approval system per tool

## Implementation Phases

### Phase 1: System Prompts Foundation

**Goal**: Enable per-model system prompts with disk persistence

#### Backend Changes

**File: `src-tauri/src/lib.rs`**
- Add `system_prompt: Option<String>` to `generate_response` command
- Add `system_prompt: Option<String>` to `generate_response_audio` command
- Add commands:
  - `save_system_prompt(model_id: String, prompt: String) -> Result<(), String>`
  - `load_system_prompt(model_id: String) -> Result<Option<String>, String>`
  - `clear_system_prompt(model_id: String) -> Result<(), String>`

**File: `src-tauri/src/inference_engine.rs`**
- Modify `generate_with_conversation()` to accept `system_prompt: Option<String>`
- When provided, prepend `ChatMessage { role: "system", content: system_prompt }` to conversation
- Track system prompt hash to detect changes and invalidate KV cache when needed

**New File: `src-tauri/src/system_prompts.rs`**
```rust
pub struct SystemPromptManager {
    config_dir: PathBuf,  // Store in app config directory
}

impl SystemPromptManager {
    pub fn load(&self, model_id: &str) -> Result<Option<String>, String>
    pub fn save(&self, model_id: &str, prompt: &str) -> Result<(), String>
    pub fn clear(&self, model_id: &str) -> Result<(), String>
}
```

#### Frontend Changes

**File: `src/App.tsx`**
- Add state: `const [systemPrompt, setSystemPrompt] = useState<string>("")`
- Load system prompt when model changes: `invoke('load_system_prompt', { modelId })`
- Pass `systemPrompt` to `generate_response` calls
- Add autosave: save system prompt when it changes

**New Component: `src/components/SystemPromptEditor.tsx`**
```tsx
interface SystemPromptEditorProps {
  modelId: string;
  systemPrompt: string;
  onSystemPromptChange: (prompt: string) => void;
}
```
- Collapsible section with textarea for editing
- Character counter
- Clear button
- Preset templates dropdown (optional enhancement)
- Auto-save indicator

**File: `src/components/ChatPanel.tsx`**
- Add SystemPromptEditor component in chat header area
- Display system message in conversation (distinct styling, collapsible)

**File: `src/types/index.ts`**
- No changes needed - `ChatMessage` already supports `role: 'system'`

### Phase 2: Tool Infrastructure

**Goal**: Build core tool system without implementing specific tools

#### Backend: Tool Registry

**New File: `src-tauri/src/tools/mod.rs`**
```rust
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,  // JSON Schema
}

pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

pub struct ToolResult {
    pub tool_call_id: String,
    pub success: bool,
    pub output: String,
    pub error: Option<String>,
}

pub enum ApprovalMode {
    Always,
    Never,
    Configurable { default: bool },
}

pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters(&self) -> serde_json::Value;
    fn requires_approval(&self) -> ApprovalMode;
    fn execute(&self, arguments: serde_json::Value) -> Result<String, String>;
}
```

**New File: `src-tauri/src/tools/registry.rs`**
```rust
pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self
    pub fn register(&mut self, tool: Box<dyn Tool>)
    pub fn get(&self, name: &str) -> Option<&dyn Tool>
    pub fn list(&self) -> Vec<ToolDefinition>
    pub fn execute(&self, call: &ToolCall) -> ToolResult
}
```

**New File: `src-tauri/src/tools/parser.rs`**
```rust
pub enum ToolCallFormat {
    Functionary,  // <function_calls>XML</function_calls>
    CommandR,     // JSON array
    Llama31,      // Native with special tokens
}

pub fn parse_tool_calls(
    output: &str,
    format: ToolCallFormat,
) -> Result<Vec<ToolCall>, String>
```

Implement parsers for:
- Functionary: Regex for `<function_calls>...</function_calls>` XML
- Command-R: JSON array parsing `[{"name": "...", "parameters": {...}}]`
- LLaMA 3.1: Special token detection + JSON

#### Backend: Model Metadata

**File: `src/types/index.ts`**
```typescript
export interface ToolCallFormat {
  type: 'functionary' | 'command-r' | 'llama31';
  startMarker?: string;
  endMarker?: string;
}

export interface Model {
  // ... existing fields
  supportsTools?: boolean;
  toolCallFormat?: ToolCallFormat;
}
```

#### Tauri Commands

**File: `src-tauri/src/lib.rs`**
```rust
#[tauri::command]
async fn list_available_tools() -> Result<Vec<ToolDefinition>, String>

#[tauri::command]
async fn execute_tool(
    tool_call: ToolCall,
    registry: State<'_, Arc<Mutex<ToolRegistry>>>,
) -> Result<ToolResult, String>

// Request approval from frontend before executing
#[tauri::command]
async fn check_tool_approval_needed(tool_name: String) -> Result<bool, String>
```

#### Frontend: Tool Types

**File: `src/types/index.ts`**
```typescript
export interface ToolDefinition {
  name: string;
  description: string;
  parameters: object;  // JSON Schema
  requiresApproval: boolean;
}

export interface ToolCall {
  id: string;
  name: string;
  arguments: object;
}

export interface ToolExecution {
  id: string;
  toolName: string;
  arguments: object;
  result?: string;
  error?: string;
  status: 'pending' | 'approved' | 'executing' | 'completed' | 'failed';
  timestamp: Date;
}

export interface ChatMessage {
  // ... existing fields
  toolCalls?: ToolCall[];
  toolResults?: ToolResult[];
}
```

### Phase 3: Tool Execution Flow

**Goal**: Implement the generation loop with tool execution

#### Backend: Tool-Aware Generation

**File: `src-tauri/src/inference_engine.rs`**

Add new method:
```rust
pub fn generate_with_tools(
    &mut self,
    conversation: &[ChatMessage],
    system_prompt: Option<&str>,
    image_rgb: &[u8],
    image_width: u32,
    image_height: u32,
    available_tools: &[ToolDefinition],
    tool_format: ToolCallFormat,
    max_iterations: usize,
) -> Result<GenerationResult, String>
```

**Algorithm**:
1. Inject tool definitions into system prompt or first message
2. Generate response
3. Parse output for tool calls using `tool_format`
4. If tool calls found:
   - Return tool calls to frontend for approval/execution
   - Frontend executes tools (approved ones)
   - Frontend sends tool results back
   - Inject tool results as new messages
   - Generate again (up to `max_iterations`)
5. Return final text response

**New Struct**:
```rust
pub struct GenerationResult {
    pub text: String,
    pub tool_calls: Vec<ToolCall>,
    pub finished: bool,  // true if no more tool calls
}
```

#### Backend: New Tauri Command

**File: `src-tauri/src/lib.rs`**
```rust
#[tauri::command]
async fn generate_with_tools(
    conversation: Vec<ChatMessage>,
    system_prompt: Option<String>,
    image_data: Vec<u8>,
    image_width: u32,
    image_height: u32,
    enabled_tools: Vec<String>,
    engine: State<'_, SharedInferenceEngine>,
    registry: State<'_, Arc<Mutex<ToolRegistry>>>,
) -> Result<GenerationResult, String>
```

#### Frontend: Tool Execution Orchestration

**File: `src/App.tsx`**

Add tool execution state:
```typescript
const [toolExecutions, setToolExecutions] = useState<ToolExecution[]>([]);
const [availableTools, setAvailableTools] = useState<ToolDefinition[]>([]);
const [enabledTools, setEnabledTools] = useState<string[]>([]);
```

Modify message handling:
```typescript
const handleSendMessage = async (message: string) => {
  // 1. Add user message
  const userMessage = createMessage('user', message);
  setMessages([...messages, userMessage]);

  // 2. Call generate_with_tools
  let finished = false;
  let iterations = 0;
  const maxIterations = 5;

  while (!finished && iterations < maxIterations) {
    const result = await invoke('generate_with_tools', {
      conversation: [...messages, userMessage],
      systemPrompt,
      imageData: rgbData,
      imageWidth: img.width,
      imageHeight: img.height,
      enabledTools,
    });

    if (result.tool_calls.length > 0) {
      // 3. Show tool calls to user (in expandable sections)
      const executions = result.tool_calls.map(call => ({
        id: call.id,
        toolName: call.name,
        arguments: call.arguments,
        status: 'pending',
      }));
      setToolExecutions(executions);

      // 4. Execute approved tools
      const toolResults = await executeTools(result.tool_calls);

      // 5. Add tool results to conversation
      messages.push({
        role: 'tool',
        content: JSON.stringify(toolResults),
        toolResults,
      });

      iterations++;
    } else {
      // 6. No tool calls, generation finished
      finished = true;
      setMessages([...messages, {
        role: 'assistant',
        content: result.text,
      }]);
    }
  }
};

const executeTools = async (toolCalls: ToolCall[]) => {
  const results = [];
  for (const call of toolCalls) {
    // Check if approval needed
    const needsApproval = await invoke('check_tool_approval_needed', {
      toolName: call.name
    });

    if (needsApproval) {
      // Show approval dialog
      const approved = await showToolApprovalDialog(call);
      if (!approved) continue;
    }

    // Execute tool
    const result = await invoke('execute_tool', { toolCall: call });
    results.push(result);
  }
  return results;
};
```

#### Frontend: Tool Display Components

**New Component: `src/components/ToolExecutionCard.tsx`**
```tsx
interface ToolExecutionCardProps {
  execution: ToolExecution;
  expanded: boolean;
  onToggleExpanded: () => void;
}
```

Displays:
- Tool name and status icon
- Expandable section with arguments (JSON formatted)
- Expandable section with result/error
- Execution time

**New Component: `src/components/ToolApprovalDialog.tsx`**
```tsx
interface ToolApprovalDialogProps {
  toolCall: ToolCall;
  toolDefinition: ToolDefinition;
  onApprove: () => void;
  onDeny: () => void;
}
```

Shows:
- Tool name and description
- Arguments (formatted)
- "Always allow this tool" checkbox
- Approve/Deny buttons

**File: `src/components/ChatPanel.tsx`**

Update message rendering to support tool messages:
- Display tool calls as expandable ToolExecutionCard components
- Show tool results with formatted output
- Add visual indicators for tool execution status

### Phase 4: Core Tools Implementation

**Goal**: Implement 4-5 essential tools

#### Tool 1: Calculator

**File: `src-tauri/src/tools/calculator.rs`**
```rust
pub struct CalculatorTool;

impl Tool for CalculatorTool {
    fn name(&self) -> &str { "calculator" }
    fn description(&self) -> &str {
        "Perform mathematical calculations. Use for arithmetic, algebra, etc."
    }
    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        })
    }
    fn requires_approval(&self) -> ApprovalMode {
        ApprovalMode::Never  // Safe, auto-approve
    }
    fn execute(&self, args: serde_json::Value) -> Result<String, String> {
        // Use meval-rs for safe expression evaluation
    }
}
```

#### Tool 2: File Read

**File: `src-tauri/src/tools/filesystem.rs`**
```rust
pub struct ReadFileTool;

impl Tool for ReadFileTool {
    fn name(&self) -> &str { "read_file" }
    fn description(&self) -> &str {
        "Read contents of a text file from the filesystem"
    }
    fn requires_approval(&self) -> ApprovalMode {
        ApprovalMode::Configurable { default: true }  // Require approval by default
    }
    fn execute(&self, args: serde_json::Value) -> Result<String, String> {
        // Validate path (no traversal attacks)
        // Read file with size limit
        // Return contents
    }
}
```

#### Tool 3: List Directory

```rust
pub struct ListDirectoryTool;
// Similar pattern, requires approval
```

#### Tool 4: Get Current Time

**File: `src-tauri/src/tools/system.rs`**
```rust
pub struct GetTimeTool;

impl Tool for GetTimeTool {
    fn requires_approval(&self) -> ApprovalMode {
        ApprovalMode::Never  // Safe, auto-approve
    }
    fn execute(&self, args: serde_json::Value) -> Result<String, String> {
        // Return current date/time in ISO format
    }
}
```

#### Tool 5: Web Search (Optional)

```rust
pub struct WebSearchTool;
// Requires approval, uses DuckDuckGo API or similar
```

#### Register Tools

**File: `src-tauri/src/lib.rs`** (in setup function)
```rust
fn main() {
    let tool_registry = Arc::new(Mutex::new(ToolRegistry::new()));

    {
        let mut registry = tool_registry.lock().unwrap();
        registry.register(Box::new(CalculatorTool));
        registry.register(Box::new(ReadFileTool));
        registry.register(Box::new(ListDirectoryTool));
        registry.register(Box::new(GetTimeTool));
    }

    tauri::Builder::default()
        .manage(tool_registry)
        // ... rest of setup
}
```

### Phase 5: Tool Configuration UI

**Goal**: Allow users to configure tool settings

**New Component: `src/components/ToolSettings.tsx`**

Settings panel showing:
- List of all available tools
- Enable/disable toggle per tool
- Approval setting per tool (Always/Never/Ask)
- Tool documentation (name, description, parameters)

**Integration**: Add settings button to toolbar that opens ToolSettings modal

### Phase 6: Polish and Error Handling

#### Error Handling
- Tool execution timeouts (30s default)
- Malformed tool call handling (show error, continue generation)
- Tool execution failures (display error, allow retry)
- Max iteration limit (prevent infinite loops)

#### Performance
- Tool result caching (optional)
- Async tool execution
- Progress indicators for long-running tools

#### Security
- Path validation for file operations
- Command injection prevention
- Resource limits (file size, execution time)
- Sandboxing for sensitive operations

## Critical Files Summary

### Backend (Rust)
- `src-tauri/src/lib.rs` - Add Tauri commands for tools and system prompts
- `src-tauri/src/inference_engine.rs` - Add `generate_with_tools()` method
- `src-tauri/src/system_prompts.rs` - NEW: System prompt persistence
- `src-tauri/src/tools/mod.rs` - NEW: Tool trait and core types
- `src-tauri/src/tools/registry.rs` - NEW: Tool registry
- `src-tauri/src/tools/parser.rs` - NEW: Multi-format tool call parsing
- `src-tauri/src/tools/calculator.rs` - NEW: Calculator tool
- `src-tauri/src/tools/filesystem.rs` - NEW: File system tools
- `src-tauri/src/tools/system.rs` - NEW: System info tools

### Frontend (TypeScript/React)
- `src/App.tsx` - Add system prompt state and tool execution orchestration
- `src/types/index.ts` - Add tool-related types
- `src/components/ChatPanel.tsx` - Update message rendering for tools
- `src/components/SystemPromptEditor.tsx` - NEW: System prompt UI
- `src/components/ToolExecutionCard.tsx` - NEW: Display tool execution
- `src/components/ToolApprovalDialog.tsx` - NEW: Tool approval UI
- `src/components/ToolSettings.tsx` - NEW: Tool configuration panel

## Testing Strategy

### Unit Tests
- Tool execution with various inputs
- Tool call parsing for each format (Functionary, Command-R, LLaMA 3.1)
- System prompt loading/saving
- Security: path traversal, command injection attempts

### Integration Tests
- Full conversation with tool calls
- Multi-turn tool usage
- Tool approval flow
- System prompt + tools together

### Manual Testing
- Test with models supporting each tool format
- Verify UI displays correctly
- Test error scenarios
- Performance with multiple tools

## Security Considerations

1. **File Access**: Validate all file paths, prevent directory traversal
2. **Command Execution**: If implementing shell tools, use strict allowlists
3. **Network Access**: Rate limiting for web tools
4. **Resource Limits**: Timeout, max file size, max iterations
5. **Approval System**: Sensitive tools require user confirmation
6. **Input Validation**: Sanitize all tool parameters

## Next Steps After Implementation

1. Add more tools based on user needs
2. Implement tool result caching
3. Add tool usage analytics/logging
4. Support for custom user-defined tools
5. Tool marketplace/plugin system (future)
