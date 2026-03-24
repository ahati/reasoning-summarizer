# reasoning-summarizer

A Go package for summarizing LLM reasoning/thinking processes using llama.cpp with Qwen 3.5.

## Features

- **Zero-config llama.cpp**: Bundled as a git submodule with static linking
- **Qwen 3.5 support**: Optimized for Qwen 3.5 small models (0.8B)
- **Streaming summarization**: Process content chunks as they arrive
- **Reasoning extraction**: Parse and summarize `<thinking>` tag content
- **Simple Go API**: Clean, idiomatic Go interface
- **CLI tool included**: Ready-to-use command-line application

## Requirements

- Go 1.23+
- C/C++ compiler (gcc or clang)
- CMake
- Make

## Installation

```bash
# Clone with submodules
git clone --recursive https://github.com/your/reasoning-summarizer.git
cd reasoning-summarizer

# Build llama.cpp and Go binary
make

# Or build separately
make llama  # Build llama.cpp static libraries
make build  # Build Go package
```

## Quick Start

### CLI Usage

```bash
# Download a model
make model

# Run basic demo
./reasoning-summarizer -model models/Qwen3.5-0.8B-Q4_K_M.gguf -demo

# Streaming content demo
./reasoning-summarizer -model models/Qwen3.5-0.8B-Q4_K_M.gguf -stream-demo

# Reasoning stream demo
./reasoning-summarizer -model models/Qwen3.5-0.8B-Q4_K_M.gguf -reasoning-demo
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `-model` | (required) | Path to GGUF model file |
| `-demo` | false | Run basic demo |
| `-stream-demo` | false | Run streaming content demo |
| `-reasoning-demo` | false | Run reasoning stream demo |
| `-ctx` | 2048 | Context size |
| `-threads` | 0 | CPU threads (0 = auto) |
| `-max-tokens` | 256 | Max summary tokens |
| `-chunk` | 500 | Chunk size for streaming (chars) |
| `-v` | false | Verbose logging |

## Go Package Usage

### Basic Summarization

```go
package main

import (
    "context"
    "fmt"
    "log"

    "reasoning-summarizer/summarizer"
)

func main() {
    cfg := summarizer.DefaultConfig("models/Qwen3.5-0.8B-Q4_K_M.gguf")
    
    s, err := summarizer.New(cfg)
    if err != nil {
        log.Fatal(err)
    }
    defer s.Close()

    summary, err := s.Summarize(context.Background(), 
        "The user asks about AI. I recall AI stands for artificial intelligence.")
    if err != nil {
        log.Fatal(err)
    }

    fmt.Println(summary.Text)
}
```

### Pipeline with Reasoning Extraction

```go
// Process stream with <thinking> tags
stream := strings.NewReader(`data: <thinking>
data: Let me analyze this...
data: </thinking>
data: The answer is 42.`)

result, err := s.Pipeline(context.Background(), stream)
// result.Extract.Reasoning -> "Let me analyze this..."
// result.Extract.Output    -> "The answer is 42."
// result.Summary.Text      -> "AI stands for artificial intelligence."
```

### Streaming Summarization

```go
streamCfg := summarizer.DefaultStreamingConfig(cfg)
streamCfg.ChunkSize = 500
streamCfg.OnSummary = func(s summarizer.Summary, chunkNum int) {
    fmt.Printf("Chunk %d: %s\n", chunkNum, s.Text)
}

ss, err := summarizer.NewStreaming(streamCfg)
if err != nil {
    log.Fatal(err)
}
defer ss.Close()

result, err := ss.ProcessStream(context.Background(), reader)
fmt.Println("Final:", result.FinalSummary)
```

## API Reference

### Package `summarizer`

```go
// Types
type Config struct {
    ModelPath         string
    ContextSize       int
    Threads           int
    MaxSummaryTokens  int
    MaxReasoningChars int
    GPULayers         int
    Logger            *slog.Logger
}

type Summary struct {
    Text      string
    Latency   time.Duration
    Truncated bool
}

type ExtractResult struct {
    HasReasoning bool
    Reasoning    string
    Output       string
}

type PipelineResult struct {
    Extract ExtractResult
    Summary Summary
}

// Functions
func DefaultConfig(modelPath string) Config
func New(cfg Config) (*Summarizer, error)
func (s *Summarizer) Summarize(ctx context.Context, reasoning string) (Summary, error)
func (s *Summarizer) Pipeline(ctx context.Context, r io.Reader) (PipelineResult, error)
func (s *Summarizer) Close()

// Streaming
func DefaultStreamingConfig(baseConfig Config) StreamingConfig
func NewStreaming(cfg StreamingConfig) (*StreamingSummarizer, error)
func (ss *StreamingSummarizer) ProcessStream(ctx context.Context, r io.Reader) (StreamingResult, error)
func (ss *StreamingSummarizer) StreamWithReasoning(ctx context.Context, r io.Reader) (StreamingResult, error)
```

## Project Structure

```
reasoning-summarizer/
├── llama/llama.go              # CGo bindings to llama.cpp
├── summarizer/
│   ├── summarizer.go           # Core summarization API
│   ├── streaming.go            # Streaming support
│   └── summarizer_test.go      # Unit tests
├── cmd/reasoning-summarizer/
│   └── main.go                 # CLI application
├── llama.cpp/                  # Git submodule
│   ├── include/                # Header files
│   └── build/                  # Static libraries
├── models/                     # GGUF models (gitignored)
├── go.mod
├── Makefile
├── README.md
└── AGENTS.md                   # Contributor guide
```

## Getting Models

Download GGUF models from Hugging Face:

```bash
# Qwen 3.5 0.8B (recommended, ~500MB)
curl -L -o models/Qwen3.5-0.8B-Q4_K_M.gguf \
  "https://huggingface.co/Romarchive/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5%200.8B%20Q4_K_M%20(2026)%5BRomarchive%5D.gguf"

# Or use make
make model
```

## How It Works

1. **llama.cpp submodule**: Source code included for transparent builds
2. **Static linking**: All libraries compiled into the binary (no external dependencies)
3. **CGo integration**: Direct C API calls via `#cgo` directives
4. **Memory management**: KV cache cleared between runs for multiple summarizations

## Testing

```bash
# Run all tests
go test -v ./...

# With coverage
go test -cover ./...

# Run specific test
go test -v ./summarizer/... -run TestExtractFromStream
```

## License

MIT License
