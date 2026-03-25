# reasoning-summarizer

A Go package for summarizing LLM reasoning/thinking processes using llama.cpp with Qwen 3.5.

## Features

- **Automatic dependency fetching**: llama.cpp is fetched and built via `go generate`
- **No submodules**: Works when cloned directly from GitHub without `--recursive`
- **Qwen 3.5 support**: Optimized for Qwen 3.5 small models (0.8B)
- **Aggressive compression**: Outputs 3-5 word summaries with ~95% compression
- **JSON-based extraction**: Reliable output format handling Qwen's thinking tokens
- **Streaming summarization**: Process content chunks as they arrive
- **Reasoning extraction**: Parse and summarize `<thinking>` tag content
- **Simple Go API**: Clean, idiomatic Go interface

## Requirements

- Go 1.23+
- C/C++ compiler (gcc or clang)
- CMake
- Make
- Git (for fetching llama.cpp)

## Installation

```bash
# Clone the repository (no --recursive needed)
git clone https://github.com/ahati/reasoning-summarizer.git
cd reasoning-summarizer

# Fetch and build llama.cpp dependencies
go generate ./...

# Build the Go binary
go build ./cmd/reasoning-summarizer

# Or use make for everything
make
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

    "github.com/ahati/reasoning-summarizer/summarizer"
)

func main() {
    cfg := summarizer.DefaultConfig("models/Qwen3.5-0.8B-Q4_K_M.gguf")
    cfg.MaxSummaryTokens = 30  // Force short output

    s, err := summarizer.New(cfg)
    if err != nil {
        log.Fatal(err)
    }
    defer s.Close()

    summary, err := s.Summarize(context.Background(),
        "I need to think about what the user is asking. They want to know the capital of France. I remember Paris is the capital.")
    if err != nil {
        log.Fatal(err)
    }

    fmt.Println(summary.Text) // Output: "Paris"
}
```

### Compression Results

| Input | Output | Compression |
|-------|--------|-------------|
| "I need to think about the capital of France..." | "Paris" | **2.9%** |
| "Let me calculate 2+2=4 and 4*3=12..." | "12" | **0.9%** |
| "Sort array in JavaScript..." | "sort function" | **3.8%** |

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
    MaxSummaryTokens  int     // Default: 256, use 20-30 for ultra-short
    MaxReasoningChars int     // Default: 6000
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
├── llama/
│   ├── llama.go              # CGo bindings to llama.cpp
│   ├── generate.go           # go:generate directive
│   └── fetch-llama.sh        # Script to fetch llama.cpp
├── summarizer/
│   ├── summarizer.go         # Core summarization API
│   ├── streaming.go          # Streaming support
│   └── summarizer_test.go    # Unit tests
├── cmd/reasoning-summarizer/
│   └── main.go               # CLI application
├── llama.cpp/                # Fetched by go:generate
│   ├── include/              # Header files
│   └── build/                # Static libraries
├── models/                   # GGUF models (gitignored)
├── go.mod
├── Makefile
├── README.md
└── AGENTS.md                 # Contributor guide
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

1. **JSON-based prompt**: Requests output as `{"summary": "..."}` for reliable extraction
2. **Thinking token handling**: Qwen 3.5 outputs thinking tokens (Tibetan chars) which are stripped
3. **Fallback safety**: If summary exceeds input length, original is returned
4. **Memory management**: KV cache cleared between runs for multiple summarizations

## Dependency Management

llama.cpp is fetched automatically during `go generate`:

```bash
# Fetch and build llama.cpp
go generate ./...

# Or fetch a specific version
LLAMA_CPP_VERSION=b8508 go generate ./...

# Environment variables
LLAMA_CPP_VERSION  - llama.cpp version/commit/tag (default: b8508)
LLAMA_CPP_REPO     - Git repository URL (default: https://github.com/ggml-org/llama.cpp.git)
CMAKE_BUILD_TYPE   - Build type (default: Release)
```

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