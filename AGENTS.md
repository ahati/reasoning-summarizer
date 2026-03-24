# Repository Guidelines

## Project Structure & Module Organization

```
reasoning-summarizer/
├── llama/                   # CGo bindings to llama.cpp
│   └── llama.go             # Low-level C API wrappers
├── summarizer/              # Core summarization package
│   ├── summarizer.go        # Main API (New, Summarize, Pipeline)
│   ├── streaming.go         # Streaming summarization support
│   └── summarizer_test.go   # Unit tests
├── cmd/reasoning-summarizer/ # CLI application
│   └── main.go              # Entry point with demo modes
├── llama.cpp/               # Git submodule (llama.cpp source)
│   └── build/               # Compiled static libraries
├── models/                  # GGUF model files (not in git)
├── go.mod                   # Module: reasoning-summarizer
└── Makefile                 # Build automation
```

## Build, Test, and Development Commands

```bash
# Build llama.cpp static libraries (required first)
make llama

# Build the Go package
make build

# Build CLI binary
make

# Run unit tests
make test
# or: go test -v ./...

# Run demos with Qwen 3.5 model
./reasoning-summarizer -model models/Qwen3.5-0.8B-Q4_K_M.gguf -demo
./reasoning-summarizer -model models/Qwen3.5-0.8B-Q4_K_M.gguf -stream-demo
./reasoning-summarizer -model models/Qwen3.5-0.8B-Q4_K_M.gguf -reasoning-demo

# Download test model
make model

# Clean build artifacts
make clean
```

## Coding Style & Naming Conventions

- **Go version**: 1.23+
- **Formatting**: Use `gofmt` and `goimports`
- **Naming**: 
  - Exported functions/types: `PascalCase` (e.g., `NewSummarizer`, `StreamingConfig`)
  - Internal functions: `camelCase`
  - Package names: lowercase, single word (`llama`, `summarizer`)
- **Documentation**: Add Go doc comments to all exported types and functions
- **CGo**: Keep CGo code isolated in `llama/` package; document memory management clearly

## Testing Guidelines

- **Framework**: Go standard `testing` package
- **Location**: Tests placed alongside source files (`*_test.go`)
- **Naming**: `Test<FunctionName>` pattern (e.g., `TestExtractFromStream`)
- **Coverage**: Run `go test -cover ./...`
- **Table-driven tests**: Preferred for multiple test cases

```bash
# Run all tests with verbose output
go test -v ./...

# Run tests for specific package
go test -v ./summarizer/...
```

## Commit & Pull Request Guidelines

- **Commit messages**: Use conventional format
  - `feat: add streaming summarization support`
  - `fix: clear KV cache between summarizations`
  - `docs: update README with API examples`
  - `test: add unit tests for ExtractFromStream`
- **Pull requests**:
  - Include clear description of changes
  - Reference any related issues
  - Ensure `go build ./...` and `go test ./...` pass
  - Update documentation for API changes

## Architecture Notes

- **llama.cpp submodule**: Must be initialized with `git clone --recursive` or `git submodule update --init`
- **Static linking**: llama.cpp libraries are statically linked via CGo LDFLAGS
- **Memory management**: Call `ClearMemory()` between inference runs to reset context
- **Thread safety**: `Summarizer` uses mutex locking; safe for concurrent use via `Summarize()` method only
