# Makefile for reasoning-summarizer
# Dependencies are fetched automatically via go:generate

.PHONY: all build clean test install generate static-libs

# Paths
LLAMA_DIR := llama.cpp
LLAMA_BUILD := $(LLAMA_DIR)/build
BUILD_DIR := build
BINARY := reasoning-summarizer

# Static libraries from llama.cpp
STATIC_LIBS := $(LLAMA_BUILD)/src/libllama.a \
               $(LLAMA_BUILD)/ggml/src/libggml.a \
               $(LLAMA_BUILD)/ggml/src/libggml-base.a \
               $(LLAMA_BUILD)/ggml/src/libggml-cpu.a

# Default: generate dependencies then build
all: $(BINARY)

# Generate llama.cpp (fetch and build if needed)
generate:
	go generate ./...

# Build the Go binary (llama.cpp must be built first)
$(BINARY): $(STATIC_LIBS)
	go build -o $(BINARY) ./cmd/reasoning-summarizer

# Build llama.cpp static libraries (fetched via go generate if needed)
$(STATIC_LIBS):
	@echo "Fetching and building llama.cpp..."
	go generate ./llama

# Build only llama.cpp (for package import preparation)
static-libs: $(STATIC_LIBS)

# Build the Go package (for library use)
build: $(STATIC_LIBS)
	go build ./...

# Run tests
test: $(STATIC_LIBS)
	go test -v ./...

# Install the binary
install: $(BINARY)
	install -m 755 $(BINARY) /usr/local/bin/

# Clean build artifacts
clean:
	rm -rf $(BINARY)
	rm -rf $(LLAMA_BUILD)
	rm -rf $(BUILD_DIR)

# Deep clean (including fetched llama.cpp)
deep-clean: clean
	rm -rf $(LLAMA_DIR)

# Download a test model
model:
	@echo "Downloading TinyLlama model..."
	mkdir -p models
	curl -L -o models/tinyllama-1.1b-q4_k_m.gguf \
		"https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

# Run demo
demo: $(BINARY) model
	./$(BINARY) -model models/tinyllama-1.1b-q4_k_m.gguf -demo

# Show help
help:
	@echo "reasoning-summarizer - LLM-based reasoning summarizer"
	@echo ""
	@echo "Targets:"
	@echo "  all          Build everything (default)"
	@echo "  generate     Fetch and build llama.cpp (run this first)"
	@echo "  build        Build the Go package"
	@echo "  static-libs  Build only llama.cpp static libraries"
	@echo "  test         Run tests"
	@echo "  install      Install binary to /usr/local/bin"
	@echo "  clean        Remove build artifacts"
	@echo "  deep-clean   Remove build artifacts and fetched llama.cpp"
	@echo "  model        Download test model"
	@echo "  demo         Run demo with test model"
	@echo ""
	@echo "Quick start:"
	@echo "  go generate ./...   # Fetch and build llama.cpp"
	@echo "  go build ./...      # Build the package"