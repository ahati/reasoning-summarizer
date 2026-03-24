# Makefile for reasoning-summarizer
# Builds llama.cpp from source automatically

.PHONY: all build clean test install llama static-libs

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

# Default: build everything
all: $(BINARY)

# Build the Go binary (llama.cpp must be built first)
$(BINARY): $(STATIC_LIBS)
	go build -o $(BINARY) ./cmd/reasoning-summarizer

# Build llama.cpp static libraries
$(STATIC_LIBS): $(LLAMA_DIR)/CMakeLists.txt
	@echo "Building llama.cpp static libraries..."
	mkdir -p $(LLAMA_BUILD)
	cd $(LLAMA_BUILD) && cmake .. \
		-DBUILD_SHARED_LIBS=OFF \
		-DLLAMA_BUILD_EXAMPLES=OFF \
		-DLLAMA_BUILD_SERVER=OFF \
		-DCMAKE_BUILD_TYPE=Release
	$(MAKE) -C $(LLAMA_BUILD) llama ggml ggml-base ggml-cpu

# Ensure llama.cpp source exists (via git submodule)
$(LLAMA_DIR)/CMakeLists.txt:
	@echo "Initializing llama.cpp submodule..."
	git submodule update --init --recursive

# Build only llama.cpp (for package import preparation)
llama: $(STATIC_LIBS)

# Alias for static libraries
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

# Deep clean (including submodule)
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
	@echo "  build        Build the Go package"
	@echo "  llama        Build only llama.cpp static libraries"
	@echo "  test         Run tests"
	@echo "  install      Install binary to /usr/local/bin"
	@echo "  clean        Remove build artifacts"
	@echo "  model        Download test model"
	@echo "  demo         Run demo with test model"
