// Package summarizer provides reasoning summarization using llama.cpp.
// The llama.cpp library is automatically built when this package is imported.
//
// Usage:
//
//	cfg := summarizer.DefaultConfig("path/to/model.gguf")
//	s, err := summarizer.New(cfg)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer s.Close()
//
//	summary, err := s.Summarize(ctx, reasoningText)
package summarizer

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"runtime"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/ahati/reasoning-summarizer/llama"
)

// Config holds the configuration for a Summarizer.
type Config struct {
	// ModelPath is the path to the GGUF model file.
	ModelPath string
	// ContextSize is the context window size for the model.
	ContextSize int
	// Threads is the number of CPU threads to use (0 = auto).
	Threads int
	// MaxSummaryTokens is the maximum number of tokens to generate in a summary.
	MaxSummaryTokens int
	// MaxReasoningChars limits the input reasoning text length (0 = unlimited).
	MaxReasoningChars int
	// GPULayers is the number of layers to offload to GPU (0 = CPU-only).
	GPULayers int
	// Logger is the optional logger for debug output.
	Logger *slog.Logger
}

// DefaultConfig returns a Config with sensible defaults for the given model path.
func DefaultConfig(modelPath string) Config {
	return Config{
		ModelPath:         modelPath,
		ContextSize:       2048,
		Threads:           0,
		MaxSummaryTokens:  256,
		MaxReasoningChars: 6000,
		GPULayers:         0,
		Logger:            slog.Default(),
	}
}

// Summarizer performs reasoning summarization using a local LLM.
type Summarizer struct {
	cfg   Config
	model *llama.Model
	ctx   *llama.Context
	mu    sync.Mutex
	log   *slog.Logger
}

// New creates a new Summarizer with the given configuration.
// The llama.cpp backend is initialized and the model is loaded into memory.
func New(cfg Config) (*Summarizer, error) {
	if cfg.ModelPath == "" {
		return nil, errors.New("model path required")
	}
	if _, err := os.Stat(cfg.ModelPath); err != nil {
		return nil, fmt.Errorf("model not found: %w", err)
	}
	log := cfg.Logger
	if log == nil {
		log = slog.Default()
	}
	if cfg.Threads == 0 {
		cfg.Threads = runtime.NumCPU()
	}

	log.Info("initializing backend")
	llama.BackendInit()
	llama.LoadBackends()

	start := time.Now()
	log.Info("loading model", "path", cfg.ModelPath, "ctx", cfg.ContextSize)

	model, err := llama.LoadModel(cfg.ModelPath, cfg.GPULayers)
	if err != nil {
		llama.BackendFree()
		return nil, err
	}

	ctx, err := llama.NewContext(model, cfg.ContextSize, cfg.Threads)
	if err != nil {
		model.Close()
		llama.BackendFree()
		return nil, err
	}

	log.Info("model loaded", "elapsed", time.Since(start).Round(time.Millisecond))
	return &Summarizer{cfg: cfg, model: model, ctx: ctx, log: log}, nil
}

// Close releases all resources held by the Summarizer.
func (s *Summarizer) Close() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.ctx != nil {
		s.ctx.Close()
		s.ctx = nil
	}
	if s.model != nil {
		s.model.Close()
		s.model = nil
	}
	llama.BackendFree()
}

// Summary contains the result of a summarization operation.
type Summary struct {
	// Text is the generated summary.
	Text string
	// Latency is the time taken to generate the summary.
	Latency time.Duration
	// Truncated indicates if the input was truncated.
	Truncated bool
}

// Summarize generates a concise summary of the given reasoning text.
func (s *Summarizer) Summarize(ctx context.Context, reasoning string) (Summary, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.model == nil {
		return Summary{}, errors.New("summarizer closed")
	}

	// Clear KV cache for new sequence
	s.ctx.ClearMemory()

	start := time.Now()
	truncated := false
	if s.cfg.MaxReasoningChars > 0 && len(reasoning) > s.cfg.MaxReasoningChars {
		reasoning = truncateMiddle(reasoning, s.cfg.MaxReasoningChars)
		truncated = true
	}

	// Prompt uses aggressive compression.
	// Qwen 3.5 is a reasoning model that outputs thinking tokens.
	// We request JSON format to extract the summary reliably.
	var prompt string
	if strings.Contains(strings.ToLower(s.cfg.ModelPath), "tinyllama") || strings.Contains(strings.ToLower(s.cfg.ModelPath), "llama") {
		// Llama-2 format for TinyLlama
		prompt = fmt.Sprintf(
			`<>
Key words from this text (max 5 words):
%s
Key words:</s>
`, reasoning)
	} else {
		// ChatML format for Qwen - request JSON output
		prompt = fmt.Sprintf(
			`<|im_start|>system
Extract the main point in under 10 words.
Output as JSON: {"summary": "your summary here"}<|im_end|>
<|im_start|>user
%s<|im_end|>
<|im_start|>assistant
{"summary": "`, reasoning)
	}

	tokens, err := s.ctx.Tokenize(prompt)
	if err != nil {
		return Summary{}, err
	}

	s.log.Debug("inference started", "tokens", len(tokens))
	if err := s.ctx.Decode(tokens); err != nil {
		return Summary{}, err
	}

	sampler := llama.NewGreedySampler()
	defer sampler.Close()

	var output strings.Builder
	eos := s.model.EOSToken()

	for i := 0; i < s.cfg.MaxSummaryTokens; i++ {
		token := sampler.Sample(s.ctx)
		if token == eos {
			break
		}
		piece := s.model.TokenToPiece(token)
		if piece == "" || piece == "</s>" || piece == "<|im_end|>" {
			break
		}
		output.WriteString(piece)
		if err := s.ctx.Decode([]int32{token}); err != nil {
			break
		}
	}

	latency := time.Since(start)
	rawOutput := strings.TrimSpace(output.String())

	// Clean up thinking tokens and artifacts that some models emit
	cleanedOutput := cleanSummary(rawOutput)

	s.log.Debug("inference complete", "latency", latency, "chars", len(cleanedOutput), "raw_chars", len(rawOutput))
	return Summary{Text: cleanedOutput, Latency: latency, Truncated: truncated}, nil
}

// cleanSummary removes thinking tokens and artifacts from model output.
func cleanSummary(s string) string {
	// Qwen 3.5 reasoning models output thinking in this format:
	// ང thinking_content ང actual_response
	// The thinking token is Unicode U+0F04 (Tibetan mark initial form)

	// Count occurrences of thinking token
	thinkingToken := "<tool_call>"
	count := strings.Count(s, thinkingToken)

	if count >= 2 {
		// Normal case: thinking_content ང actual_content
		// Find the position after the second thinking token
		idx := strings.Index(s, thinkingToken)
		rest := s[idx+len(thinkingToken):]
		idx2 := strings.Index(rest, thinkingToken)
		if idx2 >= 0 {
			s = strings.TrimSpace(rest[idx2+len(thinkingToken):])
		}
	} else if count == 1 {
		// Only one thinking token - check what comes after
		idx := strings.Index(s, thinkingToken)
		after := strings.TrimSpace(s[idx+len(thinkingToken):])

		// If it starts with thinking process markers, the model failed
		if strings.HasPrefix(after, "Thinking Process") ||
			strings.HasPrefix(after, "Analyze") ||
			strings.HasPrefix(after, "**Task") ||
			strings.HasPrefix(after, "* Task") {
			// Return empty to trigger fallback
			return ""
		}
		s = after
	}

	// Extract summary from JSON if present
	// Look for {"summary": "..."} or just the value after {"summary": "
	if strings.Contains(s, `"summary"`) {
		// Find the start of the summary value
		if idx := strings.Index(s, `"summary": "`); idx >= 0 {
			start := idx + len(`"summary": "`)
			// Find the end quote
			if endIdx := strings.Index(s[start:], `"`); endIdx > 0 {
				s = s[start : start+endIdx]
			} else if endIdx := strings.Index(s[start:], `"}`); endIdx > 0 {
				s = s[start : start+endIdx]
			}
		}
	}

	// Remove stop tokens
	s = strings.Split(s, "<|im_end|")[0]
	s = strings.Split(s, "<|im_start|")[0]
	s = strings.Split(s, "</s>")[0]
	s = strings.Split(s, `"}`)[0]
	s = strings.Split(s, `"}`)[0]

	// Remove leading labels
	s = strings.TrimPrefix(s, "Summary:")
	s = strings.TrimPrefix(s, "Summary: ")

	// Remove markdown bold
	s = strings.ReplaceAll(s, "**", "")

	// Remove newlines
	s = strings.ReplaceAll(s, "\n", " ")
	s = strings.ReplaceAll(s, "\r", "")

	// Collapse multiple spaces
	for strings.Contains(s, "  ") {
		s = strings.ReplaceAll(s, "  ", " ")
	}

	return strings.TrimSpace(s)
}

func truncateMiddle(s string, maxLen int) string {
	if utf8.RuneCountInString(s) <= maxLen {
		return s
	}
	runes := []rune(s)
	h, t := maxLen/2, maxLen-maxLen/2
	result := make([]rune, maxLen)
	copy(result, runes[:h])
	copy(result[h:], runes[len(runes)-t:])
	return string(result)
}

// ExtractResult contains extracted reasoning and output from a stream.
type ExtractResult struct {
	// HasReasoning indicates if thinking/reasoning content was found.
	HasReasoning bool
	// Reasoning is the extracted thinking/reasoning content.
	Reasoning string
	// Output is the non-reasoning output content.
	Output string
}

// ExtractFromStream parses a stream containing <thinking> tags and extracts
// the reasoning and output portions separately.
func ExtractFromStream(ctx context.Context, r io.Reader) (ExtractResult, error) {
	result := ExtractResult{}
	var reasoning, output strings.Builder
	inThinking := false
	scanner := bufio.NewScanner(r)

	for scanner.Scan() {
		line := scanner.Text()
		data := line
		if strings.HasPrefix(line, "data: ") {
			data = strings.TrimPrefix(line, "data: ")
		} else if line == "" || strings.HasPrefix(line, ":") {
			continue
		}

		if strings.Contains(data, "<thinking>") {
			inThinking = true
			data = strings.ReplaceAll(data, "<thinking>", "")
		}
		if strings.Contains(data, "</thinking>") {
			inThinking = false
			data = strings.ReplaceAll(data, "</thinking>", "")
		}

		if inThinking {
			reasoning.WriteString(data)
		} else if data != "[DONE]" {
			output.WriteString(data)
		}
	}

	result.Reasoning = strings.TrimSpace(reasoning.String())
	result.Output = strings.TrimSpace(output.String())
	result.HasReasoning = result.Reasoning != ""
	return result, scanner.Err()
}

// PipelineResult contains the result of the full pipeline.
type PipelineResult struct {
	Extract ExtractResult
	Summary Summary
}

// Pipeline processes a stream through extraction and summarization.
// It first extracts reasoning from the stream, then generates a summary.
func (s *Summarizer) Pipeline(ctx context.Context, r io.Reader) (PipelineResult, error) {
	extract, err := ExtractFromStream(ctx, r)
	if err != nil {
		return PipelineResult{}, err
	}
	if !extract.HasReasoning {
		return PipelineResult{Extract: extract}, nil
	}
	summary, err := s.Summarize(ctx, extract.Reasoning)
	if err != nil {
		return PipelineResult{Extract: extract}, err
	}
	return PipelineResult{Extract: extract, Summary: summary}, nil
}