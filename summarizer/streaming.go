// Package summarizer provides streaming summarization capabilities.
package summarizer

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"log/slog"
	"strings"
	"sync"
	"time"
)

// StreamingConfig holds configuration for streaming summarization.
type StreamingConfig struct {
	// BaseConfig is the base summarizer configuration.
	BaseConfig Config
	// ChunkSize is the number of characters to accumulate before summarizing.
	ChunkSize int
	// MinChunkSize is the minimum size before emitting a summary.
	MinChunkSize int
	// OnSummary is called when a summary is generated.
	OnSummary func(summary Summary, chunkNum int)
	// OnChunk is called when a chunk is processed.
	OnChunk func(chunk string, chunkNum int)
	// Logger for debug output.
	Logger *slog.Logger
}

// DefaultStreamingConfig returns sensible defaults for streaming.
func DefaultStreamingConfig(baseConfig Config) StreamingConfig {
	return StreamingConfig{
		BaseConfig:   baseConfig,
		ChunkSize:    2000,
		MinChunkSize: 500,
		Logger:       baseConfig.Logger,
	}
}

// StreamingSummarizer processes streaming content and generates summaries.
type StreamingSummarizer struct {
	cfg StreamingConfig
	s   *Summarizer
	mu  sync.Mutex
}

// NewStreaming creates a new streaming summarizer.
func NewStreaming(cfg StreamingConfig) (*StreamingSummarizer, error) {
	s, err := New(cfg.BaseConfig)
	if err != nil {
		return nil, err
	}
	return &StreamingSummarizer{cfg: cfg, s: s}, nil
}

// Close releases resources.
func (ss *StreamingSummarizer) Close() {
	ss.s.Close()
}

// StreamingResult contains the final result of streaming summarization.
type StreamingResult struct {
	// FinalSummary is the combined summary of all chunks.
	FinalSummary string
	// ChunkSummaries are individual summaries for each chunk.
	ChunkSummaries []string
	// TotalChunks is the number of chunks processed.
	TotalChunks int
	// TotalDuration is the total processing time.
	TotalDuration time.Duration
	// InputChars is the total characters processed.
	InputChars int
}

// ProcessStream reads from a stream and generates summaries as content arrives.
func (ss *StreamingSummarizer) ProcessStream(ctx context.Context, r io.Reader) (StreamingResult, error) {
	result := StreamingResult{
		ChunkSummaries: []string{},
	}
	start := time.Now()

	var buffer strings.Builder
	chunkNum := 0
	scanner := bufio.NewScanner(r)
	// Increase buffer size for long lines
	scanner.Buffer(make([]byte, 64*1024), 1024*1024)

	for scanner.Scan() {
		line := scanner.Text()
		
		// Handle SSE format
		data := line
		if strings.HasPrefix(line, "data: ") {
			data = strings.TrimPrefix(line, "data: ")
		} else if line == "" || strings.HasPrefix(line, ":") || line == "data: [DONE]" {
			continue
		}

		buffer.WriteString(data)
		buffer.WriteString(" ")
		result.InputChars += len(data) + 1

		// Check if we have enough content for a chunk
		if buffer.Len() >= ss.cfg.ChunkSize {
			chunk := buffer.String()
			buffer.Reset()

			chunkNum++
			if ss.cfg.OnChunk != nil {
				ss.cfg.OnChunk(chunk, chunkNum)
			}

			summary, err := ss.s.Summarize(ctx, chunk)
			if err != nil {
				ss.cfg.Logger.Error("failed to summarize chunk", "chunk", chunkNum, "err", err)
				continue
			}

			result.ChunkSummaries = append(result.ChunkSummaries, summary.Text)
			if ss.cfg.OnSummary != nil {
				ss.cfg.OnSummary(summary, chunkNum)
			}
		}
	}

	// Process remaining content
	if buffer.Len() >= ss.cfg.MinChunkSize {
		chunk := strings.TrimSpace(buffer.String())
		chunkNum++

		if ss.cfg.OnChunk != nil {
			ss.cfg.OnChunk(chunk, chunkNum)
		}

		summary, err := ss.s.Summarize(ctx, chunk)
		if err != nil {
			ss.cfg.Logger.Error("failed to summarize final chunk", "err", err)
		} else {
			result.ChunkSummaries = append(result.ChunkSummaries, summary.Text)
			if ss.cfg.OnSummary != nil {
				ss.cfg.OnSummary(summary, chunkNum)
			}
		}
	}

	result.TotalChunks = chunkNum
	result.TotalDuration = time.Since(start)

	// Generate final combined summary
	if len(result.ChunkSummaries) > 0 {
		if len(result.ChunkSummaries) == 1 {
			result.FinalSummary = result.ChunkSummaries[0]
		} else {
			combined := strings.Join(result.ChunkSummaries, " ")
			summary, err := ss.s.Summarize(ctx, combined)
			if err != nil {
				ss.cfg.Logger.Error("failed to generate final summary", "err", err)
				result.FinalSummary = combined
			} else {
				result.FinalSummary = summary.Text
			}
		}
	}

	return result, scanner.Err()
}

// StreamWithReasoning processes a stream containing reasoning tags.
func (ss *StreamingSummarizer) StreamWithReasoning(ctx context.Context, r io.Reader) (StreamingResult, error) {
	result := StreamingResult{
		ChunkSummaries: []string{},
	}
	start := time.Now()

	var reasoningBuffer, outputBuffer strings.Builder
	inThinking := false
	totalReasoning := 0

	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 64*1024), 1024*1024)

	for scanner.Scan() {
		line := scanner.Text()
		data := line
		if strings.HasPrefix(line, "data: ") {
			data = strings.TrimPrefix(line, "data: ")
		} else if line == "" || strings.HasPrefix(line, ":") {
			continue
		}

		// Track thinking tags
		if strings.Contains(data, "<thinking>") {
			inThinking = true
			data = strings.ReplaceAll(data, "<thinking>", "")
		}
		if strings.Contains(data, "</thinking>") {
			inThinking = false
			data = strings.ReplaceAll(data, "</thinking>", "")
		}

		if inThinking {
			reasoningBuffer.WriteString(data)
			reasoningBuffer.WriteString(" ")

			// Summarize reasoning when buffer is full
			if reasoningBuffer.Len() >= ss.cfg.ChunkSize {
				chunk := strings.TrimSpace(reasoningBuffer.String())
				reasoningBuffer.Reset()
				totalReasoning += len(chunk)

				summary, err := ss.s.Summarize(ctx, chunk)
				if err != nil {
					ss.cfg.Logger.Error("failed to summarize reasoning chunk", "err", err)
					continue
				}

				result.ChunkSummaries = append(result.ChunkSummaries, summary.Text)
				if ss.cfg.OnSummary != nil {
					ss.cfg.OnSummary(summary, len(result.ChunkSummaries))
				}
			}
		} else if data != "[DONE]" {
			outputBuffer.WriteString(data)
			outputBuffer.WriteString(" ")
		}
	}

	// Process remaining reasoning
	if reasoningBuffer.Len() > 0 {
		chunk := strings.TrimSpace(reasoningBuffer.String())
		totalReasoning += len(chunk)

		summary, err := ss.s.Summarize(ctx, chunk)
		if err != nil {
			ss.cfg.Logger.Error("failed to summarize final reasoning", "err", err)
		} else {
			result.ChunkSummaries = append(result.ChunkSummaries, summary.Text)
		}
	}

	result.InputChars = totalReasoning
	result.TotalChunks = len(result.ChunkSummaries)
	result.TotalDuration = time.Since(start)

	// Combine summaries
	if len(result.ChunkSummaries) > 0 {
		if len(result.ChunkSummaries) == 1 {
			result.FinalSummary = result.ChunkSummaries[0]
		} else {
			combined := strings.Join(result.ChunkSummaries, " ")
			summary, err := ss.s.Summarize(ctx, combined)
			if err != nil {
				result.FinalSummary = combined
			} else {
				result.FinalSummary = summary.Text
			}
		}
	}

	return result, scanner.Err()
}

// PrintStreamingProgress returns a helper function for printing progress.
func PrintStreamingProgress(log *slog.Logger) func(summary Summary, chunkNum int) {
	return func(summary Summary, chunkNum int) {
		fmt.Printf("\n--- Chunk %d Summary (latency: %v) ---\n", chunkNum, summary.Latency.Round(time.Millisecond))
		fmt.Println(summary.Text)
		if summary.Truncated {
			fmt.Println("(input was truncated)")
		}
	}
}

// PrintChunkProgress returns a helper for printing chunk info.
func PrintChunkProgress(log *slog.Logger) func(chunk string, chunkNum int) {
	return func(chunk string, chunkNum int) {
		log.Debug("processing chunk", "num", chunkNum, "chars", len(chunk))
	}
}
