
// Command reasoning-summarizer is a CLI tool for summarizing reasoning text.
package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"strings"
	"time"

	"github.com/ahati/reasoning-summarizer/summarizer"
)

func main() {
	model := flag.String("model", "", "path to GGUF model file")
	ctxSize := flag.Int("ctx", 2048, "context size")
	threads := flag.Int("threads", 0, "CPU threads (0=auto)")
	maxTokens := flag.Int("max-tokens", 256, "max summary tokens")
	verbose := flag.Bool("v", false, "verbose logging")
	demo := flag.Bool("demo", false, "run basic demo")
	streamDemo := flag.Bool("stream-demo", false, "run streaming demo")
	reasoningDemo := flag.Bool("reasoning-demo", false, "run reasoning stream demo")
	chunkSize := flag.Int("chunk", 500, "chunk size for streaming (chars)")
	flag.Parse()

	if *model == "" {
		fmt.Fprintln(os.Stderr, "Usage: reasoning-summarizer -model <path>")
		fmt.Fprintln(os.Stderr, "\nDemos:")
		fmt.Fprintln(os.Stderr, "  -demo            Run basic demo")
		fmt.Fprintln(os.Stderr, "  -stream-demo     Run streaming content demo")
		fmt.Fprintln(os.Stderr, "  -reasoning-demo  Run reasoning stream demo")
		fmt.Fprintln(os.Stderr, "\nOptions:")
		fmt.Fprintln(os.Stderr, "  -model <path>    Path to GGUF model file (required)")
		fmt.Fprintln(os.Stderr, "  -ctx <size>      Context size (default 2048)")
		fmt.Fprintln(os.Stderr, "  -threads <n>     CPU threads (0=auto)")
		fmt.Fprintln(os.Stderr, "  -max-tokens <n>  Max summary tokens (default 256)")
		fmt.Fprintln(os.Stderr, "  -chunk <chars>   Chunk size for streaming (default 500)")
		fmt.Fprintln(os.Stderr, "  -v               Verbose logging")
		os.Exit(1)
	}

	level := slog.LevelInfo
	if *verbose {
		level = slog.LevelDebug
	}
	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: level}))

	cfg := summarizer.DefaultConfig(*model)
	cfg.ContextSize = *ctxSize
	cfg.Threads = *threads
	cfg.MaxSummaryTokens = *maxTokens
	cfg.Logger = logger

	if *streamDemo {
		runStreamDemo(cfg, logger, *chunkSize)
		return
	}
	if *reasoningDemo {
		runReasoningStreamDemo(cfg, logger, *chunkSize)
		return
	}
	if *demo {
		runBasicDemo(cfg, logger)
		return
	}

	s, err := summarizer.New(cfg)
	if err != nil {
		logger.Error("failed to initialize", "err", err)
		os.Exit(1)
	}
	defer s.Close()

	result, err := s.Pipeline(context.Background(), os.Stdin)
	if err != nil {
		logger.Error("pipeline failed", "err", err)
		os.Exit(1)
	}

	fmt.Println("=== Output ===")
	fmt.Println(result.Extract.Output)
	fmt.Println("\n=== Summary ===")
	if result.Extract.HasReasoning {
		fmt.Println(result.Summary.Text)
	} else {
		fmt.Println("(no reasoning found)")
	}
}

func runBasicDemo(cfg summarizer.Config, log *slog.Logger) {
	s, err := summarizer.New(cfg)
	if err != nil {
		log.Error("failed to initialize", "err", err)
		os.Exit(1)
	}
	defer s.Close()

	stream := strings.NewReader(strings.Join([]string{
		"data: <thinking>",
		"data: The user asks for France's capital.",
		"data: I recall Paris is the capital.",
		"data: </thinking>",
		"data: Paris is the capital of France.",
		"data: [DONE]",
	}, "\n"))

	log.Info("running basic demo")
	result, err := s.Pipeline(context.Background(), stream)
	if err != nil {
		log.Error("demo failed", "err", err)
		os.Exit(1)
	}

	fmt.Println("=== Output ===")
	fmt.Println(result.Extract.Output)
	fmt.Println("\n=== Reasoning ===")
	fmt.Println(result.Extract.Reasoning)
	fmt.Println("\n=== Summary ===")
	fmt.Println(result.Summary.Text)
}

func runStreamDemo(cfg summarizer.Config, log *slog.Logger, chunkSize int) {
	streamCfg := summarizer.DefaultStreamingConfig(cfg)
	streamCfg.ChunkSize = chunkSize
	streamCfg.MinChunkSize = 100
	streamCfg.OnSummary = func(s summarizer.Summary, chunkNum int) {
		fmt.Printf("\n--- Chunk %d Summary (latency: %v) ---\n", chunkNum, s.Latency.Round(time.Millisecond))
		fmt.Println(s.Text)
	}
	streamCfg.Logger = log

	ss, err := summarizer.NewStreaming(streamCfg)
	if err != nil {
		log.Error("failed to initialize streaming summarizer", "err", err)
		os.Exit(1)
	}
	defer ss.Close()

	article := generateLongArticle()
	stream := strings.NewReader(article)

	log.Info("running streaming demo", "chunk_size", chunkSize)
	fmt.Println("=== Processing Streaming Content ===")
	fmt.Printf("Total input: %d characters\n\n", len(article))

	result, err := ss.ProcessStream(context.Background(), stream)
	if err != nil {
		log.Error("streaming failed", "err", err)
		os.Exit(1)
	}

	fmt.Println("\n=== Final Summary ===")
	fmt.Println(result.FinalSummary)
	fmt.Printf("\nStats: %d chunks, %d chars processed, %v total time\n",
		result.TotalChunks, result.InputChars, result.TotalDuration.Round(time.Millisecond))
}

func runReasoningStreamDemo(cfg summarizer.Config, log *slog.Logger, chunkSize int) {
	streamCfg := summarizer.DefaultStreamingConfig(cfg)
	streamCfg.ChunkSize = chunkSize
	streamCfg.MinChunkSize = 100
	streamCfg.OnSummary = func(s summarizer.Summary, chunkNum int) {
		fmt.Printf("\n--- Reasoning Chunk %d Summary (latency: %v) ---\n", chunkNum, s.Latency.Round(time.Millisecond))
		fmt.Println(s.Text)
	}
	streamCfg.Logger = log

	ss, err := summarizer.NewStreaming(streamCfg)
	if err != nil {
		log.Error("failed to initialize streaming summarizer", "err", err)
		os.Exit(1)
	}
	defer ss.Close()

	reasoningStream := generateReasoningStream()
	stream := strings.NewReader(reasoningStream)

	log.Info("running reasoning stream demo", "chunk_size", chunkSize)
	fmt.Println("=== Processing Reasoning Stream ===")

	result, err := ss.StreamWithReasoning(context.Background(), stream)
	if err != nil {
		log.Error("streaming failed", "err", err)
		os.Exit(1)
	}

	fmt.Println("\n=== Combined Reasoning Summary ===")
	fmt.Println(result.FinalSummary)
	fmt.Printf("\nStats: %d reasoning chunks, %d chars, %v total time\n",
		result.TotalChunks, result.InputChars, result.TotalDuration.Round(time.Millisecond))
}

func generateLongArticle() string {
	chunks := []string{
		"data: Artificial Intelligence has revolutionized many industries.",
		"data: Machine learning models can now process vast amounts of data.",
		"data: Natural language processing enables computers to understand human language.",
		"data: Computer vision allows machines to interpret visual information.",
		"data: Robotics combines AI with physical machines for automation.",
		"data: Healthcare benefits from AI in diagnosis and drug discovery.",
		"data: Financial services use AI for fraud detection and trading.",
		"data: Autonomous vehicles rely on AI for navigation and safety.",
		"data: AI assistants help with daily tasks and productivity.",
		"data: Ethical considerations are important in AI development.",
		"data: Bias in AI systems is an ongoing challenge.",
		"data: Privacy concerns arise with AI data collection.",
		"data: Job displacement is a concern with AI automation.",
		"data: Education is being transformed by AI-powered tools.",
		"data: Creative industries explore AI-generated content.",
		"data: Scientific research accelerates with AI analysis.",
		"data: Climate modeling benefits from AI predictions.",
		"data: Security systems use AI for threat detection.",
		"data: Supply chain optimization uses AI logistics.",
		"data: Customer service is enhanced by AI chatbots.",
		"data: [DONE]",
	}
	return strings.Join(chunks, "\n")
}

func generateReasoningStream() string {
	return strings.Join([]string{
		"data: <thinking>",
		"data: Let me analyze the problem step by step.",
		"data: First, I need to understand what the user is asking.",
		"data: The question involves multiple concepts that need to be connected.",
		"data: I should consider the context and potential implications.",
		"data: Looking at the evidence, there are several key points.",
		"data: The first point relates to the fundamental principles involved.",
		"data: The second point considers practical applications.",
		"data: I need to weigh the pros and cons of each approach.",
		"data: Additionally, there are edge cases to consider.",
		"data: The historical context provides important background.",
		"data: Recent developments have changed the landscape.",
		"data: I should also consider future implications.",
		"data: Based on my analysis, I can form a conclusion.",
		"data: The reasoning leads to a clear recommendation.",
		"data: </thinking>",
		"data: Based on my analysis, the answer is clear.",
		"data: [DONE]",
	}, "\n")
}
