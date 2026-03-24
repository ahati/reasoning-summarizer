package summarizer

import (
	"context"
	"strings"
	"testing"
)

func TestExtractFromStream(t *testing.T) {
	tests := []struct {
		name       string
		input      string
		wantReason bool
		wantReasoning string
		wantOutput string
	}{
		{
			name: "with thinking tags",
			input: strings.Join([]string{
				"data: <thinking>",
				"data: Step 1: Analyze the question",
				"data: Step 2: Find the answer",
				"data: </thinking>",
				"data: The answer is 42.",
				"data: [DONE]",
			}, "\n"),
			wantReason: true,
			wantReasoning: "Step 1: Analyze the questionStep 2: Find the answer",
			wantOutput: "The answer is 42.",
		},
		{
			name: "without thinking tags",
			input: strings.Join([]string{
				"data: Hello world",
				"data: [DONE]",
			}, "\n"),
			wantReason: false,
			wantReasoning: "",
			wantOutput: "Hello world",
		},
		{
			name: "empty input",
			input: "",
			wantReason: false,
			wantReasoning: "",
			wantOutput: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := strings.NewReader(tt.input)
			result, err := ExtractFromStream(context.Background(), r)
			if err != nil {
				t.Fatalf("ExtractFromStream() error = %v", err)
			}
			if result.HasReasoning != tt.wantReason {
				t.Errorf("HasReasoning = %v, want %v", result.HasReasoning, tt.wantReason)
			}
			if result.Reasoning != tt.wantReasoning {
				t.Errorf("Reasoning = %q, want %q", result.Reasoning, tt.wantReasoning)
			}
			if result.Output != tt.wantOutput {
				t.Errorf("Output = %q, want %q", result.Output, tt.wantOutput)
			}
		})
	}
}

func TestTruncateMiddle(t *testing.T) {
	tests := []struct {
		input   string
		maxLen  int
		wantLen int
	}{
		{"short", 10, 5},
		{"this is a longer string that needs truncation", 20, 20},
		{"中文字符测试中文字符测试", 6, 6}, // Unicode test
	}

	for _, tt := range tests {
		got := truncateMiddle(tt.input, tt.maxLen)
		if len([]rune(got)) != tt.wantLen {
			t.Errorf("truncateMiddle(%q, %d) length = %d, want %d", 
				tt.input, tt.maxLen, len([]rune(got)), tt.wantLen)
		}
	}
}

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig("/path/to/model.gguf")
	if cfg.ModelPath != "/path/to/model.gguf" {
		t.Errorf("ModelPath = %q, want /path/to/model.gguf", cfg.ModelPath)
	}
	if cfg.ContextSize != 2048 {
		t.Errorf("ContextSize = %d, want 2048", cfg.ContextSize)
	}
	if cfg.MaxSummaryTokens != 256 {
		t.Errorf("MaxSummaryTokens = %d, want 256", cfg.MaxSummaryTokens)
	}
}
