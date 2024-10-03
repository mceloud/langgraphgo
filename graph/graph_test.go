package graph_test

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/mceloud/langgraphgo/graph"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/openai"
)

func ExampleMessageGraph() {
	model, err := openai.New()
	if err != nil {
		panic(err)
	}

	g := graph.NewMessageGraph()

	g.AddNode("oracle", func(ctx context.Context, state interface{}) (interface{}, error) {
		r, err := model.GenerateContent(ctx, state.([]llms.MessageContent), llms.WithTemperature(0.0))
		if err != nil {
			return nil, err
		}
		return append(state.([]llms.MessageContent),
			llms.TextParts(llms.ChatMessageTypeAI, r.Choices[0].Content),
		), nil
	})
	g.AddNode(graph.END, func(_ context.Context, state interface{}) (interface{}, error) {
		return state, nil
	})

	g.AddEdge("oracle", graph.END)
	g.SetEntryPoint("oracle")

	runnable, err := g.Compile()
	if err != nil {
		panic(err)
	}

	ctx := context.Background()
	// Let's run it!
	res, err := runnable.Invoke(ctx, []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeHuman, "What is 1 + 1?"),
	})
	if err != nil {
		panic(err)
	}

	fmt.Println(res)

	// Output:
	// [{human [{What is 1 + 1?}]} {ai [{1 + 1 equals 2.}]}]
}

//nolint:funlen,gocognit,cyclop
func TestMessageGraph(t *testing.T) {
	t.Parallel()
	testCases := []struct {
		name           string
		buildGraph     func() *graph.MessageGraph
		inputMessages  []llms.MessageContent
		expectedOutput []llms.MessageContent
		expectedError  error
	}{
		{
			name: "Simple graph",
			buildGraph: func() *graph.MessageGraph {
				g := graph.NewMessageGraph()
				g.AddNode("node1", func(_ context.Context, state interface{}) (interface{}, error) {
					return append(state.([]llms.MessageContent), llms.TextParts(llms.ChatMessageTypeAI, "Node 1")), nil
				})
				g.AddNode("node2", func(_ context.Context, state interface{}) (interface{}, error) {
					return append(state.([]llms.MessageContent), llms.TextParts(llms.ChatMessageTypeAI, "Node 2")), nil
				})
				g.AddEdge("node1", "node2")
				g.AddEdge("node2", graph.END)
				g.SetEntryPoint("node1")
				return g
			},
			inputMessages: []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "Input")},
			expectedOutput: []llms.MessageContent{
				llms.TextParts(llms.ChatMessageTypeHuman, "Input"),
				llms.TextParts(llms.ChatMessageTypeAI, "Node 1"),
				llms.TextParts(llms.ChatMessageTypeAI, "Node 2"),
			},
			expectedError: nil,
		},
		{
			name: "Entry point not set",
			buildGraph: func() *graph.MessageGraph {
				g := graph.NewMessageGraph()
				g.AddNode("node1", func(_ context.Context, state interface{}) (interface{}, error) {
					return state, nil
				})
				return g
			},
			expectedError: graph.ErrEntryPointNotSet,
		},
		{
			name: "Node not found",
			buildGraph: func() *graph.MessageGraph {
				g := graph.NewMessageGraph()
				g.AddNode("node1", func(_ context.Context, state interface{}) (interface{}, error) {
					return state, nil
				})
				g.AddEdge("node1", "node2")
				g.SetEntryPoint("node1")
				return g
			},
			expectedError: fmt.Errorf("%w: node2", graph.ErrNodeNotFound),
		},
		{
			name: "No outgoing edge",
			buildGraph: func() *graph.MessageGraph {
				g := graph.NewMessageGraph()
				g.AddNode("node1", func(_ context.Context, state interface{}) (interface{}, error) {
					return state, nil
				})
				g.SetEntryPoint("node1")
				return g
			},
			expectedError: fmt.Errorf("%w: node1", graph.ErrNoOutgoingEdge),
		},
		{
			name: "Error in node function",
			buildGraph: func() *graph.MessageGraph {
				g := graph.NewMessageGraph()
				g.AddNode("node1", func(_ context.Context, _ interface{}) (interface{}, error) {
					return nil, errors.New("node error")
				})
				g.AddEdge("node1", graph.END)
				g.SetEntryPoint("node1")
				return g
			},
			expectedError: errors.New("error in node node1: node error"),
		},
		{
			name: "Conditional edge - condition for edge fulfilled",
			buildGraph: func() *graph.MessageGraph {
				g := graph.NewMessageGraph()
				g.AddNode("node1", func(_ context.Context, state interface{}) (interface{}, error) {
					return append(state.([]llms.MessageContent), llms.TextParts(llms.ChatMessageTypeAI, "function calling: use calculator")), nil
				})
				g.AddNode("node2", func(_ context.Context, state interface{}) (interface{}, error) {
					return append(state.([]llms.MessageContent), llms.TextParts(llms.ChatMessageTypeAI, "Node 2")), nil
				})
				g.AddNode("calculator", func(_ context.Context, state interface{}) (interface{}, error) {
					return append(state.([]llms.MessageContent), llms.TextParts(llms.ChatMessageTypeTool, "1+1=2")), nil
				})
				g.AddConditionalEdge("node1", func(_ context.Context, state interface{}) []string {
					if content, ok := state.([]llms.MessageContent)[len(state.([]llms.MessageContent))-1].Parts[0].(llms.TextContent); ok {
						if strings.Contains(content.Text, "calculator") {
							return []string{"calculator"}
						}
					}
					return []string{"node2"}
				})
				g.AddEdge("node2", graph.END)
				g.AddEdge("calculator", graph.END)
				g.SetEntryPoint("node1")
				return g
			},
			inputMessages: []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "what is 1+1?")},
			expectedOutput: []llms.MessageContent{
				llms.TextParts(llms.ChatMessageTypeHuman, "what is 1+1?"),
				llms.TextParts(llms.ChatMessageTypeAI, "function calling: use calculator"),
				llms.TextParts(llms.ChatMessageTypeTool, "1+1=2"),
			},
			expectedError: nil,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			g := tc.buildGraph()
			runnable, err := g.Compile()
			if err != nil {
				if tc.expectedError == nil || !errors.Is(err, tc.expectedError) {
					t.Fatalf("unexpected compile error: %v", err)
				}
				return
			}

			output, err := runnable.Invoke(context.Background(), tc.inputMessages)
			if err != nil {
				if tc.expectedError == nil || err.Error() != tc.expectedError.Error() {
					t.Fatalf("unexpected invoke error: '%v', expected '%v'", err, tc.expectedError)
				}
				return
			}

			if tc.expectedError != nil {
				t.Fatalf("expected error %v, but got nil", tc.expectedError)
			}

			if len(output.([]llms.MessageContent)) != len(tc.expectedOutput) {
				t.Fatalf("expected output length %d, but got %d", len(tc.expectedOutput), len(output.([]llms.MessageContent)))
			}

			for i, msg := range output.([]llms.MessageContent) {
				got := fmt.Sprint(msg)
				expected := fmt.Sprint(tc.expectedOutput[i])
				if got != expected {
					t.Errorf("expected output[%d] content %q, but got %q", i, expected, got)
				}
			}
		})
	}
}
