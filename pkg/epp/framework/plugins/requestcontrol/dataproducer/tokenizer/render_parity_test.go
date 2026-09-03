/*
Copyright 2025 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package tokenizer

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/google/go-cmp/cmp"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/openai"
)

// TestChatPayloadFidelity asserts that the render payload for an OpenAI chat body
// carries every field of the original request, including fields the typed
// ChatCompletionsRequest does not model (reasoning, tool_call_id, audio/video
// blocks, vendor extensions). The render backend forwards this payload to the
// tokenizer, so any dropped field skews token counts against the body the serving
// pod receives. Each body includes such fields; the payload must equal the input.
func TestChatPayloadFidelity(t *testing.T) {
	bodies := map[string]string{
		"reasoning and tool_call_id": `{
			"model": "glm-5-2",
			"messages": [
				{"role": "user", "content": "hi"},
				{"role": "assistant", "content": "", "reasoning": "let me think", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
				{"role": "tool", "tool_call_id": "c1", "content": "42"}
			]
		}`,
		"audio and video blocks": `{
			"model": "glm-5-2",
			"messages": [
				{"role": "user", "content": [
					{"type": "text", "text": "describe"},
					{"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
					{"type": "video_url", "video_url": {"url": "http://example.com/v.mp4"}}
				]}
			]
		}`,
		"vendor extension fields": `{
			"model": "glm-5-2",
			"messages": [{"role": "user", "content": "hi"}],
			"reasoning_effort": "high",
			"x_vendor_option": {"nested": true}
		}`,
	}

	parser := openai.NewOpenAIParser()
	headers := map[string]string{":path": "/v1/chat/completions"}

	for name, body := range bodies {
		t.Run(name, func(t *testing.T) {
			raw := []byte(body)
			res, err := parser.ParseRequest(context.Background(), raw, headers)
			if err != nil {
				t.Fatalf("ParseRequest: %v", err)
			}

			pm, ok := chatPayload(res.Body).AsMap()
			if !ok {
				t.Fatalf("chatPayload did not yield a map")
			}
			var want fwkrh.PayloadMap
			if err := json.Unmarshal(raw, &want); err != nil {
				t.Fatalf("unmarshal want: %v", err)
			}
			if diff := cmp.Diff(want, pm); diff != "" {
				t.Errorf("render payload dropped fields (-want +got):\n%s", diff)
			}
		})
	}
}
