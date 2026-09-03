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
	"fmt"
	"runtime"
	"strings"
	"testing"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/openai"
)

func benchChatBody(approxBytes int) []byte {
	content := strings.Repeat("The quick brown fox jumps over the lazy dog. ", 8)
	n := approxBytes / (len(content) + 40)
	if n < 1 {
		n = 1
	}
	msgs := make([]map[string]any, n)
	for i := range msgs {
		role := "user"
		if i%2 == 1 {
			role = "assistant"
		}
		msgs[i] = map[string]any{"role": role, "content": content}
	}
	data, err := json.Marshal(map[string]any{"model": "glm-5-2", "messages": msgs})
	if err != nil {
		panic(err)
	}
	return data
}

// BenchmarkChatDecodePath quantifies the full decode cost of the chat prefix-cache
// path, separating the two phases the parser change relocates work between:
//
//	parse_only          ParseRequest, the pre-admission phase (single typed scan).
//	parse_plus_render   ParseRequest + chatPayload, adding the full-map decode the
//	                    render backend performs at tokenize-time (post-admission).
//
// The render-backend total is roughly the sum of both; the win over the prior
// design is that the map decode no longer runs on the pre-queue admission path.
// The estimate backend never calls chatPayload, so parse_only is its whole cost.
func BenchmarkChatDecodePath(b *testing.B) {
	body := benchChatBody(800 << 10)
	parser := openai.NewOpenAIParser()
	headers := map[string]string{":path": "/v1/chat/completions"}

	b.Run("parse_only", func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(int64(len(body)))
		for i := 0; i < b.N; i++ {
			res, err := parser.ParseRequest(context.Background(), body, headers)
			if err != nil {
				b.Fatal(err)
			}
			runtime.KeepAlive(res)
		}
	})

	b.Run("parse_plus_render", func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(int64(len(body)))
		for i := 0; i < b.N; i++ {
			res, err := parser.ParseRequest(context.Background(), body, headers)
			if err != nil {
				b.Fatal(err)
			}
			runtime.KeepAlive(chatPayload(res.Body))
		}
	})
}

// BenchmarkAdmissionDecodeWork measures aggregate decode work across a batch of
// requests at varying shed fractions, contrasting the two designs:
//
//	eager     the full map decode runs at parse for every request, before
//	          admission -- shed requests pay it too.
//	deferred  parse is cheap for every request; the map decode runs only for
//	          admitted requests, after the queue.
//
// It models the queue outcome (shed fraction), not real flow-control timing, so
// the numbers are deterministic. Each op processes one batch of requests. At
// shed=0 the designs are identical; the gap widens with the shed fraction, which
// is exactly the overload regime.
func BenchmarkAdmissionDecodeWork(b *testing.B) {
	body := benchChatBody(800 << 10)
	parser := openai.NewOpenAIParser()
	headers := map[string]string{":path": "/v1/chat/completions"}
	const batch = 100

	for _, shed := range []int{0, 50, 90} {
		admitted := func(i int) bool { return i%100 >= shed }

		b.Run(fmt.Sprintf("shed=%d/eager", shed), func(b *testing.B) {
			b.ReportAllocs()
			for n := 0; n < b.N; n++ {
				for i := 0; i < batch; i++ {
					res, err := parser.ParseRequest(context.Background(), body, headers)
					if err != nil {
						b.Fatal(err)
					}
					runtime.KeepAlive(chatPayload(res.Body))
				}
			}
		})

		b.Run(fmt.Sprintf("shed=%d/deferred", shed), func(b *testing.B) {
			b.ReportAllocs()
			for n := 0; n < b.N; n++ {
				for i := 0; i < batch; i++ {
					res, err := parser.ParseRequest(context.Background(), body, headers)
					if err != nil {
						b.Fatal(err)
					}
					if admitted(i) {
						runtime.KeepAlive(chatPayload(res.Body))
					} else {
						runtime.KeepAlive(res)
					}
				}
			}
		})
	}
}
