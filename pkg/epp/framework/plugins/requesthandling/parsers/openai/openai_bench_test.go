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

package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"runtime"
	"strings"
	"testing"
)

// benchSink prevents the compiler from eliminating ParseRequest calls whose
// results are otherwise unused.
var benchSink any

// chatBody builds a valid /v1/chat/completions body whose serialized size is at
// least approxBytes, spread across many text messages so the object and string
// count scales with size the way long-context traffic does.
func chatBody(approxBytes int) []byte {
	const sentence = "The quick brown fox jumps over the lazy dog. "
	content := strings.Repeat(sentence, 8) // ~360 bytes of text per message
	perMsg := len(content) + 40            // rough per-message JSON overhead
	n := approxBytes / perMsg
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
	data, err := json.Marshal(map[string]any{
		"model":      "glm-5-2",
		"messages":   msgs,
		"max_tokens": 128,
		"stream":     false,
	})
	if err != nil {
		panic(err)
	}
	return data
}

// BenchmarkParseRequest measures ParseRequest across body sizes. The parser runs
// synchronously on the ext_proc goroutine, so allocs/op under a long-context
// surge is what drives GC pressure on the shared process.
//
// Run:
//
//	go test -run='^$' -bench=BenchmarkParseRequest -benchmem -count=10 \
//	    ./pkg/epp/framework/plugins/requesthandling/parsers/openai/ | tee bench.out
//	benchstat bench.out
func BenchmarkParseRequest(b *testing.B) {
	parser := NewOpenAIParser()
	headers := map[string]string{":path": "/v1/chat/completions"}

	for _, size := range []int{1 << 10, 100 << 10, 800 << 10} {
		body := chatBody(size)
		b.Run(fmt.Sprintf("body=%dKiB", len(body)>>10), func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(len(body)))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				res, err := parser.ParseRequest(context.Background(), body, headers)
				if err != nil {
					b.Fatal(err)
				}
				benchSink = res
			}
		})
	}
}

// BenchmarkParseRequestParallel measures concurrent parses of large bodies, the
// aggregate allocation that feeds GC assist and steals CPU from other goroutines.
//
// Run:
//
//	go test -run='^$' -bench=BenchmarkParseRequestParallel -benchmem -count=10 \
//	    ./pkg/epp/framework/plugins/requesthandling/parsers/openai/ | tee bench.out
//	benchstat bench.out
func BenchmarkParseRequestParallel(b *testing.B) {
	parser := NewOpenAIParser()
	headers := map[string]string{":path": "/v1/chat/completions"}
	body := chatBody(800 << 10)

	b.ReportAllocs()
	b.SetBytes(int64(len(body)))
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			res, err := parser.ParseRequest(context.Background(), body, headers)
			if err != nil {
				b.Fatal(err)
			}
			runtime.KeepAlive(res)
		}
	})
}
