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
	"testing"

	"github.com/google/go-cmp/cmp"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

// maxParseAllocs caps allocations for parsing a large chat body. A failure is a
// regression to investigate (e.g. a second full decode reintroduced), not a
// number to raise.
const maxParseAllocs = 9000

// TestParseRequestAllocs guards the single-scan decode: parsing a large
// long-context body must not reintroduce the per-scalar boxing of a full
// map[string]any decode.
func TestParseRequestAllocs(t *testing.T) {
	parser := NewOpenAIParser()
	headers := map[string]string{":path": "/v1/chat/completions"}
	body := chatBody(800 << 10)

	avg := testing.AllocsPerRun(20, func() {
		res, err := parser.ParseRequest(context.Background(), body, headers)
		if err != nil {
			t.Fatal(err)
		}
		benchSink = res
	})
	if avg > maxParseAllocs {
		t.Errorf("ParseRequest allocations regressed: got %.0f, ceiling %d", avg, maxParseAllocs)
	}
}

// TestRewriteModelNameRaw checks that rewriting the model on a RawJSONPayload
// replaces only the model and preserves every other field of the forwarded body.
func TestRewriteModelNameRaw(t *testing.T) {
	parser := NewOpenAIParser()
	headers := map[string]string{":path": "/v1/chat/completions"}
	body := []byte(`{"model":"old","stream":true,"temperature":0.7,"messages":[{"role":"user","content":"hi"}]}`)

	res, err := parser.ParseRequest(context.Background(), body, headers)
	if err != nil {
		t.Fatalf("ParseRequest: %v", err)
	}
	payload, ok := res.Body.Payload.(fwkrh.MarshalablePayload)
	if !ok {
		t.Fatalf("payload %T is not marshalable", res.Body.Payload)
	}

	rewritten, err := parser.RewriteModelName(payload, "new-model")
	if err != nil {
		t.Fatalf("RewriteModelName: %v", err)
	}
	out, err := rewritten.Marshal()
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}

	var got map[string]any
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatalf("unmarshal rewritten: %v", err)
	}
	var want map[string]any
	if err := json.Unmarshal(body, &want); err != nil {
		t.Fatalf("unmarshal original: %v", err)
	}
	want["model"] = "new-model"
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("rewritten body mismatch (-want +got):\n%s", diff)
	}
}

// TestRewriteModelNamePayloadMap covers the branch taken after a plugin has
// mutated the body into a PayloadMap (see InferenceRequestBody.MutatePayloadMap).
func TestRewriteModelNamePayloadMap(t *testing.T) {
	parser := NewOpenAIParser()
	pm := fwkrh.PayloadMap{"model": "old", "prompt": "hi"}

	rewritten, err := parser.RewriteModelName(pm, "new-model")
	if err != nil {
		t.Fatalf("RewriteModelName: %v", err)
	}
	got, ok := rewritten.(fwkrh.PayloadMap)
	if !ok {
		t.Fatalf("expected PayloadMap, got %T", rewritten)
	}
	if got["model"] != "new-model" {
		t.Errorf("model = %v, want new-model", got["model"])
	}
	if got["prompt"] != "hi" {
		t.Errorf("prompt = %v, want hi", got["prompt"])
	}
}
