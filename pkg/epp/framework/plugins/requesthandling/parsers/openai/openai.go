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
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"

	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

const (
	OpenAIParserType = "openai-parser"

	conversationsAPI   = "conversations"
	responsesAPI       = "responses"
	chatCompletionsAPI = "chat/completions"
	completionsAPI     = "completions"
	embeddingsAPI      = "embeddings"
	// imagesGenerationsAPI is the OpenAI-compatible image generation endpoint/
	imagesGenerationsAPI = "images/generations"

	streamingRespPrefix = "data: "
	streamingEndMsg     = "data: [DONE]"

	contentType = "content-type"
	// The base media type for Server-Sent Events. We check for this substring
	// to account for optional parameters like "; charset=utf-8" often appended by proxies.
	eventStreamType = "text/event-stream"

	promptTokensField        = "prompt_tokens"
	inputTokensField         = "input_tokens"
	completionTokensField    = "completion_tokens"
	outputTokensField        = "output_tokens"
	promptTokensDetailsField = "prompt_tokens_details"
	inputTokensDetailsField  = "input_tokens_details"
	cachedTokensField        = "cached_tokens"
	totalTokensField         = "total_tokens"
)

// compile-time type validation
var (
	_ fwkrh.Parser            = &OpenAIParser{}
	_ fwkrh.ModelNameRewriter = &OpenAIParser{}
)

// OpenAIParser implements the fwkrh.Parser interface for OpenAI API
// https://developers.openai.com/api/reference/overview
type OpenAIParser struct {
	typedName fwkplugin.TypedName
}

// NewOpenAIParser creates a new OpenAIParser.
func NewOpenAIParser() *OpenAIParser {
	return &OpenAIParser{
		typedName: fwkplugin.TypedName{
			Type: OpenAIParserType,
			Name: OpenAIParserType,
		},
	}
}

// TypedName returns the type and name tuple of this plugin instance.
func (p *OpenAIParser) TypedName() fwkplugin.TypedName {
	return p.typedName
}

func (p *OpenAIParser) Claims() fwkrh.Claims {
	return fwkrh.Claims{
		Paths: []string{
			chatCompletionsAPI,
			completionsAPI,
			embeddingsAPI,
			responsesAPI,
			conversationsAPI,
			chatCompletionsAPI + "/render",
			completionsAPI + "/render",
			imagesGenerationsAPI,
		},
		Protocols: []v1.AppProtocol{v1.AppProtocolH2C, v1.AppProtocolHTTP},
	}
}

func OpenAIParserPluginFactory(name string, _ *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	return NewOpenAIParser().WithName(name), nil
}

func (p *OpenAIParser) WithName(name string) *OpenAIParser {
	p.typedName.Name = name
	return p
}

// ParseRequest decodes the body once into the typed body and forwards the
// original bytes as the payload, so the payload is never materialized as a map.
func (p *OpenAIParser) ParseRequest(ctx context.Context, body []byte, headers map[string]string) (*fwkrh.ParseResult, error) {
	apiType := determineAPITypeFromPath(request.GetRequestPath(headers))
	extractedBody, err := extractRequestBody(apiType, body)
	if err != nil {
		return nil, err
	}
	extractedBody.Payload = fwkrh.RawJSONPayload(body)
	return &fwkrh.ParseResult{Body: extractedBody, SkipResponseProcessing: false}, nil
}

// RewriteModelName writes the resolved model into the payload. A RawJSONPayload
// is shallow-decoded (top-level keys only) to avoid materializing nested content;
// a PayloadMap, set once a plugin mutates the body, is edited in place.
func (p *OpenAIParser) RewriteModelName(payload fwkrh.MarshalablePayload, model string) (fwkrh.MarshalablePayload, error) {
	switch v := payload.(type) {
	case fwkrh.PayloadMap:
		v["model"] = model
		return v, nil
	case fwkrh.RawJSONPayload:
		top := make(map[string]json.RawMessage)
		if err := json.Unmarshal(v, &top); err != nil {
			return nil, fmt.Errorf("rewriting model name: %w", err)
		}
		encoded, err := json.Marshal(model)
		if err != nil {
			return nil, err
		}
		top["model"] = encoded
		out, err := json.Marshal(top)
		if err != nil {
			return nil, err
		}
		return fwkrh.RawJSONPayload(out), nil
	default:
		return payload, nil
	}
}

// commonScalars holds the top-level scalars shared across OpenAI APIs. They are
// kept as raw bytes and coerced leniently so a malformed value reads as absent
// instead of failing the whole parse.
type commonScalars struct {
	Model  json.RawMessage `json:"model"`
	Stream json.RawMessage `json:"stream"`
}

func (s commonScalars) model() string {
	var v string
	if len(s.Model) == 0 || json.Unmarshal(s.Model, &v) != nil {
		return ""
	}
	return v
}

func (s commonScalars) stream() bool {
	var v bool
	if len(s.Stream) == 0 || json.Unmarshal(s.Stream, &v) != nil {
		return false
	}
	return v
}

// maxOutputTokens returns the first raw value that is a non-negative whole
// number; precedence follows argument order. Absent, null, and malformed values
// are skipped. Decoding into *float64 leaves null as nil rather than zero.
func maxOutputTokens(raws ...json.RawMessage) *int64 {
	for _, raw := range raws {
		if len(raw) == 0 {
			continue
		}
		var f *float64
		if json.Unmarshal(raw, &f) != nil || f == nil {
			continue
		}
		if *f < 0 || *f != math.Trunc(*f) {
			continue
		}
		out := int64(*f)
		return &out
	}
	return nil
}

// ParseResponse extracts usage metadata from the provider's response.
// It automatically detects and handles both standard JSON responses and SSE streams.
func (p *OpenAIParser) ParseResponse(ctx context.Context, body []byte, headers map[string]string, _ bool) (*fwkrh.ParsedResponse, error) {
	if len(body) == 0 {
		// An empty body can occur during streaming; for instance, Envoy proxies
		// may emit a trailing empty body with the EndOfStream flag set to true.
		return nil, nil //nolint:nilnil
	}

	isStream := false
	for k, v := range headers {
		if strings.ToLower(k) == contentType && strings.Contains(strings.ToLower(v), eventStreamType) {
			isStream = true
			break
		}
	}
	if isStream {
		return p.parseStreamResponse(body)
	}

	usage, err := extractUsage(body)
	if err != nil {
		return nil, err
	}
	return &fwkrh.ParsedResponse{Usage: usage}, nil
}

func (p *OpenAIParser) parseStreamResponse(chunk []byte) (*fwkrh.ParsedResponse, error) {
	usage := extractUsageStreaming(chunk)
	return &fwkrh.ParsedResponse{
		Usage:          usage,
		StreamedEvents: countStreamEvents(chunk),
	}, nil
}

// countStreamEvents counts the SSE data events in a chunk, excluding the terminator. An event
// split across two chunks is counted with the half that carries the prefix; a split inside the
// prefix drops the event and a split inside the terminator counts it, both a one-event error.
func countStreamEvents(chunk []byte) int {
	count := 0
	for line := range bytes.SplitSeq(chunk, []byte("\n")) {
		content, ok := bytes.CutPrefix(line, []byte(streamingRespPrefix))
		if ok && !isStreamTerminator(content) {
			count++
		}
	}
	return count
}

// isStreamTerminator reports whether an SSE data payload is the [DONE] terminator, tolerating a
// trailing \r left by CRLF line splitting.
func isStreamTerminator(content []byte) bool {
	return bytes.Equal(bytes.TrimSuffix(content, []byte("\r")), []byte("[DONE]"))
}

// determineAPITypeFromPath determines the API type based on the request path.
// The suffix-based matching supports both standard OpenAI paths (e.g. /v1/chat/completions)
// and provider-specific paths (e.g. Vertex AI's /v1/projects/.../chat/completions).
// Sub-paths /render under chat-completions and completions share the parent's body schema.
func determineAPITypeFromPath(path string) string {
	if request.MatchPathSuffix(path, "/conversations") {
		return conversationsAPI
	}
	if request.MatchPathSuffix(path, "/responses") {
		return responsesAPI
	}
	if request.MatchPathSuffix(path, "/chat/completions") ||
		request.MatchPathSuffix(path, "/chat/completions/render") {
		return chatCompletionsAPI
	}
	if request.MatchPathSuffix(path, "/completions") ||
		request.MatchPathSuffix(path, "/completions/render") {
		return completionsAPI
	}
	if request.MatchPathSuffix(path, "/embeddings") {
		return embeddingsAPI
	}
	if request.MatchPathSuffix(path, "/images/generations") {
		return imagesGenerationsAPI
	}

	// Default to completions API for backward compatibility with existing clients and integration tests
	return completionsAPI
}

// extractRequestBody decodes the raw body once into the typed body for the
// already-resolved API type, folding the model/stream/max-output scalars into the
// same pass. The per-API wrapper embeds the typed request struct so its custom
// unmarshalers still run.
func extractRequestBody(apiType string, rawBody []byte) (*fwkrh.InferenceRequestBody, error) {
	switch apiType {
	case conversationsAPI:
		var req struct {
			fwkrh.ConversationsRequest
			commonScalars
		}
		if err := json.Unmarshal(rawBody, &req); err != nil || len(req.Items) == 0 {
			return nil, errors.New("invalid conversations request: must have items field")
		}
		conversations := req.ConversationsRequest
		return withScalars(&fwkrh.InferenceRequestBody{Conversations: &conversations}, req.commonScalars, nil), nil

	case responsesAPI:
		var req struct {
			fwkrh.ResponsesRequest
			commonScalars
			MaxOutput json.RawMessage `json:"max_output_tokens"`
		}
		if err := json.Unmarshal(rawBody, &req); err != nil || req.Input == nil {
			return nil, errors.New("invalid responses request: must have input field")
		}
		responses := req.ResponsesRequest
		return withScalars(&fwkrh.InferenceRequestBody{Responses: &responses}, req.commonScalars, maxOutputTokens(req.MaxOutput)), nil

	case chatCompletionsAPI:
		var req struct {
			fwkrh.ChatCompletionsRequest
			commonScalars
			MaxCompletion json.RawMessage `json:"max_completion_tokens"`
			MaxTokens     json.RawMessage `json:"max_tokens"`
		}
		if err := json.Unmarshal(rawBody, &req); err != nil || validateChatCompletionsMessages(req.Messages) != nil {
			return nil, errors.New("invalid chat completions request: must have valid messages field")
		}
		chatCompletions := req.ChatCompletionsRequest
		return withScalars(&fwkrh.InferenceRequestBody{ChatCompletions: &chatCompletions}, req.commonScalars, maxOutputTokens(req.MaxCompletion, req.MaxTokens)), nil

	case completionsAPI:
		var req struct {
			fwkrh.CompletionsRequest
			commonScalars
			MaxTokens json.RawMessage `json:"max_tokens"`
		}
		if err := json.Unmarshal(rawBody, &req); err != nil || req.Prompt.IsEmpty() {
			return nil, errors.New("invalid completions request: must have prompt field")
		}
		completions := req.CompletionsRequest
		return withScalars(&fwkrh.InferenceRequestBody{Completions: &completions}, req.commonScalars, maxOutputTokens(req.MaxTokens)), nil

	case embeddingsAPI:
		var req struct {
			fwkrh.EmbeddingsRequest
			commonScalars
		}
		if err := json.Unmarshal(rawBody, &req); err != nil || req.Input.IsEmpty() {
			return nil, errors.New("invalid embeddings request: must have input field")
		}
		embeddings := req.EmbeddingsRequest
		return withScalars(&fwkrh.InferenceRequestBody{Embeddings: &embeddings}, req.commonScalars, nil), nil

	case imagesGenerationsAPI:
		var req struct {
			fwkrh.ImagesGenerationsRequest
			commonScalars
		}
		if err := json.Unmarshal(rawBody, &req); err != nil || req.Prompt == "" {
			return nil, errors.New("invalid images generations request: must have prompt field")
		}
		images := req.ImagesGenerationsRequest
		return withScalars(&fwkrh.InferenceRequestBody{Images: &images}, req.commonScalars, nil), nil
	default:
		return nil, errors.New("unsupported API endpoint")
	}
}

// withScalars populates the derived model/stream/max-output fields on the body.
func withScalars(b *fwkrh.InferenceRequestBody, s commonScalars, maxOut *int64) *fwkrh.InferenceRequestBody {
	b.Model = s.model()
	b.Stream = s.stream()
	b.MaxOutputTokens = maxOut
	return b
}

func validateChatCompletionsMessages(messages []fwkrh.Message) error {
	if len(messages) == 0 {
		return errors.New("chat-completions request must have at least one message")
	}
	return nil
}

// toInt coerces a JSON-decoded number-ish value into an int. JSON numbers
// land as float64 after json.Unmarshal into map[string]any; some
// non-conforming providers emit strings. Anything else is ignored so that
// usage extraction stays best-effort rather than panicking.
func toInt(v any) int {
	switch n := v.(type) {
	case float64:
		return int(n)
	case json.Number:
		i, _ := n.Int64()
		return int(i)
	case string:
		if f, err := strconv.ParseFloat(n, 64); err == nil {
			return int(f)
		}
	}
	return 0
}

func extractUsage(responseBytes []byte) (*fwkrh.Usage, error) {
	var responseBody struct {
		Usage map[string]any `json:"usage"`
	}
	err := json.Unmarshal(responseBytes, &responseBody)
	if err != nil {
		return nil, err
	}
	if responseBody.Usage == nil {
		return nil, nil //nolint:nilnil
	}

	usage := fwkrh.Usage{}

	// Chat/Completions APIs use prompt_tokens. Responses/Conversations APIs use input_tokens.
	for _, inputTokens := range []string{promptTokensField, inputTokensField} {
		if v, ok := responseBody.Usage[inputTokens]; ok && v != nil {
			usage.PromptTokens = toInt(v)
			break
		}
	}

	// Chat/Completions APIs use completion_tokens. Responses/Conversations APIs use output_tokens.
	for _, outputTokens := range []string{completionTokensField, outputTokensField} {
		if v, ok := responseBody.Usage[outputTokens]; ok && v != nil {
			usage.CompletionTokens = toInt(v)
			break
		}
	}

	// Chat/Completions APIs use prompt_tokens_details. Responses/Conversations APIs use input_tokens_details.
	for _, details := range []string{promptTokensDetailsField, inputTokensDetailsField} {
		if detailsMap, ok := responseBody.Usage[details].(map[string]any); ok {
			if cachedTokens, ok := detailsMap[cachedTokensField]; ok {
				usage.PromptTokenDetails = &fwkrh.PromptTokenDetails{
					CachedTokens: toInt(cachedTokens),
				}
			}
		}
	}

	// total_tokens field name is consistent across all API types.
	if v, ok := responseBody.Usage[totalTokensField]; ok && v != nil {
		usage.TotalTokens = toInt(v)
	}

	return &usage, nil
}

// Example message if "stream_options": {"include_usage": "true"} is included in the request:
// data: {"id":"...","object":"text_completion","created":1739400043,"model":"small-segment-lora-0","choices":[],
// "usage":{"prompt_tokens":7,"total_tokens":17,"completion_tokens":10}}
//
// data: [DONE]
//
// Noticed that vLLM returns two entries in one response.
// We need to strip the `data:` prefix and next Data: [DONE] from the message to fetch response data.
//
// If include_usage is not included in the request, `data: [DONE]` is returned separately, which
// indicates end of streaming.
//
// For ResponsesAPI streaming, usage is nested in the response object:
//
//	event: response.completed
//	data: {"response":{"usage":{"input_tokens":31,..},...},"type":"response.completed"}
//
// It extracts usage from events with type="response.completed".
func extractUsageStreaming(responseBytes []byte) *fwkrh.Usage {
	var streamResponse struct {
		Usage    *fwkrh.Usage `json:"usage"`
		Response struct {
			Usage json.RawMessage `json:"usage"` // Delay JSON decoding until we know we have usage data
		} `json:"response"`
		Type string `json:"type"`
	}

	lines := bytes.SplitSeq(responseBytes, []byte("\n"))
	for line := range lines {
		content, ok := bytes.CutPrefix(line, []byte(streamingRespPrefix))
		if !ok {
			continue
		}
		// When the stream is terminated with [DONE] or there's not any usage data, skip the line
		if isStreamTerminator(content) || !bytes.Contains(content, []byte("usage")) {
			continue
		}
		if err := json.Unmarshal(content, &streamResponse); err != nil {
			continue
		}
		// Standard ChatCompletion / vLLM usage format
		if streamResponse.Usage != nil {
			return streamResponse.Usage
		}
		// Responses API streaming format
		if len(streamResponse.Response.Usage) > 0 && streamResponse.Type == "response.completed" {
			jsonBytes, _ := json.Marshal(map[string]any{
				"usage": streamResponse.Response.Usage,
			})
			if usage, err := extractUsage(jsonBytes); err == nil && usage != nil {
				return usage
			}
		}
	}
	return nil
}
