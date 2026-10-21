# RFC: Decode-only admission under prefill saturation

## Problem

The combined admission gate uses the greater of prefill and decode saturation for every request. It cannot tell whether a request requires remote prefill. If prefill saturation is 1.5, decode saturation is 0.3, and the priority ceiling is 1.0, the combined gate blocks requests that could run entirely on decode.

## Admission and dispatch

1. Run token producers that support execution before admission. They receive no endpoint candidates.
2. Ask the decode-first disaggregation handler whether the request can run without remote prefill. It returns true when no prefill decider is configured, the prefix-based decider is disabled, or the prompt is below a configured token threshold. Cache-dependent decisions remain unclassified.
3. Enqueue the request in its existing flow queue with the classification attached.
4. If the combined gate closes the band but decode is below its ceiling, expose only classified requests to the existing fairness policy. The policy selects a flow, then dispatch uses that flow's first eligible request under its ordering policy.

In the example above, a classified short request can dispatch while a prefill-dependent request in the same flow remains queued. Decode saturation at or above the ceiling blocks both requests.

Each flow keeps its identity, queue capacity accounting, ordering policy, and TTL. Eligible requests can pass blocked work; ordering still applies among eligible requests. Matching selection scans the queue under a read lock and costs O(n) for a queue with n requests. Custom queues without matching selection can bypass only when their first request is eligible.

## Producer contract

Early producers may modify only `Body.TokenizedRequest`. Other body fields and token slices are shared and must remain read-only. The director gives each invocation a separate body and token struct, and retains its token result only after successful completion. A producer that finishes after its timeout cannot replace the queued body's token data. Scheduling reuses successful token data.

Token production happens before enqueue, so its execution time is outside the queue TTL. Render-backed tokenizers can issue requests to their configured service during this phase. The request context and producer timeout bound those calls.

## Policy limits

`UsageLimitPolicy` receives combined saturation once per dispatch cycle. Saturation-dependent policies can still close lower-priority bands because of prefill load. The default static policy permits the bypass described here. Evaluating priority policies separately for each stage requires a contract that produces both views without advancing dispatch pacing state twice.

Dispatch reservations remain pool-wide. Pending decode-only dispatches can briefly contribute to prefill saturation until `PreRequest` publishes their endpoint accounting.

## Scope

This draft covers prompt-threshold decisions. Long prompts with a short uncached suffix require decode selection before classification and revalidation after queue wait. Missing token data, prefill-first configurations, configurations with an encode decider, and unsupported prefill deciders retain the combined gate. A disabled or absent prefill decider needs no token data to prove that remote prefill is unnecessary.

No API configuration or dependency changes are required.

## Validation

Tests verify that a short request passes queued prefill-dependent work through the real flow controller and registry, and that blocked work dispatches when capacity returns. They also cover saturated decode capacity, priority ceilings, threshold boundaries, classification before enqueue, late token writes after a producer timeout, and matching selection that preserves queue accounting and handles.
