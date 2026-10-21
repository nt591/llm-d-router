# RFC: Decode-only admission under prefill saturation

## Problem

Flow control measures prefill and decode saturation separately, then uses their maximum to gate each priority band. Admission finishes before token production and decode selection. A request that the PD decider would send directly to decode therefore waits when prefill is saturated, even if decode has capacity.

## Draft behavior

Endpoint-independent token production runs before flow-control admission. The decode-first disaggregation handler proves that remote prefill is unnecessary when the decider is disabled or the prompt is below either configured token threshold. Missing token data and decisions that depend on a selected pod's cache remain subject to both stages. Prefill-first configurations and configurations with an encode decider retain the combined gate.

When combined saturation closes a band but decode saturation is below its ceiling, selection exposes only proven decode-only requests to the existing fairness policy. Each flow retains its identity, ordering policy, queue capacity, and TTL. The matching queue operation selects the earliest eligible item under the configured ordering policy without removing blocked work. It scans the queue under a read lock, so its cost is linear in queue length.

Token production uses a separate body copy before admission. A producer that completes after its timeout cannot publish token data into the queued body. Successful token data is retained for scheduling.

## Policy limitation

UsageLimitPolicy runs once per dispatch cycle with combined saturation. Saturation-dependent policies can still close lower-priority bands because of prefill load. The draft proves bypass with the default static policy and preserves the pacing contract for other policies. Complete stage-specific priority policy evaluation requires a contract that computes both stage views in one cycle without advancing policy state twice.

Dispatch reservations remain pool-wide. Pending decode-only dispatches can briefly contribute to prefill saturation until PreRequest publishes their endpoint accounting.

## Scope

This draft covers prompt-threshold decisions. Long prompts with a short uncached suffix require a decode selection before classification and revalidation after queue wait. This draft does not classify those requests as decode-only.

No public API configuration or dependency changes are required. Custom queue implementations that do not support matching selection can bypass only when their first request is eligible.

## Validation

Regression tests cover a decode-only request behind prefill-dependent work, saturated decode capacity, a priority ceiling, token threshold boundaries, classification before enqueue, and selection that preserves queue accounting and handles. Existing tests cover the combined gate for unclassified work.
