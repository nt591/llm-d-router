/*
Copyright 2026 The llm-d Authors.

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

package internal

import (
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

func isDecodeOnly(item flowcontrol.QueueItemAccessor) bool {
	request := item.OriginalRequest().InferenceRequest()
	if request == nil {
		return false
	}
	only, _ := scheduling.ReadRequestAttribute[bool](request, flowcontrol.DecodeOnlyRequestAttributeKey)
	return only
}

// eligibleBand preserves flow identity and policy state while exposing eligible requests.
type eligibleBand struct {
	flowcontrol.PriorityBandAccessor
	predicate func(flowcontrol.QueueItemAccessor) bool
}

func (b *eligibleBand) eligible(q flowcontrol.FlowQueueAccessor) flowcontrol.FlowQueueAccessor {
	if q == nil {
		return nil
	}
	var item flowcontrol.QueueItemAccessor
	if matching, ok := q.(flowcontrol.MatchingQueue); ok {
		item = matching.PeekMatching(b.predicate)
	} else {
		item = q.Peek()
		if item != nil && !b.predicate(item) {
			item = nil
		}
	}
	if item == nil {
		return nil
	}
	return &eligibleQueue{FlowQueueAccessor: q, item: item}
}

func (b *eligibleBand) IterateQueues(callback func(flowcontrol.FlowQueueAccessor) bool) {
	b.PriorityBandAccessor.IterateQueues(func(q flowcontrol.FlowQueueAccessor) bool {
		eligible := b.eligible(q)
		return eligible == nil || callback(eligible)
	})
}

func (b *eligibleBand) FlowKeys() []flowcontrol.FlowKey {
	var keys []flowcontrol.FlowKey
	b.IterateQueues(func(q flowcontrol.FlowQueueAccessor) bool {
		keys = append(keys, q.FlowKey())
		return true
	})
	return keys
}

func (b *eligibleBand) Queue(id string) flowcontrol.FlowQueueAccessor {
	return b.eligible(b.PriorityBandAccessor.Queue(id))
}

type eligibleQueue struct {
	flowcontrol.FlowQueueAccessor
	item flowcontrol.QueueItemAccessor
}

func (q *eligibleQueue) Peek() flowcontrol.QueueItemAccessor { return q.item }
