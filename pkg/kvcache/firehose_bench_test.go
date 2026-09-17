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

package kvcache_test

import (
	"fmt"
	"os"
	runtimemetrics "runtime/metrics"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
)

// BenchmarkPreciseFirehose measures match latency and GC cost on a large index
// under concurrent writes. This is the load shape of a glm-5-3-precise EPP.
//
// KVBLOCK_STRESS_KEYS sets the resident key count (default 1,000,000).
// The hot sizes 752 and 3750 bracket the prod matched-prefix length.
// Use a fixed count so setup runs once:
//
//	go test -run=^$ -bench=PreciseFirehose -benchtime=2000x -count=6 ./pkg/kvcache/
func BenchmarkPreciseFirehose(b *testing.B) {
	for _, hotKeys := range []int{752, 3750} {
		b.Run(fmt.Sprintf("hot=%d", hotKeys), func(b *testing.B) {
			benchmarkPreciseFirehose(b, hotKeys)
		})
	}
}

func benchmarkPreciseFirehose(b *testing.B, hotKeys int) {
	const numPods, writers = 96, 8
	residentKeys := firehoseResidentKeys(b)

	inner, err := kvblock.NewInMemoryIndex(&kvblock.InMemoryIndexConfig{Size: 1 << 21, PodCacheSize: 128})
	if err != nil {
		b.Fatal(err)
	}
	ctx := benchContext()

	// The foreground scores the hot prefix. Every rank holds it. Reads promote
	// it, so writes never evict it.
	hot := make([]kvblock.BlockHash, hotKeys)
	for i := range hot {
		hot[i] = kvblock.BlockHash(uint64(i) + 1)
	}
	hotPods := make([]kvblock.PodEntry, numPods)
	for p := range hotPods {
		hotPods[p] = kvblock.PodEntry{PodIdentifier: fmt.Sprintf("10.0.%d.%d:8000", p/256, p%256), DeviceTier: "gpu"}
	}
	if err := inner.Add(ctx, nil, hot, hotPods); err != nil {
		b.Fatal(err)
	}

	// Resident keys add heap for the GC to scan. They never match. Their range
	// is separate from the hot and writer ranges.
	const residentBase = uint64(10_000_000)
	batch := make([]kvblock.BlockHash, 64)
	for p := 0; p < 8; p++ {
		entry := []kvblock.PodEntry{{PodIdentifier: fmt.Sprintf("10.2.%d.%d:8000", p/256, p%256), DeviceTier: "gpu"}}
		for base := 0; base < residentKeys; base += len(batch) {
			n := min(len(batch), residentKeys-base)
			for j := 0; j < n; j++ {
				batch[j] = kvblock.BlockHash(residentBase + uint64(base+j))
			}
			if err := inner.Add(ctx, nil, batch[:n], entry); err != nil {
				b.Fatal(err)
			}
		}
	}

	idx := kvblock.NewTracedIndex(kvblock.NewInstrumentedIndex(inner))
	indexer := benchMatcher(b, idx)

	// Each writer adds a batch, evicts the last one, and clears its pod at times.
	// Writer keys sit above the hot range, so writes contend but never match.
	// KVBLOCK_WRITE_RATE is blocks/s for all writers; 0 is unbounded.
	writeRate := firehoseWriteRate(b)
	var writerInterval time.Duration
	if writeRate > 0 {
		loopsPerSec := float64(writeRate) / 64 / float64(writers)
		writerInterval = time.Duration(float64(time.Second) / loopsPerSec)
	}

	stop := make(chan struct{})
	var wg sync.WaitGroup
	for w := 0; w < writers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			pod := fmt.Sprintf("10.1.0.%d:8000", w)
			entry := []kvblock.PodEntry{{PodIdentifier: pod, DeviceTier: "gpu"}}
			cur := make([]kvblock.BlockHash, 64)
			prev := make([]kvblock.BlockHash, 64)
			havePrev := false
			var tick *time.Ticker
			if writerInterval > 0 {
				tick = time.NewTicker(writerInterval)
				defer tick.Stop()
			}
			for i := uint64(0); ; i++ {
				if tick != nil {
					select {
					case <-stop:
						return
					case <-tick.C:
					}
				} else {
					select {
					case <-stop:
						return
					default:
					}
				}
				base := (uint64(w+1) << 40) | (i * 64)
				for j := range cur {
					cur[j] = kvblock.BlockHash(base + uint64(j))
				}
				_ = idx.Add(ctx, cur, cur, entry)
				if havePrev {
					for _, k := range prev {
						_ = idx.Evict(ctx, k, kvblock.EngineKey, entry)
					}
				}
				copy(prev, cur)
				havePrev = true
				if i%256 == 255 {
					_ = idx.Clear(ctx, pod)
					havePrev = false
				}
			}
		}(w)
	}

	gcBefore := readFirehoseGCSeconds()
	cyclesBefore := readFirehoseGCCycles()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		matches, err := indexer.MatchBlockKeys(ctx, hot, nil)
		if err != nil {
			b.Fatal(err)
		}
		if len(matches) != numPods {
			b.Fatalf("matched %d pods, want %d", len(matches), numPods)
		}
	}
	b.StopTimer()
	gcAfter := readFirehoseGCSeconds()
	cyclesAfter := readFirehoseGCCycles()
	close(stop)
	wg.Wait()

	b.ReportMetric((gcAfter-gcBefore)*1e9/float64(b.N), "gc-cpu-ns/op")
	// gc-cycles shows how many collections ran. Trust gc-cpu only when it is high.
	b.ReportMetric(float64(cyclesAfter-cyclesBefore), "gc-cycles")
	if s := b.Elapsed().Seconds(); s > 0 {
		b.ReportMetric((gcAfter-gcBefore)/s*100, "gc-cpu%")
	}
	if writeRate > 0 {
		b.ReportMetric(float64(writeRate), "offered-adm/s")
	}
}

// firehoseWriteRate reads KVBLOCK_WRITE_RATE (blocks/s, 0 = unbounded).
func firehoseWriteRate(b *testing.B) int {
	b.Helper()
	v := os.Getenv("KVBLOCK_WRITE_RATE")
	if v == "" {
		return 0
	}
	rate, err := strconv.Atoi(v)
	if err != nil || rate < 0 {
		b.Fatalf("invalid KVBLOCK_WRITE_RATE %q", v)
	}
	return rate
}

func firehoseResidentKeys(b *testing.B) int {
	b.Helper()
	const def = 1_000_000
	v := os.Getenv("KVBLOCK_STRESS_KEYS")
	if v == "" {
		return def
	}
	keys, err := strconv.Atoi(v)
	if err != nil || keys <= 0 {
		b.Fatalf("invalid KVBLOCK_STRESS_KEYS %q", v)
	}
	return keys
}

func readFirehoseGCSeconds() float64 {
	samples := []runtimemetrics.Sample{{Name: "/cpu/classes/gc/total:cpu-seconds"}}
	runtimemetrics.Read(samples)
	return samples[0].Value.Float64()
}

func readFirehoseGCCycles() uint64 {
	samples := []runtimemetrics.Sample{{Name: "/gc/cycles/total:gc-cycles"}}
	runtimemetrics.Read(samples)
	return samples[0].Value.Uint64()
}
