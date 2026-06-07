package generate

import (
	"math"
	"testing"
)

// TestSoftmax verifies that softmax outputs sum to 1.
func TestSoftmax(t *testing.T) {
	logits := []float64{1.0, 2.0, 3.0, 4.0}
	probs := softmax(logits)
	sum := 0.0
	for _, p := range probs {
		sum += p
	}
	if math.Abs(sum-1.0) > 1e-6 {
		t.Errorf("Softmax probs sum = %f, want 1.0", sum)
	}
}

// TestTemperature verifies that lower temperature concentrates the distribution.
func TestTemperature(t *testing.T) {
	logits := []float64{1.0, 2.0, 3.0, 10.0}

	// High temperature → flatter distribution
	highTemp := make([]float64, len(logits))
	copy(highTemp, logits)
	for i := range highTemp {
		highTemp[i] /= 2.0
	}
	highProbs := softmax(highTemp)

	// Low temperature → sharper distribution
	lowTemp := make([]float64, len(logits))
	copy(lowTemp, logits)
	for i := range lowTemp {
		lowTemp[i] /= 0.1
	}
	lowProbs := softmax(lowTemp)

	// The max probability should be higher with low temperature
	maxHigh := 0.0
	maxLow := 0.0
	for _, p := range highProbs {
		if p > maxHigh {
			maxHigh = p
		}
	}
	for _, p := range lowProbs {
		if p > maxLow {
			maxLow = p
		}
	}
	if maxLow <= maxHigh {
		t.Errorf("Low temperature should concentrate distribution: maxLow=%.4f maxHigh=%.4f", maxLow, maxHigh)
	}
}

// TestRepetitionPenalty verifies that repeated token probabilities are reduced.
func TestRepetitionPenalty(t *testing.T) {
	// Set up a sampler with high repetition penalty
	cfg := DefaultConfig()
	cfg.RepetitionPenalty = 1.5
	cfg.Temperature = 1.0
	cfg.TopK = 0
	cfg.TopP = 1.0

	// Logits where token 3 is high
	logits := []float64{0, 0, 0, 5.0, 0, 0, 0, 0, 0, 0}
	context := []int{3} // token 3 already in context

	// Apply repetition penalty manually
	penalized := make([]float64, len(logits))
	copy(penalized, logits)
	for _, id := range context {
		if id < len(penalized) {
			if penalized[id] > 0 {
				penalized[id] /= cfg.RepetitionPenalty
			} else {
				penalized[id] *= cfg.RepetitionPenalty
			}
		}
	}

	probsBefore := softmax(logits)
	probsAfter := softmax(penalized)

	if probsAfter[3] >= probsBefore[3] {
		t.Errorf("Repetition penalty did not reduce prob of token 3: before=%.4f after=%.4f",
			probsBefore[3], probsAfter[3])
	}
}

// TestNucleusNormalization verifies that top-p filtered probabilities still sum to ~1.
func TestNucleusNormalization(t *testing.T) {
	logits := []float64{1, 2, 3, 4, 5}
	probs := softmax(logits)

	// Apply top-p filtering at 0.9
	topP := 0.9
	indices := make([]int, len(probs))
	for i := range indices {
		indices[i] = i
	}
	// Sort descending
	for i := range indices {
		for j := i + 1; j < len(indices); j++ {
			if probs[indices[j]] > probs[indices[i]] {
				indices[i], indices[j] = indices[j], indices[i]
			}
		}
	}

	cumsum := 0.0
	cutoff := -1
	for _, idx := range indices {
		cumsum += probs[idx]
		if cumsum >= topP {
			cutoff = idx
			break
		}
	}
	if cutoff >= 0 {
		threshold := probs[cutoff]
		for i, p := range probs {
			if p < threshold {
				probs[i] = 0
			}
		}
		sum := 0.0
		for _, p := range probs {
			sum += p
		}
		if sum > 0 {
			for i := range probs {
				probs[i] /= sum
			}
		}
		// Verify normalisation
		total := 0.0
		for _, p := range probs {
			total += p
		}
		if math.Abs(total-1.0) > 1e-6 {
			t.Errorf("Top-p filtered probs sum = %f, want 1.0", total)
		}
	}
}

// TestHashString verifies deterministic hashing.
func TestHashString(t *testing.T) {
	h1 := hashString("ciao")
	h2 := hashString("ciao")
	if h1 != h2 {
		t.Error("hashString is not deterministic")
	}
	h3 := hashString("hello")
	if h1 == h3 {
		t.Error("hashString should differ for different inputs")
	}
}

// TestThinkingDelay verifies that longer responses produce longer delays.
func TestThinkingDelay(t *testing.T) {
	d1 := ThinkingDelay(5)
	d2 := ThinkingDelay(100)
	if d2 <= d1 {
		t.Errorf("Longer response should have longer delay: d1=%v d2=%v", d1, d2)
	}
}
