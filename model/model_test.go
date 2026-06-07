package model

import (
	"math"
	"testing"
)

// TestTensorMatMul verifies a 2×3 @ 3×2 → 2×2 matrix product.
func TestTensorMatMul(t *testing.T) {
	a := NewTensorFrom([]float64{1, 2, 3, 4, 5, 6}, 2, 3, false)
	b := NewTensorFrom([]float64{7, 8, 9, 10, 11, 12}, 3, 2, false)
	c := MatMul(a, b)
	// Expected: [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
	//           = [[58, 64], [139, 154]]
	want := []float64{58, 64, 139, 154}
	for i, v := range c.Data {
		if math.Abs(v-want[i]) > 1e-9 {
			t.Errorf("MatMul[%d]: got %f want %f", i, v, want[i])
		}
	}
}

// TestBackwardAdd verifies gradients for element-wise addition via a scalar sum loss.
func TestBackwardAdd(t *testing.T) {
	a := NewTensor(2, 2, true)
	b := NewTensor(2, 2, true)
	for i := range a.Data {
		a.Data[i] = float64(i + 1)
		b.Data[i] = float64(i + 2)
	}
	c := Add(a, b)
	loss := SumAll(c) // reduce to scalar so Backward seeds loss.Grad[0]=1
	loss.Backward()
	// dL/da = dL/dc * 1 = 1 for all elements
	for i, g := range a.Grad {
		if math.Abs(g-1.0) > 1e-9 {
			t.Errorf("a.Grad[%d] = %f, want 1.0", i, g)
		}
	}
	for i, g := range b.Grad {
		if math.Abs(g-1.0) > 1e-9 {
			t.Errorf("b.Grad[%d] = %f, want 1.0", i, g)
		}
	}
}

// TestBackwardMatMulNumerical verifies MatMul gradients against finite differences.
func TestBackwardMatMulNumerical(t *testing.T) {
	SetSeed(42)
	rows, inner, cols := 3, 4, 2

	a := NewTensor(rows, inner, true)
	b := NewTensor(inner, cols, true)
	for i := range a.Data {
		a.Data[i] = randNorm() * 0.1
	}
	for i := range b.Data {
		b.Data[i] = randNorm() * 0.1
	}

	// Compute analytical gradients
	c := MatMul(a, b)
	// Sum all output elements as scalar loss
	loss := NewTensor(1, 1, true)
	loss.Data[0] = 0
	for _, v := range c.Data {
		loss.Data[0] += v
	}
	loss.children = []*Tensor{c}
	loss.backwardFn = func() {
		for i := range c.Grad {
			c.Grad[i] += loss.Grad[0]
		}
	}
	loss.Backward()

	// Numerical gradient check for a[0]
	eps := 1e-4
	for idx := range a.Data {
		orig := a.Data[idx]
		a.Data[idx] = orig + eps
		cPlus := MatMul(a, b)
		lossPlus := sumTensor(cPlus)
		a.Data[idx] = orig - eps
		cMinus := MatMul(a, b)
		lossMinus := sumTensor(cMinus)
		a.Data[idx] = orig
		numGrad := (lossPlus - lossMinus) / (2 * eps)
		analGrad := a.Grad[idx]
		rel := math.Abs(numGrad-analGrad) / (math.Abs(analGrad) + 1e-8)
		if rel > 1e-3 {
			t.Errorf("a.Grad[%d]: numerical=%.6f analytical=%.6f relative_err=%.6f", idx, numGrad, analGrad, rel)
		}
	}
}

// TestGELUGrad verifies GELU gradients numerically by using a scalar sum loss.
func TestGELUGrad(t *testing.T) {
	xs := []float64{-1.0, 0.0, 0.5, 1.0, 2.0}
	eps := 1e-5
	for _, x := range xs {
		a := NewTensorFrom([]float64{x}, 1, 1, true)
		out := GELU(a)
		out.Backward() // scalar output, seeds out.Grad[0]=1
		numGrad := (geluVal(x+eps) - geluVal(x-eps)) / (2 * eps)
		rel := math.Abs(numGrad-a.Grad[0]) / (math.Abs(a.Grad[0]) + 1e-8)
		if rel > 1e-4 {
			t.Errorf("GELU grad at x=%.2f: numerical=%.6f analytical=%.6f", x, numGrad, a.Grad[0])
		}
	}
}

// TestAttentionShape verifies that MultiHeadAttention output has shape [seqLen × dModel].
func TestAttentionShape(t *testing.T) {
	SetSeed(1)
	seqLen, dModel, nHeads := 8, 16, 2
	attn := NewMultiHeadAttention(dModel, nHeads)
	x := NewTensor(seqLen, dModel, false)
	for i := range x.Data {
		x.Data[i] = globalRand() * 0.1
	}
	out := attn.Forward(x)
	if out.Rows != seqLen || out.Cols != dModel {
		t.Errorf("Attention output shape: got [%d×%d], want [%d×%d]", out.Rows, out.Cols, seqLen, dModel)
	}
}

// TestCausalMask verifies that future positions are masked to near-zero attention weight.
func TestCausalMask(t *testing.T) {
	seqLen := 4
	mask := buildCausalMask(seqLen)
	// Position (0,1), (0,2), (0,3) must be masked (true); (0,0) must not.
	if mask[0*seqLen+0] {
		t.Error("Diagonal should NOT be masked")
	}
	if !mask[0*seqLen+1] {
		t.Error("Future position (0,1) should be masked")
	}
	if !mask[0*seqLen+2] {
		t.Error("Future position (0,2) should be masked")
	}
	if mask[1*seqLen+0] {
		t.Error("Past position (1,0) should NOT be masked")
	}
}

// TestLayerNormInvariance checks that LayerNorm output has ~zero mean and unit variance (before scale/shift).
func TestLayerNormInvariance(t *testing.T) {
	SetSeed(7)
	rows, cols := 4, 16
	x := NewTensor(rows, cols, false)
	for i := range x.Data {
		x.Data[i] = randNorm()
	}
	gamma := NewTensor(1, cols, false)
	beta := NewTensor(1, cols, false)
	for i := range gamma.Data {
		gamma.Data[i] = 1.0
	}

	out := LayerNormForward(x, gamma, beta, 1e-5)

	for r := range rows {
		row := out.Data[r*cols : r*cols+cols]
		mean := 0.0
		for _, v := range row {
			mean += v
		}
		mean /= float64(cols)
		if math.Abs(mean) > 1e-6 {
			t.Errorf("Row %d mean = %e, want ~0", r, mean)
		}
		variance := 0.0
		for _, v := range row {
			d := v - mean
			variance += d * d
		}
		variance /= float64(cols)
		if math.Abs(variance-1.0) > 1e-4 {
			t.Errorf("Row %d variance = %f, want ~1", r, variance)
		}
	}
}

// TestForwardPass checks that the full transformer forward pass produces logits with the right shape.
func TestForwardPass(t *testing.T) {
	SetSeed(42)
	cfg := &TransformerConfig{
		VocabSize: 100,
		DModel:    32,
		NHeads:    2,
		NLayers:   2,
		DFF:       64,
		MaxSeqLen: 16,
	}
	m := NewTransformerModel(cfg)
	tokens := []int{1, 5, 10, 15, 2}
	logits := m.Forward(tokens)
	if logits.Rows != len(tokens) {
		t.Errorf("logits rows = %d, want %d", logits.Rows, len(tokens))
	}
	if logits.Cols != cfg.VocabSize {
		t.Errorf("logits cols = %d, want %d", logits.Cols, cfg.VocabSize)
	}
}

// BenchmarkForward measures forward pass throughput.
func BenchmarkForward(b *testing.B) {
	SetSeed(1)
	cfg := &TransformerConfig{
		VocabSize: 1000,
		DModel:    64,
		NHeads:    4,
		NLayers:   2,
		DFF:       128,
		MaxSeqLen: 32,
	}
	m := NewTransformerModel(cfg)
	tokens := make([]int, 16)
	for i := range tokens {
		tokens[i] = i + 1
	}
	b.ResetTimer()
	for range b.N {
		_ = m.Forward(tokens)
	}
}

// sumTensor sums all elements (helper for numerical gradient checks).
func sumTensor(t *Tensor) float64 {
	sum := 0.0
	for _, v := range t.Data {
		sum += v
	}
	return sum
}
