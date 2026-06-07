package model

import "math"

// MultiHeadAttention implements causal multi-head self-attention.
type MultiHeadAttention struct {
	WQ, WK, WV *Tensor // [DModel × DModel]
	WO         *Tensor // [DModel × DModel]
	BQ, BK, BV *Tensor // [1 × DModel]
	BO         *Tensor // [1 × DModel]
	NumHeads   int
	DModel     int
	DHead      int // DModel / NumHeads
}

// NewMultiHeadAttention initialises Q, K, V, O projection weights.
func NewMultiHeadAttention(dModel, numHeads int) *MultiHeadAttention {
	dHead := dModel / numHeads
	scale := math.Sqrt(2.0 / float64(dModel))

	newW := func() *Tensor {
		w := NewTensor(dModel, dModel, true)
		for i := range w.Data {
			w.Data[i] = randNorm() * scale
		}
		return w
	}
	newB := func() *Tensor { return NewTensor(1, dModel, true) }

	return &MultiHeadAttention{
		WQ:       newW(),
		WK:       newW(),
		WV:       newW(),
		WO:       newW(),
		BQ:       newB(),
		BK:       newB(),
		BV:       newB(),
		BO:       newB(),
		NumHeads: numHeads,
		DModel:   dModel,
		DHead:    dHead,
	}
}

// buildCausalMask returns a flat bool slice [seqLen×seqLen] where true = masked (j>i).
func buildCausalMask(seqLen int) []bool {
	mask := make([]bool, seqLen*seqLen)
	for i := range seqLen {
		for j := range seqLen {
			if j > i {
				mask[i*seqLen+j] = true
			}
		}
	}
	return mask
}

// Forward computes multi-head causal self-attention.
// x is [seqLen × DModel]; returns [seqLen × DModel].
func (m *MultiHeadAttention) Forward(x *Tensor) *Tensor {
	seqLen := x.Rows

	Q := Add(MatMul(x, m.WQ), m.BQ) // [seqLen × DModel]
	K := Add(MatMul(x, m.WK), m.BK)
	V := Add(MatMul(x, m.WV), m.BV)

	causalMask := buildCausalMask(seqLen)
	scale := 1.0 / math.Sqrt(float64(m.DHead))

	headOutputs := make([]*Tensor, m.NumHeads)

	for h := range m.NumHeads {
		start := h * m.DHead
		end := start + m.DHead

		Qh := SliceCols(Q, start, end) // [seqLen × DHead]
		Kh := SliceCols(K, start, end)
		Vh := SliceCols(V, start, end)

		// Scores = Qh @ Kh^T / sqrt(DHead) — [seqLen × seqLen]
		KhT := transposeGraph(Kh) // [DHead × seqLen]
		scores := MatMul(Qh, KhT) // [seqLen × seqLen]
		scores = Scale(scores, scale)

		// Apply causal mask
		scores = MaskFill(scores, causalMask, -1e9)

		attn := SoftmaxRows(scores) // [seqLen × seqLen]

		headOut := MatMul(attn, Vh) // [seqLen × DHead]
		headOutputs[h] = headOut
	}

	// Concatenate heads: [seqLen × DModel]
	concat := ConcatCols(headOutputs...)
	out := Add(MatMul(concat, m.WO), m.BO)
	return out
}

// transposeGraph returns a Tensor that is the transpose of a, participating in the graph.
func transposeGraph(a *Tensor) *Tensor {
	out := NewTensor(a.Cols, a.Rows, a.RequiresGrad)
	for r := range a.Rows {
		for c := range a.Cols {
			out.Data[c*a.Rows+r] = a.Data[r*a.Cols+c]
		}
	}
	out.children = []*Tensor{a}
	out.backwardFn = func() {
		if a.RequiresGrad {
			for r := range a.Rows {
				for c := range a.Cols {
					a.Grad[r*a.Cols+c] += out.Grad[c*a.Rows+r]
				}
			}
		}
	}
	return out
}

// Parameters returns all learnable weights.
func (m *MultiHeadAttention) Parameters() []*Tensor {
	return []*Tensor{m.WQ, m.BQ, m.WK, m.BK, m.WV, m.BV, m.WO, m.BO}
}
