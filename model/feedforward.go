package model

import "math"

// FeedForward implements a two-layer FFN: Linear → GELU → Linear.
type FeedForward struct {
	W1, B1 *Tensor // [DModel × DFF], [1 × DFF]
	W2, B2 *Tensor // [DFF × DModel], [1 × DModel]
	DModel int
	DFF    int
}

// NewFeedForward initialises weights with He normal initialisation.
func NewFeedForward(dModel, dFF int) *FeedForward {
	w1 := NewTensor(dModel, dFF, true)
	w2 := NewTensor(dFF, dModel, true)
	b1 := NewTensor(1, dFF, true)
	b2 := NewTensor(1, dModel, true)

	scale1 := math.Sqrt(2.0 / float64(dModel))
	for i := range w1.Data {
		w1.Data[i] = randNorm() * scale1
	}
	scale2 := math.Sqrt(2.0 / float64(dFF))
	for i := range w2.Data {
		w2.Data[i] = randNorm() * scale2
	}
	// biases remain zero

	return &FeedForward{W1: w1, B1: b1, W2: w2, B2: b2, DModel: dModel, DFF: dFF}
}

// Forward computes GELU(x @ W1 + B1) @ W2 + B2.
// x is [seqLen × DModel]; output is [seqLen × DModel].
func (f *FeedForward) Forward(x *Tensor) *Tensor {
	h := Add(MatMul(x, f.W1), f.B1)    // [seqLen × DFF]
	h = GELU(h)                         // [seqLen × DFF]
	out := Add(MatMul(h, f.W2), f.B2)  // [seqLen × DModel]
	return out
}

// Parameters returns all learnable weights and biases.
func (f *FeedForward) Parameters() []*Tensor {
	return []*Tensor{f.W1, f.B1, f.W2, f.B2}
}
