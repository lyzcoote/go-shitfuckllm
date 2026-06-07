package model

// LayerNorm applies layer normalization with learnable gamma (scale) and beta (bias).
type LayerNorm struct {
	Gamma *Tensor // [1 × DModel]
	Beta  *Tensor // [1 × DModel]
	DModel int
	Eps    float64
}

// NewLayerNorm creates a LayerNorm with gamma=1, beta=0.
func NewLayerNorm(dModel int) *LayerNorm {
	gamma := NewTensor(1, dModel, true)
	beta := NewTensor(1, dModel, true)
	for i := range gamma.Data {
		gamma.Data[i] = 1.0
	}
	return &LayerNorm{Gamma: gamma, Beta: beta, DModel: dModel, Eps: 1e-5}
}

// Forward normalizes x [seqLen × dModel] and returns the result.
// The residual is added after: out = LayerNorm(x) is passed to sub-layer,
// then sub-layer output is added to x externally (pre-norm pattern).
func (ln *LayerNorm) Forward(x *Tensor) *Tensor {
	return LayerNormForward(x, ln.Gamma, ln.Beta, ln.Eps)
}

// Parameters returns the learnable parameters.
func (ln *LayerNorm) Parameters() []*Tensor {
	return []*Tensor{ln.Gamma, ln.Beta}
}
