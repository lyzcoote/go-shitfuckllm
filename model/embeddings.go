package model

import "math"

// Embeddings holds the token embedding table and precomputed sinusoidal positional encoding.
type Embeddings struct {
	TokenEmbed *Tensor    // [VocabSize × DModel], learnable
	PosEnc     [][]float64 // [MaxSeqLen × DModel], fixed
	DModel     int
	VocabSize  int
	MaxSeqLen  int
}

// NewEmbeddings creates the embedding table and precomputes positional encodings.
func NewEmbeddings(vocabSize, dModel, maxSeqLen int) *Embeddings {
	embed := NewTensor(vocabSize, dModel, true)
	// Xavier uniform initialization: scale = sqrt(6 / (fan_in + fan_out))
	scale := math.Sqrt(6.0 / float64(vocabSize+dModel))
	for i := range embed.Data {
		embed.Data[i] = (globalRand()-0.5) * 2 * scale
	}

	// Sinusoidal positional encoding (fixed, not trained)
	pos := make([][]float64, maxSeqLen)
	for p := range pos {
		pos[p] = make([]float64, dModel)
		for i := 0; i < dModel/2; i++ {
			angle := float64(p) / math.Pow(10000.0, 2.0*float64(i)/float64(dModel))
			pos[p][2*i] = math.Sin(angle)
			pos[p][2*i+1] = math.Cos(angle)
		}
	}

	return &Embeddings{
		TokenEmbed: embed,
		PosEnc:     pos,
		DModel:     dModel,
		VocabSize:  vocabSize,
		MaxSeqLen:  maxSeqLen,
	}
}

// Forward looks up token embeddings and adds positional encodings.
// tokens is a slice of token IDs; returns [len(tokens) × DModel].
func (e *Embeddings) Forward(tokens []int) *Tensor {
	seqLen := len(tokens)
	out := NewTensor(seqLen, e.DModel, true)

	for i, tok := range tokens {
		if tok < 0 || tok >= e.VocabSize {
			tok = 3 // UNK
		}
		for d := 0; d < e.DModel; d++ {
			out.Data[i*e.DModel+d] = e.TokenEmbed.Data[tok*e.DModel+d] + e.PosEnc[i][d]
		}
	}

	// Backward: propagate grad into the embedding rows
	toksCopy := make([]int, seqLen)
	copy(toksCopy, tokens)
	out.children = []*Tensor{e.TokenEmbed}
	out.backwardFn = func() {
		for i, tok := range toksCopy {
			if tok < 0 || tok >= e.VocabSize {
				tok = 3
			}
			for d := 0; d < e.DModel; d++ {
				e.TokenEmbed.Grad[tok*e.DModel+d] += out.Grad[i*e.DModel+d]
			}
		}
	}
	return out
}

// Parameters returns the learnable embedding table.
func (e *Embeddings) Parameters() []*Tensor {
	return []*Tensor{e.TokenEmbed}
}
