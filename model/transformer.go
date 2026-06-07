package model

import (
	"encoding/gob"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// TransformerConfig holds all model hyperparameters.
type TransformerConfig struct {
	VocabSize int     `json:"vocab_size"`
	DModel    int     `json:"d_model"`
	NHeads    int     `json:"n_heads"`
	NLayers   int     `json:"n_layers"`
	DFF       int     `json:"d_ff"`
	MaxSeqLen int     `json:"max_seq_len"`
	Dropout   float64 `json:"dropout"`
}

// DefaultConfig returns the hyperparameters specified in the project requirements.
func DefaultConfig() *TransformerConfig {
	return &TransformerConfig{
		VocabSize: 8192,
		DModel:    256,
		NHeads:    4,
		NLayers:   4,
		DFF:       1024,
		MaxSeqLen: 128,
		Dropout:   0.1,
	}
}

// TransformerBlock is one decoder layer: pre-norm attention + pre-norm FFN.
type TransformerBlock struct {
	Attn *MultiHeadAttention
	FFN  *FeedForward
	LN1  *LayerNorm
	LN2  *LayerNorm
}

// newTransformerBlock creates a single decoder block.
func newTransformerBlock(cfg *TransformerConfig) *TransformerBlock {
	return &TransformerBlock{
		Attn: NewMultiHeadAttention(cfg.DModel, cfg.NHeads),
		FFN:  NewFeedForward(cfg.DModel, cfg.DFF),
		LN1:  NewLayerNorm(cfg.DModel),
		LN2:  NewLayerNorm(cfg.DModel),
	}
}

// Forward applies one transformer block with pre-norm and residual connections.
func (b *TransformerBlock) Forward(x *Tensor) *Tensor {
	// Pre-norm attention + residual
	normed := b.LN1.Forward(x)
	attnOut := b.Attn.Forward(normed)
	x = Add(x, attnOut)

	// Pre-norm FFN + residual
	normed2 := b.LN2.Forward(x)
	ffnOut := b.FFN.Forward(normed2)
	x = Add(x, ffnOut)
	return x
}

// Parameters returns all learnable parameters of this block.
func (b *TransformerBlock) Parameters() []*Tensor {
	params := b.Attn.Parameters()
	params = append(params, b.FFN.Parameters()...)
	params = append(params, b.LN1.Parameters()...)
	params = append(params, b.LN2.Parameters()...)
	return params
}

// TransformerModel is the complete decoder-only transformer.
type TransformerModel struct {
	Config  *TransformerConfig
	Embed   *Embeddings
	Blocks  []*TransformerBlock
	LNFinal *LayerNorm
	LMHead  *Tensor // [DModel × VocabSize]
	Training bool
}

// NewTransformerModel constructs and initialises a transformer from the given config.
func NewTransformerModel(cfg *TransformerConfig) *TransformerModel {
	blocks := make([]*TransformerBlock, cfg.NLayers)
	for i := range cfg.NLayers {
		blocks[i] = newTransformerBlock(cfg)
	}

	// LM head: project DModel → VocabSize (weight tying with embed is optional)
	lmHead := NewTensor(cfg.DModel, cfg.VocabSize, true)
	scale := 0.02
	for i := range lmHead.Data {
		lmHead.Data[i] = randNorm() * scale
	}

	return &TransformerModel{
		Config:  cfg,
		Embed:   NewEmbeddings(cfg.VocabSize, cfg.DModel, cfg.MaxSeqLen),
		Blocks:  blocks,
		LNFinal: NewLayerNorm(cfg.DModel),
		LMHead:  lmHead,
		Training: false,
	}
}

// Forward runs the full forward pass. tokens is a sequence of token IDs.
// Returns logits [seqLen × VocabSize].
func (m *TransformerModel) Forward(tokens []int) *Tensor {
	x := m.Embed.Forward(tokens) // [seqLen × DModel]
	for _, block := range m.Blocks {
		x = block.Forward(x)
	}
	x = m.LNFinal.Forward(x)          // [seqLen × DModel]
	logits := MatMul(x, m.LMHead)     // [seqLen × VocabSize]
	return logits
}

// Parameters returns every learnable parameter in the model.
func (m *TransformerModel) Parameters() []*Tensor {
	params := m.Embed.Parameters()
	for _, b := range m.Blocks {
		params = append(params, b.Parameters()...)
	}
	params = append(params, m.LNFinal.Parameters()...)
	params = append(params, m.LMHead)
	return params
}

// NumParameters returns the total parameter count.
func (m *TransformerModel) NumParameters() int {
	total := 0
	for _, p := range m.Parameters() {
		total += len(p.Data)
	}
	return total
}

// ZeroGrad zeros gradients for all parameters.
func (m *TransformerModel) ZeroGrad() {
	for _, p := range m.Parameters() {
		p.ZeroGrad()
	}
}

// ────────────────────────────────────────────────────────────────────────────
// Checkpoint save / load
// ────────────────────────────────────────────────────────────────────────────

type checkpointData struct {
	Params [][]float64
}

// Save serialises all parameter Data slices to dir/model.bin and the config to dir/config.json.
func (m *TransformerModel) Save(dir string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("mkdir %s: %w", dir, err)
	}

	// Save config
	cfgBytes, err := json.MarshalIndent(m.Config, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, "config.json"), cfgBytes, 0o644); err != nil {
		return err
	}

	// Save weights
	params := m.Parameters()
	cd := checkpointData{Params: make([][]float64, len(params))}
	for i, p := range params {
		cd.Params[i] = make([]float64, len(p.Data))
		copy(cd.Params[i], p.Data)
	}
	f, err := os.Create(filepath.Join(dir, "model.bin"))
	if err != nil {
		return err
	}
	defer f.Close()
	return gob.NewEncoder(f).Encode(&cd)
}

// Load restores weights from dir/model.bin. The model must have the same architecture.
func (m *TransformerModel) Load(dir string) error {
	f, err := os.Open(filepath.Join(dir, "model.bin"))
	if err != nil {
		return fmt.Errorf("open model.bin: %w", err)
	}
	defer f.Close()

	var cd checkpointData
	if err := gob.NewDecoder(f).Decode(&cd); err != nil {
		return fmt.Errorf("decode model.bin: %w", err)
	}

	params := m.Parameters()
	if len(cd.Params) != len(params) {
		return fmt.Errorf("parameter count mismatch: checkpoint has %d, model has %d", len(cd.Params), len(params))
	}
	for i, p := range params {
		if len(cd.Params[i]) != len(p.Data) {
			return fmt.Errorf("param %d size mismatch: %d vs %d", i, len(cd.Params[i]), len(p.Data))
		}
		copy(p.Data, cd.Params[i])
	}
	return nil
}

// LoadConfig reads a config.json from dir.
func LoadConfig(dir string) (*TransformerConfig, error) {
	data, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	var cfg TransformerConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, err
	}
	return &cfg, nil
}
