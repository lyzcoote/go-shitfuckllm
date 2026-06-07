package model

import "math"

// AdamWConfig holds all hyperparameters for the AdamW optimizer.
type AdamWConfig struct {
	LR          float64 // base learning rate
	Beta1       float64 // first moment decay (default 0.9)
	Beta2       float64 // second moment decay (default 0.999)
	Eps         float64 // numerical stability (default 1e-8)
	WeightDecay float64 // L2 regularisation (default 0.01)
	GradClip    float64 // max global gradient norm (default 1.0)
	WarmupSteps int     // linear warmup steps
	TotalSteps  int     // total training steps (for cosine decay)
}

// DefaultAdamWConfig returns sensible defaults matching the project requirements.
func DefaultAdamWConfig(totalSteps int) AdamWConfig {
	return AdamWConfig{
		LR:          3e-4,
		Beta1:       0.9,
		Beta2:       0.999,
		Eps:         1e-8,
		WeightDecay: 0.01,
		GradClip:    1.0,
		WarmupSteps: 500,
		TotalSteps:  totalSteps,
	}
}

// AdamW implements the AdamW optimiser with a warmup + cosine decay schedule.
type AdamW struct {
	Config AdamWConfig
	Params []*Tensor
	m      [][]float64 // first moment estimates
	v      [][]float64 // second moment estimates
	step   int
}

// NewAdamW creates an AdamW optimiser over the given parameters.
func NewAdamW(params []*Tensor, cfg AdamWConfig) *AdamW {
	m := make([][]float64, len(params))
	v := make([][]float64, len(params))
	for i, p := range params {
		m[i] = make([]float64, len(p.Data))
		v[i] = make([]float64, len(p.Data))
	}
	return &AdamW{Config: cfg, Params: params, m: m, v: v}
}

// LearningRate returns the current learning rate after schedule.
func (o *AdamW) LearningRate() float64 {
	step := o.step + 1
	lr := o.Config.LR

	if step <= o.Config.WarmupSteps {
		// Linear warmup from 0 to lr
		return lr * float64(step) / float64(o.Config.WarmupSteps)
	}
	// Cosine decay from lr to lr*0.1
	progress := float64(step-o.Config.WarmupSteps) / float64(max(1, o.Config.TotalSteps-o.Config.WarmupSteps))
	progress = math.Min(progress, 1.0)
	cosine := 0.5 * (1.0 + math.Cos(math.Pi*progress))
	return lr * (0.1 + 0.9*cosine)
}

// ZeroGrad zeros the gradients of all managed parameters.
func (o *AdamW) ZeroGrad() {
	for _, p := range o.Params {
		p.ZeroGrad()
	}
}

// ClipGradients clips all parameter gradients by global norm. Returns the pre-clip norm.
func (o *AdamW) ClipGradients() float64 {
	norm := 0.0
	for _, p := range o.Params {
		for _, g := range p.Grad {
			norm += g * g
		}
	}
	norm = math.Sqrt(norm)
	if norm > o.Config.GradClip && norm > 0 {
		scale := o.Config.GradClip / norm
		for _, p := range o.Params {
			rawScale(p.Grad, scale)
		}
	}
	return norm
}

// Step applies one AdamW update to all parameters.
func (o *AdamW) Step() {
	lr := o.LearningRate()
	o.step++
	b1, b2 := o.Config.Beta1, o.Config.Beta2
	eps := o.Config.Eps
	wd := o.Config.WeightDecay

	bc1 := 1.0 - math.Pow(b1, float64(o.step))
	bc2 := 1.0 - math.Pow(b2, float64(o.step))

	for i, p := range o.Params {
		for j, g := range p.Grad {
			o.m[i][j] = b1*o.m[i][j] + (1-b1)*g
			o.v[i][j] = b2*o.v[i][j] + (1-b2)*g*g

			mHat := o.m[i][j] / bc1
			vHat := o.v[i][j] / bc2

			// AdamW: weight decay applied directly to weights (decoupled)
			p.Data[j] -= lr * (mHat/(math.Sqrt(vHat)+eps) + wd*p.Data[j])
		}
	}
}
