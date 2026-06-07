package train

import (
	"context"
	"fmt"
	"log"
	"math"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"

	"awesomeProject/model"
	"awesomeProject/tokenizer"
)

// TrainingPair is a single (input, output) example for online learning. It mirrors
// the conversational structure used by the static dataset (input → output).
type TrainingPair struct {
	Input  string
	Output string
}

// OnlineConfig holds hyperparameters for continual, real-time learning. The learning
// rate is deliberately much lower than the initial training run, and gradient clipping
// is more restrictive, to keep updates small and avoid catastrophic forgetting.
type OnlineConfig struct {
	LR              float64       // online learning rate (default 1e-5)
	BufferSize      int           // pairs accumulated before each update (default 16)
	GradClip        float64       // max global gradient norm (default 0.5)
	SeqLen          int           // max sequence length per example
	CheckpointEvery int           // save a checkpoint every N online steps (default 500)
	CheckpointDir   string        // directory for online checkpoints
	IdleFlush       time.Duration // flush a partial buffer after this idle period (0 = never)
	ChannelCap      int           // training channel capacity
	EWCLambda       float64       // Elastic Weight Consolidation strength (0 = disabled)
	Workers         int           // data-parallel workers (≤1 = sequential). Each worker > 1 holds a model replica.
}

// DefaultOnlineConfig returns the online-learning defaults from the project spec.
func DefaultOnlineConfig() OnlineConfig {
	return OnlineConfig{
		LR:              1e-5,
		BufferSize:      16,
		GradClip:        0.5,
		SeqLen:          128,
		CheckpointEvery: 500,
		CheckpointDir:   "checkpoints",
		IdleFlush:       30 * time.Second,
		ChannelCap:      256,
		EWCLambda:       0.0,
		Workers:         1,
	}
}

// OnlineTrainer consumes TrainingPairs from a channel and continuously updates the
// model weights in the background, one small gradient step per buffer flush.
type OnlineTrainer struct {
	model     *model.TransformerModel
	optimizer *model.AdamW
	bpe       *tokenizer.BPE

	trainCh    chan TrainingPair
	buffer     []TrainingPair
	bufferSize int
	cfg        OnlineConfig
	mu         sync.Mutex // guards buffer

	// Data parallelism: workers ≥ 2 splits each mini-batch across goroutines, each
	// computing forward/backward on its own model replica (replicas = workers-1; worker 0
	// reuses the master). Gradients are reduced into the master before the optimiser step.
	workers  int
	replicas []*model.TransformerModel

	// ModelMu guards the shared model weights. The bot's generation path and the
	// message index's embedding lookups must lock it too, so reads never race with
	// the in-place weight updates performed here.
	ModelMu *sync.Mutex

	stepCount int64 // atomic

	// EWC anchoring (optional)
	starParams [][]float64
	fisher     [][]float64

	// stats
	statsMu  sync.Mutex
	lastLoss float64
	sumLoss  float64
	flushes  int64
}

// NewOnlineTrainer builds an online trainer over an already-constructed model. It uses
// AdamW with a near-constant low learning rate (warmup=1, very large horizon) so the
// schedule does not decay the rate during a long-running session.
func NewOnlineTrainer(m *model.TransformerModel, bpe *tokenizer.BPE, cfg OnlineConfig) *OnlineTrainer {
	optCfg := model.AdamWConfig{
		LR:          cfg.LR,
		Beta1:       0.9,
		Beta2:       0.999,
		Eps:         1e-8,
		WeightDecay: 0.0, // no decay online: don't pull learned weights toward zero
		GradClip:    cfg.GradClip,
		WarmupSteps: 1,
		TotalSteps:  1 << 30, // effectively constant LR over any realistic session
	}
	workers := cfg.Workers
	if workers < 1 {
		workers = 1
	}
	replicas := makeReplicas(m, workers)

	return &OnlineTrainer{
		model:      m,
		optimizer:  model.NewAdamW(m.Parameters(), optCfg),
		bpe:        bpe,
		trainCh:    make(chan TrainingPair, cfg.ChannelCap),
		buffer:     make([]TrainingPair, 0, cfg.BufferSize),
		bufferSize: cfg.BufferSize,
		cfg:        cfg,
		ModelMu:    &sync.Mutex{},
		workers:    workers,
		replicas:   replicas,
	}
}

// Channel returns a send-only handle to the training channel (used by the index).
func (t *OnlineTrainer) Channel() chan<- TrainingPair {
	return t.trainCh
}

// Submit enqueues a training pair without blocking; the pair is dropped if the channel
// is full, so producers (the Discord handler) are never stalled by training.
func (t *OnlineTrainer) Submit(p TrainingPair) {
	select {
	case t.trainCh <- p:
	default:
	}
}

// Start runs the background training loop until ctx is cancelled. It blocks on the
// training channel (no busy-waiting) and also flushes a partial buffer when the chat
// goes idle, so learning still happens during quiet periods.
func (t *OnlineTrainer) Start(ctx context.Context) {
	log.Printf("online trainer started (lr=%.1e, buffer=%d, clip=%.2f, workers=%d)", t.cfg.LR, t.bufferSize, t.cfg.GradClip, t.workers)

	interval := t.cfg.IdleFlush
	if interval <= 0 {
		interval = time.Hour
	}
	idle := time.NewTicker(interval)
	defer idle.Stop()

	for {
		select {
		case <-ctx.Done():
			t.ForceFlush()
			log.Println("online trainer stopped")
			return
		case pair := <-t.trainCh:
			t.mu.Lock()
			t.buffer = append(t.buffer, pair)
			full := len(t.buffer) >= t.bufferSize
			t.mu.Unlock()
			if full {
				t.flush()
			}
		case <-idle.C:
			t.flush() // drain whatever has accumulated during idle time
		}
	}
}

// ForceFlush performs an update on the current buffer regardless of its size. Safe to
// call on shutdown; it is a no-op when the buffer is empty.
func (t *OnlineTrainer) ForceFlush() {
	t.flush()
}

// flush pops the accumulated buffer and applies one gradient step, then logs and
// checkpoints as needed.
func (t *OnlineTrainer) flush() {
	t.mu.Lock()
	if len(t.buffer) == 0 {
		t.mu.Unlock()
		return
	}
	batch := t.buffer
	t.buffer = make([]TrainingPair, 0, t.bufferSize)
	t.mu.Unlock()

	start := time.Now()
	avgLoss, counted, norm := t.trainBatch(batch)
	if counted == 0 {
		return
	}

	step := atomic.AddInt64(&t.stepCount, 1)

	t.statsMu.Lock()
	t.lastLoss = avgLoss
	t.sumLoss += avgLoss
	t.flushes++
	t.statsMu.Unlock()

	log.Printf("online: step=%d loss=%.4f ppl=%.2f pairs=%d grad_norm=%.3f drain=%s",
		step, avgLoss, expSafe(avgLoss), counted, norm, time.Since(start).Round(time.Millisecond))

	if t.cfg.CheckpointEvery > 0 && step%int64(t.cfg.CheckpointEvery) == 0 {
		if err := t.saveCheckpoint(step); err != nil {
			log.Printf("online: checkpoint failed: %v", err)
		} else {
			log.Printf("online: checkpoint saved (step %d)", step)
		}
	}
}

// trainBatch runs forward/backward over a mini-batch and applies a single AdamW step.
// It holds ModelMu for the whole update so concurrent readers see a consistent model.
// With Workers > 1 it data-parallelises the batch across model replicas.
func (t *OnlineTrainer) trainBatch(batch []TrainingPair) (avgLoss float64, counted int, gradNorm float64) {
	t.ModelMu.Lock()
	defer t.ModelMu.Unlock()

	if t.workers > 1 && len(batch) > 1 {
		return t.trainBatchParallel(batch)
	}
	return t.trainBatchSeq(batch)
}

// trainBatchSeq processes the batch on a single core path (matmul may still parallelise
// internally). Caller must hold ModelMu.
func (t *OnlineTrainer) trainBatchSeq(batch []TrainingPair) (avgLoss float64, counted int, gradNorm float64) {
	t.optimizer.ZeroGrad()

	var total float64
	for _, pair := range batch {
		inp, tgt, ok := t.encodePair(pair)
		if !ok {
			continue
		}
		logits := t.model.Forward(inp)
		loss := model.CrossEntropyLoss(logits, tgt, tokenizer.PAD)
		loss.Backward() // gradients accumulate across the mini-batch
		total += loss.Data[0]
		counted++
	}
	if counted == 0 {
		t.optimizer.ZeroGrad()
		return 0, 0, 0
	}

	t.scaleGrads(1.0 / float64(counted))
	if t.cfg.EWCLambda > 0 && t.starParams != nil {
		t.addEWCGrad()
	}
	gradNorm = t.optimizer.ClipGradients()
	t.optimizer.Step()
	t.optimizer.ZeroGrad()

	return total / float64(counted), counted, gradNorm
}

// trainBatchParallel data-parallelises the mini-batch: worker 0 uses the master model,
// workers 1..W-1 use replicas. Each accumulates gradients into its own model; the
// gradients are then summed into the master and a single AdamW step is taken. This is
// mathematically equivalent to sequential gradient accumulation (the per-example gradients
// are the same set of terms, summed). Caller must hold ModelMu.
func (t *OnlineTrainer) trainBatchParallel(batch []TrainingPair) (avgLoss float64, counted int, gradNorm float64) {
	// Encode and filter pairs up front, then fan the examples out across replicas.
	inputs := make([][]int, 0, len(batch))
	targets := make([][]int, 0, len(batch))
	for _, pair := range batch {
		inp, tgt, ok := t.encodePair(pair)
		if !ok {
			continue
		}
		inputs = append(inputs, inp)
		targets = append(targets, tgt)
	}
	if len(inputs) == 0 {
		t.model.ZeroGrad()
		return 0, 0, 0
	}

	sumLoss, counted := parallelAccumulate(t.model, t.replicas, t.workers, inputs, targets)
	if counted == 0 {
		t.model.ZeroGrad()
		return 0, 0, 0
	}

	t.scaleGrads(1.0 / float64(counted))
	if t.cfg.EWCLambda > 0 && t.starParams != nil {
		t.addEWCGrad()
	}
	gradNorm = t.optimizer.ClipGradients()
	t.optimizer.Step()
	t.optimizer.ZeroGrad()

	return sumLoss / float64(counted), counted, gradNorm
}

// zeroSlice sets every element of s to zero.
func zeroSlice(s []float64) {
	for i := range s {
		s[i] = 0
	}
}

// encodePair builds (input, target) token slices from a pair using the same
// "input → output" convention as the static dataset; target is the input shifted by one.
func (t *OnlineTrainer) encodePair(p TrainingPair) (inp, tgt []int, ok bool) {
	text := p.Input + " → " + p.Output
	ids := t.bpe.Encode(text)
	if len(ids) > t.cfg.SeqLen {
		ids = ids[:t.cfg.SeqLen]
	}
	if len(ids) < 2 {
		return nil, nil, false
	}
	return ids[:len(ids)-1], ids[1:], true
}

// scaleGrads multiplies every parameter gradient by s.
func (t *OnlineTrainer) scaleGrads(s float64) {
	for _, p := range t.model.Parameters() {
		for i := range p.Grad {
			p.Grad[i] *= s
		}
	}
}

// ReplayBatch trains the model on a fixed set of pairs for the given number of epochs,
// in mini-batches of BufferSize. Used by --mode=replay and the /replay command. Returns
// the average loss of the final epoch.
func (t *OnlineTrainer) ReplayBatch(pairs []TrainingPair, epochs int) float64 {
	if len(pairs) == 0 || epochs <= 0 {
		return 0
	}
	var lastEpochLoss float64
	for e := 0; e < epochs; e++ {
		var sum float64
		var batches int
		for start := 0; start < len(pairs); start += t.bufferSize {
			end := start + t.bufferSize
			if end > len(pairs) {
				end = len(pairs)
			}
			avg, counted, _ := t.trainBatch(pairs[start:end])
			if counted > 0 {
				atomic.AddInt64(&t.stepCount, 1)
				sum += avg
				batches++
			}
		}
		if batches > 0 {
			lastEpochLoss = sum / float64(batches)
		}
		log.Printf("replay: epoch %d/%d loss=%.4f", e+1, epochs, lastEpochLoss)
	}
	return lastEpochLoss
}

// ────────────────────────────────────────────────────────────────────────────
// Elastic Weight Consolidation (optional anchoring to mitigate forgetting)
// ────────────────────────────────────────────────────────────────────────────

// EnableEWC turns on EWC regularisation toward the current weights (θ*). If fisher is
// nil a uniform importance of 1.0 is used; otherwise it must match Parameters() layout.
func (t *OnlineTrainer) EnableEWC(lambda float64, fisher [][]float64) {
	t.ModelMu.Lock()
	defer t.ModelMu.Unlock()
	t.cfg.EWCLambda = lambda
	t.fisher = fisher
	params := t.model.Parameters()
	t.starParams = make([][]float64, len(params))
	for i, p := range params {
		t.starParams[i] = append([]float64(nil), p.Data...)
	}
}

// addEWCGrad adds the gradient of λ·Σ Fᵢ·(θᵢ − θ*ᵢ)² to the parameter gradients.
// Caller must hold ModelMu.
func (t *OnlineTrainer) addEWCGrad() {
	params := t.model.Parameters()
	for i, p := range params {
		star := t.starParams[i]
		for j := range p.Data {
			f := 1.0
			if t.fisher != nil && i < len(t.fisher) && j < len(t.fisher[i]) {
				f = t.fisher[i][j]
			}
			p.Grad[j] += 2.0 * t.cfg.EWCLambda * f * (p.Data[j] - star[j])
		}
	}
}

// ────────────────────────────────────────────────────────────────────────────
// Checkpointing & stats
// ────────────────────────────────────────────────────────────────────────────

// saveCheckpoint writes the current model to checkpoints/online_<step>/.
func (t *OnlineTrainer) saveCheckpoint(step int64) error {
	dir := filepath.Join(t.cfg.CheckpointDir, fmt.Sprintf("online_%d", step))
	return t.saveTo(dir)
}

// SaveLatest writes the current model to a stable directory (used on shutdown).
func (t *OnlineTrainer) SaveLatest(dir string) error {
	t.ModelMu.Lock()
	defer t.ModelMu.Unlock()
	return t.saveTo(dir)
}

// saveTo writes a self-contained checkpoint: model weights, config, AND the tokenizer
// vocabulary. Bundling the vocab makes a checkpoint resumable on its own, avoiding a
// model/vocab size mismatch when data/vocab.json has changed between runs.
func (t *OnlineTrainer) saveTo(dir string) error {
	if err := t.model.Save(dir); err != nil {
		return err
	}
	return t.bpe.Save(filepath.Join(dir, "vocab.json"))
}

// OnlineStats is a snapshot of online-training progress for the /stats command.
type OnlineStats struct {
	Steps    int64
	Flushes  int64
	LastLoss float64
	AvgLoss  float64
	Buffered int
}

// Stats returns a snapshot of current online-training metrics.
func (t *OnlineTrainer) Stats() OnlineStats {
	t.statsMu.Lock()
	flushes := t.flushes
	last := t.lastLoss
	avg := 0.0
	if flushes > 0 {
		avg = t.sumLoss / float64(flushes)
	}
	t.statsMu.Unlock()

	t.mu.Lock()
	buffered := len(t.buffer)
	t.mu.Unlock()

	return OnlineStats{
		Steps:    atomic.LoadInt64(&t.stepCount),
		Flushes:  flushes,
		LastLoss: last,
		AvgLoss:  avg,
		Buffered: buffered,
	}
}

// Steps returns the number of online gradient steps performed so far.
func (t *OnlineTrainer) Steps() int64 {
	return atomic.LoadInt64(&t.stepCount)
}

// expSafe is exp with overflow guarding, for perplexity logging.
func expSafe(x float64) float64 {
	if x > 700 {
		return 1e300
	}
	return math.Exp(x)
}
