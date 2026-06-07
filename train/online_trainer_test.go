package train

import (
	"context"
	"math"
	"sync"
	"testing"
	"time"

	"awesomeProject/model"
	"awesomeProject/tokenizer"
)

// smallTrainer builds an OnlineTrainer over a tiny model/BPE for fast tests.
func smallTrainer(t *testing.T, lr float64) *OnlineTrainer {
	t.Helper()
	bpe := tokenizer.New()
	bpe.Train([]string{"ciao come stai bene grazie hello how are you fine thanks i am good tutto bene foo bar"}, 96)
	cfg := &model.TransformerConfig{VocabSize: bpe.VocabLen(), DModel: 32, NHeads: 2, NLayers: 1, DFF: 64, MaxSeqLen: 32}
	model.SetSeed(7)
	m := model.NewTransformerModel(cfg)

	oc := DefaultOnlineConfig()
	oc.LR = lr
	oc.BufferSize = 16
	oc.SeqLen = 32
	oc.CheckpointEvery = 0
	oc.IdleFlush = 0
	return NewOnlineTrainer(m, bpe, oc)
}

// smallTrainerW is like smallTrainer but with a configurable data-parallel worker count.
func smallTrainerW(t *testing.T, lr float64, workers int) *OnlineTrainer {
	t.Helper()
	bpe := tokenizer.New()
	bpe.Train([]string{"ciao come stai bene grazie hello how are you fine thanks i am good tutto bene foo bar baz"}, 96)
	cfg := &model.TransformerConfig{VocabSize: bpe.VocabLen(), DModel: 32, NHeads: 2, NLayers: 1, DFF: 64, MaxSeqLen: 32}
	model.SetSeed(7)
	m := model.NewTransformerModel(cfg)

	oc := DefaultOnlineConfig()
	oc.LR = lr
	oc.BufferSize = 16
	oc.SeqLen = 32
	oc.CheckpointEvery = 0
	oc.IdleFlush = 0
	oc.Workers = workers
	return NewOnlineTrainer(m, bpe, oc)
}

// lossOf computes the current cross-entropy loss of a pair without training.
func lossOf(tr *OnlineTrainer, p TrainingPair) float64 {
	inp, tgt, ok := tr.encodePair(p)
	if !ok {
		return 0
	}
	logits := tr.model.Forward(inp)
	l := model.CrossEntropyLoss(logits, tgt, tokenizer.PAD)
	return l.Data[0]
}

// TestOnlineStepReducesLoss checks that two flushes over identical pairs reduce the loss.
func TestOnlineStepReducesLoss(t *testing.T) {
	tr := smallTrainer(t, 5e-3)
	pair := TrainingPair{Input: "ciao come stai", Output: "bene grazie"}

	before := lossOf(tr, pair)

	batch := make([]TrainingPair, 16)
	for i := range batch {
		batch[i] = pair
	}
	// Two flushes = two gradient steps (32 identical pairs at bufferSize 16).
	tr.trainBatch(batch)
	tr.trainBatch(batch)

	after := lossOf(tr, pair)
	if !(after < before) {
		t.Errorf("loss did not decrease after 2 flushes: before=%.4f after=%.4f", before, after)
	}
}

// TestNoExplodingGradients verifies weights stay bounded after 100 online steps.
func TestNoExplodingGradients(t *testing.T) {
	tr := smallTrainer(t, 1e-3)
	pairs := []TrainingPair{
		{Input: "ciao", Output: "bene grazie"},
		{Input: "hello how are you", Output: "i am good"},
		{Input: "come stai", Output: "tutto bene"},
		{Input: "thanks", Output: "fine"},
	}

	for step := 0; step < 100; step++ {
		tr.trainBatch([]TrainingPair{pairs[step%len(pairs)]})
	}

	for pi, p := range tr.model.Parameters() {
		for j, v := range p.Data {
			if math.IsNaN(v) || math.IsInf(v, 0) || v < -10 || v > 10 {
				t.Fatalf("weight out of range at param %d[%d] = %v", pi, j, v)
			}
		}
	}
}

// TestDataParallelReducesLoss verifies data-parallel training (workers=4) lowers the loss.
func TestDataParallelReducesLoss(t *testing.T) {
	tr := smallTrainerW(t, 5e-3, 4)
	pair := TrainingPair{Input: "ciao come stai", Output: "bene grazie"}

	before := lossOf(tr, pair)
	batch := make([]TrainingPair, 16)
	for i := range batch {
		batch[i] = pair
	}
	tr.trainBatch(batch)
	tr.trainBatch(batch)
	after := lossOf(tr, pair)

	if !(after < before) {
		t.Errorf("data-parallel loss did not decrease: before=%.4f after=%.4f", before, after)
	}
}

// TestDataParallelMatchesSequential checks that one data-parallel step lands on the same
// weights as a sequential step — data parallelism must be numerically equivalent to
// sequential gradient accumulation (up to floating-point summation order).
func TestDataParallelMatchesSequential(t *testing.T) {
	bpe := tokenizer.New()
	bpe.Train([]string{"ciao come stai bene grazie hello how are you fine thanks i am good tutto bene foo bar baz"}, 96)
	cfg := &model.TransformerConfig{VocabSize: bpe.VocabLen(), DModel: 32, NHeads: 2, NLayers: 1, DFF: 64, MaxSeqLen: 32}

	model.SetSeed(123)
	mSeq := model.NewTransformerModel(cfg)
	snap := snapshotWeights(mSeq)
	mPar := model.NewTransformerModel(cfg)
	restoreWeights(mPar, snap) // make both models start from identical weights

	oc := DefaultOnlineConfig()
	oc.LR = 1e-3
	oc.SeqLen = 32
	oc.CheckpointEvery = 0
	oc.IdleFlush = 0

	ocSeq := oc
	ocSeq.Workers = 1
	ocPar := oc
	ocPar.Workers = 4

	trSeq := NewOnlineTrainer(mSeq, bpe, ocSeq)
	trPar := NewOnlineTrainer(mPar, bpe, ocPar)

	batch := []TrainingPair{
		{Input: "ciao", Output: "bene grazie"},
		{Input: "hello how are you", Output: "i am good"},
		{Input: "come stai", Output: "tutto bene"},
		{Input: "thanks", Output: "fine"},
		{Input: "ciao come", Output: "stai bene"},
		{Input: "foo", Output: "bar baz"},
		{Input: "good", Output: "thanks"},
		{Input: "bene", Output: "grazie"},
	}
	trSeq.trainBatch(batch)
	trPar.trainBatch(batch)

	pa := mSeq.Parameters()
	pb := mPar.Parameters()
	var maxDiff float64
	for i := range pa {
		for j := range pa[i].Data {
			if d := math.Abs(pa[i].Data[j] - pb[i].Data[j]); d > maxDiff {
				maxDiff = d
			}
		}
	}
	t.Logf("max weight diff sequential vs parallel(4) = %g", maxDiff)
	if maxDiff > 1e-6 {
		t.Errorf("data-parallel diverged from sequential: max weight diff = %g (want ≤ 1e-6)", maxDiff)
	}
}

// TestParallelAccumulateMatchesSequential checks the shared data-parallel helper (used by
// BOTH the offline and online trainers) produces the same SUMMED gradients as a plain
// sequential accumulation over token batches.
func TestParallelAccumulateMatchesSequential(t *testing.T) {
	cfg := &model.TransformerConfig{VocabSize: 50, DModel: 32, NHeads: 2, NLayers: 1, DFF: 64, MaxSeqLen: 16}
	model.SetSeed(99)
	mSeq := model.NewTransformerModel(cfg)
	snap := snapshotWeights(mSeq)
	mPar := model.NewTransformerModel(cfg)
	restoreWeights(mPar, snap) // identical initial weights

	var inputs, targets [][]int
	for e := 0; e < 6; e++ {
		in := make([]int, 8)
		tg := make([]int, 8)
		for k := 0; k < 8; k++ {
			in[k] = (e*8+k)%49 + 1 // token ids 1..49 (avoid PAD=0)
			tg[k] = (e*8+k+1)%49 + 1
		}
		inputs = append(inputs, in)
		targets = append(targets, tg)
	}

	// Sequential summed gradients.
	mSeq.ZeroGrad()
	var seqLoss float64
	for i := range inputs {
		logits := mSeq.Forward(inputs[i])
		loss := model.CrossEntropyLoss(logits, targets[i], tokenizer.PAD)
		loss.Backward()
		seqLoss += loss.Data[0]
	}

	// Parallel summed gradients via the shared helper.
	replicas := makeReplicas(mPar, 4)
	parLoss, n := parallelAccumulate(mPar, replicas, 4, inputs, targets)
	if n != len(inputs) {
		t.Fatalf("counted %d examples, want %d", n, len(inputs))
	}

	ps := mSeq.Parameters()
	pp := mPar.Parameters()
	var maxDiff float64
	for i := range ps {
		for j := range ps[i].Grad {
			if d := math.Abs(ps[i].Grad[j] - pp[i].Grad[j]); d > maxDiff {
				maxDiff = d
			}
		}
	}
	t.Logf("max grad diff sequential vs parallel(4) = %g", maxDiff)
	if maxDiff > 1e-9 {
		t.Errorf("summed gradients diverge: max diff = %g", maxDiff)
	}
	if math.Abs(seqLoss-parLoss) > 1e-9 {
		t.Errorf("summed loss diverges: %v vs %v", seqLoss, parLoss)
	}
}

func snapshotWeights(m *model.TransformerModel) [][]float64 {
	ps := m.Parameters()
	s := make([][]float64, len(ps))
	for i, p := range ps {
		s[i] = append([]float64(nil), p.Data...)
	}
	return s
}

func restoreWeights(m *model.TransformerModel, s [][]float64) {
	ps := m.Parameters()
	for i, p := range ps {
		copy(p.Data, s[i])
	}
}

// TestConcurrentTrainAndRead runs the background trainer while another goroutine reads
// the model under ModelMu, validating the shared-mutex design is race-free (use -race).
func TestConcurrentTrainAndRead(t *testing.T) {
	tr := smallTrainer(t, 1e-4)
	ctx, cancel := context.WithCancel(context.Background())
	go tr.Start(ctx)

	var wg sync.WaitGroup

	// Producer: stream training pairs into the trainer.
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 200; i++ {
			tr.Submit(TrainingPair{Input: "ciao come stai", Output: "bene grazie"})
		}
	}()

	// Reader: simulate the bot's generation path reading weights under ModelMu.
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 200; i++ {
			tr.ModelMu.Lock()
			_ = tr.model.Forward([]int{1, 5, 6, 7, 2})
			tr.ModelMu.Unlock()
		}
	}()

	wg.Wait()
	cancel()
	time.Sleep(50 * time.Millisecond) // let the trainer observe ctx.Done and flush
}

// TestOnlineStepCounter verifies that a buffer flush advances the step counter.
func TestOnlineStepCounter(t *testing.T) {
	tr := smallTrainer(t, 1e-3)
	pair := TrainingPair{Input: "ciao come stai", Output: "bene grazie"}

	// Populate the internal buffer (normally done by the Start loop reading trainCh).
	tr.mu.Lock()
	for i := 0; i < 16; i++ {
		tr.buffer = append(tr.buffer, pair)
	}
	tr.mu.Unlock()

	tr.ForceFlush()
	if tr.Steps() != 1 {
		t.Errorf("step count after one flush = %d, want 1", tr.Steps())
	}
}

// TestCheckpointSaveLoad verifies a saved checkpoint reloads bit-for-bit identical.
func TestCheckpointSaveLoad(t *testing.T) {
	tr := smallTrainer(t, 1e-3)
	// Make the weights non-trivial first.
	tr.trainBatch([]TrainingPair{{Input: "ciao come stai", Output: "bene grazie"}})

	dir := t.TempDir()
	if err := tr.SaveLatest(dir); err != nil {
		t.Fatalf("SaveLatest: %v", err)
	}

	reloaded := model.NewTransformerModel(tr.model.Config)
	if err := reloaded.Load(dir); err != nil {
		t.Fatalf("Load: %v", err)
	}

	orig := tr.model.Parameters()
	got := reloaded.Parameters()
	if len(orig) != len(got) {
		t.Fatalf("param count mismatch: %d vs %d", len(orig), len(got))
	}
	for i := range orig {
		if len(orig[i].Data) != len(got[i].Data) {
			t.Fatalf("param %d size mismatch", i)
		}
		for j := range orig[i].Data {
			if orig[i].Data[j] != got[i].Data[j] {
				t.Fatalf("param %d[%d] mismatch: %v vs %v", i, j, orig[i].Data[j], got[i].Data[j])
			}
		}
	}
}
