// Package train implements the full training loop for the transformer model.
package train

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"time"

	"awesomeProject/dataset"
	"awesomeProject/model"
	"awesomeProject/tokenizer"
)

// Config holds all training hyperparameters.
type Config struct {
	DModel            int
	NHeads            int
	NLayers           int
	DFF               int
	BatchSize         int
	SeqLen            int
	VocabSize         int
	LR                float64
	Epochs            int
	WarmupSteps       int
	CheckpointDir     string
	EarlyStopPatience int
	LogEveryN         int
	Seed              int64
	DataPath          string
	VocabPath         string
	Workers           int // data-parallel workers (≤1 = sequential)
}

// DefaultConfig returns training defaults from the project requirements.
func DefaultConfig() Config {
	return Config{
		DModel:            256,
		NHeads:            4,
		NLayers:           4,
		DFF:               1024,
		BatchSize:         32,
		SeqLen:            128,
		VocabSize:         8192,
		LR:                3e-4,
		Epochs:            20,
		WarmupSteps:       500,
		CheckpointDir:     "checkpoints",
		EarlyStopPatience: 3,
		LogEveryN:         10,
		Seed:              42,
		DataPath:          "data/train.jsonl",
		VocabPath:         "data/vocab.json",
		Workers:           1,
	}
}

// Trainer orchestrates data loading, model training, checkpointing, and evaluation.
type Trainer struct {
	Model        *model.TransformerModel
	Opt          *model.AdamW
	BPE          *tokenizer.BPE
	TrainBatches []dataset.Batch
	ValBatches   []dataset.Batch
	Cfg          Config
	Step         int
	BestVal      float64
	NoImprove    int
	StartTime    time.Time

	workers  int
	replicas []*model.TransformerModel // for data-parallel training (workers-1 replicas)
}

// New builds a Trainer: generates dataset if needed, trains BPE, loads data, creates model.
func New(cfg Config) (*Trainer, error) {
	rng := rand.New(rand.NewSource(cfg.Seed))

	// 1. Generate dataset if missing
	if _, err := os.Stat(cfg.DataPath); os.IsNotExist(err) {
		log.Println("Generating dataset...")
		n, err := dataset.GenerateDataset(cfg.DataPath, cfg.Seed)
		if err != nil {
			return nil, fmt.Errorf("generate dataset: %w", err)
		}
		log.Printf("Generated %d dialog pairs → %s\n", n, cfg.DataPath)
	}

	// 2. Train or load BPE
	var bpe *tokenizer.BPE
	if _, err := os.Stat(cfg.VocabPath); os.IsNotExist(err) {
		log.Println("Training BPE tokenizer...")
		var err2 error
		bpe, err2 = trainBPEFromFile(cfg.DataPath, cfg.VocabPath, cfg.VocabSize)
		if err2 != nil {
			return nil, fmt.Errorf("train BPE: %w", err2)
		}
		log.Printf("BPE trained: vocab size = %d\n", bpe.VocabLen())
	} else {
		var err error
		bpe, err = tokenizer.Load(cfg.VocabPath)
		if err != nil {
			return nil, fmt.Errorf("load BPE: %w", err)
		}
		log.Printf("BPE loaded: vocab size = %d\n", bpe.VocabLen())
	}

	// 3. Load & split data
	ds, err := dataset.LoadJSONL(cfg.DataPath, bpe)
	if err != nil {
		return nil, fmt.Errorf("load dataset: %w", err)
	}
	ds.Shuffle(rng)
	trainDS, valDS, _ := ds.Split(0.8, 0.1)
	log.Printf("Dataset: %d train, %d val pairs\n", len(trainDS.Pairs), len(valDS.Pairs))

	trainBatches := trainDS.Batches(cfg.BatchSize, cfg.SeqLen)
	valBatches := valDS.Batches(cfg.BatchSize, cfg.SeqLen)

	// 4. Build model
	actualVocab := bpe.VocabLen()
	modelCfg := &model.TransformerConfig{
		VocabSize: actualVocab,
		DModel:    cfg.DModel,
		NHeads:    cfg.NHeads,
		NLayers:   cfg.NLayers,
		DFF:       cfg.DFF,
		MaxSeqLen: cfg.SeqLen,
		Dropout:   0.1,
	}
	m := model.NewTransformerModel(modelCfg)
	log.Printf("Model: %d parameters\n", m.NumParameters())

	// 5. Optimizer
	totalSteps := len(trainBatches) * cfg.Epochs * cfg.BatchSize
	optCfg := model.DefaultAdamWConfig(totalSteps)
	optCfg.LR = cfg.LR
	optCfg.WarmupSteps = cfg.WarmupSteps
	opt := model.NewAdamW(m.Parameters(), optCfg)

	workers := cfg.Workers
	if workers < 1 {
		workers = 1
	}
	replicas := makeReplicas(m, workers)
	if workers > 1 {
		log.Printf("Data-parallel training: %d workers (%d replicas)\n", workers, len(replicas))
	}

	return &Trainer{
		Model:        m,
		Opt:          opt,
		BPE:          bpe,
		TrainBatches: trainBatches,
		ValBatches:   valBatches,
		Cfg:          cfg,
		BestVal:      math.Inf(1),
		StartTime:    time.Now(),
		workers:      workers,
		replicas:     replicas,
	}, nil
}

// Run executes the full training loop.
func (t *Trainer) Run() error {
	if err := os.MkdirAll(t.Cfg.CheckpointDir, 0o755); err != nil {
		return err
	}

	for epoch := 1; epoch <= t.Cfg.Epochs; epoch++ {
		trainLoss := t.trainEpoch(epoch)
		valLoss, valPpl := t.evalEpoch()

		log.Printf("Epoch %d/%d — train_loss=%.4f  val_loss=%.4f  val_ppl=%.2f  lr=%.2e\n",
			epoch, t.Cfg.Epochs, trainLoss, valLoss, valPpl, t.Opt.LearningRate())

		// Checkpoint
		ckptDir := filepath.Join(t.Cfg.CheckpointDir, fmt.Sprintf("epoch_%02d", epoch))
		if err := t.Model.Save(ckptDir); err != nil {
			log.Printf("Warning: checkpoint save failed: %v\n", err)
		} else {
			log.Printf("Checkpoint saved → %s\n", ckptDir)
		}

		// Early stopping
		if valLoss < t.BestVal {
			t.BestVal = valLoss
			t.NoImprove = 0
			bestDir := filepath.Join(t.Cfg.CheckpointDir, "best")
			_ = t.Model.Save(bestDir)
		} else {
			t.NoImprove++
			if t.NoImprove >= t.Cfg.EarlyStopPatience {
				log.Printf("Early stopping after %d epochs without improvement.\n", t.NoImprove)
				break
			}
		}
	}

	elapsed := time.Since(t.StartTime)
	log.Printf("Training complete in %v. Best val loss: %.4f\n", elapsed.Round(time.Second), t.BestVal)
	return nil
}

// trainEpoch runs one full pass over the training batches.
func (t *Trainer) trainEpoch(epoch int) float64 {
	totalLoss := 0.0
	steps := 0
	total := len(t.TrainBatches)

	t.Opt.ZeroGrad()

	for bIdx, batch := range t.TrainBatches {
		var batchLoss float64
		var counted int

		if t.workers > 1 && len(batch.InputIDs) > 1 {
			// Data-parallel: fan the batch's examples out across model replicas, then
			// reduce gradients into the master. Gradients are summed (not averaged), matching
			// the sequential path below.
			sumLoss, n := parallelAccumulate(t.Model, t.replicas, t.workers, batch.InputIDs, batch.TargetIDs)
			batchLoss, counted = sumLoss, n
		} else {
			t.Opt.ZeroGrad()
			for i := range batch.InputIDs {
				logits := t.Model.Forward(batch.InputIDs[i])
				loss := model.CrossEntropyLoss(logits, batch.TargetIDs[i], tokenizer.PAD)
				loss.Backward()
				batchLoss += loss.Data[0]
				counted++
			}
		}
		if counted > 0 {
			batchLoss /= float64(counted)
		}

		norm := t.Opt.ClipGradients()
		t.Opt.Step()
		t.Opt.ZeroGrad()
		t.Step++

		totalLoss += batchLoss
		steps++

		if steps%t.Cfg.LogEveryN == 0 || bIdx == total-1 {
			ppl := math.Exp(batchLoss)
			bar := progressBar(bIdx+1, total, 30)
			fmt.Printf("\rEpoch %d %s loss=%.4f ppl=%.2f lr=%.2e gnorm=%.3f",
				epoch, bar, batchLoss, ppl, t.Opt.LearningRate(), norm)
		}
	}
	fmt.Println()
	if steps == 0 {
		return 0
	}
	return totalLoss / float64(steps)
}

// evalEpoch computes loss and perplexity on the validation set.
func (t *Trainer) evalEpoch() (loss, ppl float64) {
	totalLoss := 0.0
	count := 0
	for _, batch := range t.ValBatches {
		for i := range batch.InputIDs {
			logits := t.Model.Forward(batch.InputIDs[i])
			l := model.CrossEntropyLoss(logits, batch.TargetIDs[i], tokenizer.PAD)
			totalLoss += l.Data[0]
			count++
		}
	}
	if count == 0 {
		return 0, 1
	}
	avgLoss := totalLoss / float64(count)
	return avgLoss, math.Exp(avgLoss)
}

// progressBar returns a simple ASCII progress bar string.
func progressBar(done, total, width int) string {
	if total == 0 {
		return "[" + strings.Repeat(" ", width) + "]"
	}
	frac := float64(done) / float64(total)
	filled := int(frac * float64(width))
	if filled > width {
		filled = width
	}
	bar := "[" + strings.Repeat("=", filled)
	if filled < width {
		bar += ">"
		bar += strings.Repeat(" ", width-filled-1)
	}
	bar += fmt.Sprintf("] %d/%d", done, total)
	return bar
}

// BLEUScore computes corpus-level BLEU-4 between hypothesis and reference token sequences.
func BLEUScore(hypotheses, references [][]int) float64 {
	if len(hypotheses) == 0 {
		return 0
	}

	var logSum float64
	var hypLen, refLen int

	for n := 1; n <= 4; n++ {
		matches := 0
		total := 0
		for k, hyp := range hypotheses {
			ref := references[k]
			hNgrams := ngramCounts(hyp, n)
			rNgrams := ngramCounts(ref, n)
			for ng, cnt := range hNgrams {
				refCnt := rNgrams[ng]
				clip := cnt
				if clip > refCnt {
					clip = refCnt
				}
				matches += clip
				total += cnt
			}
		}
		if total == 0 || matches == 0 {
			return 0
		}
		logSum += math.Log(float64(matches) / float64(total))
	}

	for _, h := range hypotheses {
		hypLen += len(h)
	}
	for _, r := range references {
		refLen += len(r)
	}

	bp := 1.0
	if hypLen < refLen {
		bp = math.Exp(1.0 - float64(refLen)/float64(hypLen))
	}
	return bp * math.Exp(logSum/4.0)
}

// ngramCounts returns a map of n-gram keys → count for tokens.
func ngramCounts(tokens []int, n int) map[string]int {
	counts := make(map[string]int)
	for i := 0; i <= len(tokens)-n; i++ {
		parts := make([]string, n)
		for j := range n {
			parts[j] = fmt.Sprintf("%d", tokens[i+j])
		}
		key := strings.Join(parts, ",")
		counts[key]++
	}
	return counts
}

// trainBPEFromFile reads corpus from a JSONL file and trains a BPE tokenizer.
func trainBPEFromFile(dataPath, vocabPath string, vocabSize int) (*tokenizer.BPE, error) {
	f, err := os.Open(dataPath)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var corpus []string
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1<<20), 1<<20)
	for scanner.Scan() {
		var p dataset.DialogPair
		if err := json.Unmarshal(scanner.Bytes(), &p); err != nil {
			continue
		}
		corpus = append(corpus, p.Input)
		corpus = append(corpus, p.Output)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}

	bpe := tokenizer.New()
	bpe.Train(corpus, vocabSize)

	if err := bpe.Save(vocabPath); err != nil {
		return nil, err
	}
	return bpe, nil
}
