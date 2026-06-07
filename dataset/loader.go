package dataset

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"strings"

	"awesomeProject/tokenizer"
)

// Batch is a mini-batch of tokenized dialog sequences ready for training.
type Batch struct {
	InputIDs  [][]int  // [BatchSize][SeqLen]
	TargetIDs [][]int  // [BatchSize][SeqLen] — shifted left by 1, padded with PAD
	Masks     [][]bool // [BatchSize][SeqLen] — true = real (non-pad) token
}

// Dataset holds all dialog pairs with an attached tokenizer.
type Dataset struct {
	Pairs []DialogPair
	BPE   *tokenizer.BPE
}

// LoadJSONL reads a JSONL file and builds a Dataset using the provided BPE tokenizer.
func LoadJSONL(path string, bpe *tokenizer.BPE) (*Dataset, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}
	defer f.Close()

	var pairs []DialogPair
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1<<20), 1<<20)
	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}
		var p DialogPair
		if err := json.Unmarshal(line, &p); err != nil {
			return nil, fmt.Errorf("parse line: %w", err)
		}
		pairs = append(pairs, p)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scan: %w", err)
	}
	return &Dataset{Pairs: pairs, BPE: bpe}, nil
}

// WriteJSONL writes dialog pairs to path in the same JSONL format LoadJSONL reads
// (one {"input","output","lang"} object per line). When appendMode is true it appends
// to an existing file (creating it if absent) instead of truncating — useful for
// accumulating several channel harvests into one dataset. Returns the number of pairs
// written. Empty pairs (blank input or output) are skipped.
func WriteJSONL(path string, pairs []DialogPair, appendMode bool) (int, error) {
	if dir := filepath.Dir(path); dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return 0, fmt.Errorf("mkdir %s: %w", dir, err)
		}
	}

	flags := os.O_CREATE | os.O_WRONLY | os.O_TRUNC
	if appendMode {
		flags = os.O_CREATE | os.O_WRONLY | os.O_APPEND
	}
	f, err := os.OpenFile(path, flags, 0o644)
	if err != nil {
		return 0, fmt.Errorf("open %s: %w", path, err)
	}
	defer f.Close()

	w := bufio.NewWriter(f)
	enc := json.NewEncoder(w) // Encode writes a trailing newline → one JSON object per line
	written := 0
	for _, p := range pairs {
		if strings.TrimSpace(p.Input) == "" || strings.TrimSpace(p.Output) == "" {
			continue
		}
		if err := enc.Encode(p); err != nil {
			return written, fmt.Errorf("encode pair: %w", err)
		}
		written++
	}
	if err := w.Flush(); err != nil {
		return written, fmt.Errorf("flush %s: %w", path, err)
	}
	return written, nil
}

// Split divides the dataset into train/val/test subsets using the given ratios.
// trainRatio + valRatio must be < 1.0; the remainder becomes test.
func (d *Dataset) Split(trainRatio, valRatio float64) (train, val, test *Dataset) {
	n := len(d.Pairs)
	nTrain := int(float64(n) * trainRatio)
	nVal := int(float64(n) * valRatio)

	mkDataset := func(pairs []DialogPair) *Dataset {
		return &Dataset{Pairs: pairs, BPE: d.BPE}
	}
	return mkDataset(d.Pairs[:nTrain]),
		mkDataset(d.Pairs[nTrain : nTrain+nVal]),
		mkDataset(d.Pairs[nTrain+nVal:])
}

// Shuffle randomises the order of pairs in-place.
func (d *Dataset) Shuffle(rng *rand.Rand) {
	rng.Shuffle(len(d.Pairs), func(i, j int) {
		d.Pairs[i], d.Pairs[j] = d.Pairs[j], d.Pairs[i]
	})
}

// Batches tokenises all pairs and returns mini-batches.
// Each sequence is built as: [BOS] input_tokens → output_tokens [EOS]
// truncated / padded to seqLen. Target is the same sequence shifted left by 1.
func (d *Dataset) Batches(batchSize, seqLen int) []Batch {
	// Tokenise all pairs
	sequences := make([][]int, 0, len(d.Pairs))
	for _, p := range d.Pairs {
		text := p.Input + " → " + p.Output
		ids := d.BPE.Encode(text) // includes BOS and EOS
		// Truncate to seqLen
		if len(ids) > seqLen {
			ids = ids[:seqLen]
		}
		sequences = append(sequences, ids)
	}

	var batches []Batch
	for start := 0; start < len(sequences); start += batchSize {
		end := start + batchSize
		if end > len(sequences) {
			end = len(sequences)
		}
		batch := makeBatch(sequences[start:end], seqLen)
		batches = append(batches, batch)
	}
	return batches
}

// makeBatch pads a slice of sequences to seqLen and builds input/target/mask.
func makeBatch(seqs [][]int, seqLen int) Batch {
	bs := len(seqs)
	inputIDs := make([][]int, bs)
	targetIDs := make([][]int, bs)
	masks := make([][]bool, bs)

	for i, seq := range seqs {
		inp := make([]int, seqLen)
		tgt := make([]int, seqLen)
		msk := make([]bool, seqLen)

		// Fill with PAD
		for j := range inp {
			inp[j] = tokenizer.PAD
			tgt[j] = tokenizer.PAD
		}

		// Copy tokens
		for j, tok := range seq {
			if j >= seqLen {
				break
			}
			inp[j] = tok
			msk[j] = true
		}

		// Target = input shifted left by 1
		for j := 0; j < seqLen-1; j++ {
			tgt[j] = inp[j+1]
		}
		// Last target position is PAD (no next token)
		tgt[seqLen-1] = tokenizer.PAD

		inputIDs[i] = inp
		targetIDs[i] = tgt
		masks[i] = msk
	}
	return Batch{InputIDs: inputIDs, TargetIDs: targetIDs, Masks: masks}
}
