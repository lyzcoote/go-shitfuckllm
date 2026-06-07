package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
)

// vocabFile is the JSON serialisation format for a BPE vocabulary.
type vocabFile struct {
	Vocab  map[string]int `json:"vocab"`
	Merges [][]string     `json:"merges"` // each entry is [A, B, Result]
}

// Save writes the vocabulary and merge rules to path as JSON.
func (b *BPE) Save(path string) error {
	vf := vocabFile{
		Vocab:  b.Vocab,
		Merges: make([][]string, len(b.Merges)),
	}
	for i, m := range b.Merges {
		vf.Merges[i] = []string{m.A, m.B, m.Result}
	}
	data, err := json.MarshalIndent(vf, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal vocab: %w", err)
	}
	return os.WriteFile(path, data, 0o644)
}

// Load restores a BPE from a JSON file previously written by Save.
func Load(path string) (*BPE, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read vocab: %w", err)
	}
	var vf vocabFile
	if err := json.Unmarshal(data, &vf); err != nil {
		return nil, fmt.Errorf("unmarshal vocab: %w", err)
	}

	b := New()
	b.Vocab = vf.Vocab
	b.InvVocab = make(map[int]string, len(vf.Vocab))
	for tok, id := range vf.Vocab {
		b.InvVocab[id] = tok
	}
	b.Merges = make([]Merge, len(vf.Merges))
	for i, m := range vf.Merges {
		if len(m) != 3 {
			return nil, fmt.Errorf("invalid merge entry at index %d", i)
		}
		b.Merges[i] = Merge{A: m[0], B: m[1], Result: m[2]}
		b.MergeRank[[2]string{m[0], m[1]}] = i
	}
	b.VocabSize = len(vf.Vocab)
	return b, nil
}
