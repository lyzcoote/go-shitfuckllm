package main

import (
	"os"
	"path/filepath"
	"testing"

	"awesomeProject/model"
	"awesomeProject/tokenizer"
	"awesomeProject/train"
)

// TestSeedCorpusEncodesChat verifies the bootstrap tokenizer (built from the character
// seed, with no dialog dataset) can encode realistic Italian/English chat text without
// an explosion of [UNK] tokens — so the bot can start and learn purely from chat.
func TestSeedCorpusEncodesChat(t *testing.T) {
	bpe := tokenizer.New()
	bpe.Train(seedCorpus(), bootstrapVocabSize)

	samples := []string{
		"Ciao come stai? Tutto bene 😊",
		"hello how are you doing today",
		"perché non rispondi? 🤔",
		"lol that was great 🔥 gg",
		"Buongiorno a tutti! 100% daccordo",
	}

	total, unk := 0, 0
	for _, s := range samples {
		ids := bpe.EncodeNoSpecial(s)
		if len(ids) == 0 {
			t.Errorf("empty encoding for %q", s)
		}
		for _, id := range ids {
			total++
			if id == tokenizer.UNK {
				unk++
			}
		}
	}

	if total == 0 {
		t.Fatal("no tokens produced")
	}
	unkRate := float64(unk) / float64(total)
	if unkRate > 0.10 {
		t.Errorf("UNK rate too high: %.1f%% (%d/%d) — seed coverage insufficient", unkRate*100, unk, total)
	}

	// The "→" pair separator must round-trip without UNK (used by online training pairs).
	for _, id := range bpe.EncodeNoSpecial("ciao → bene") {
		if id == tokenizer.UNK {
			t.Error("pair separator text produced an UNK token")
		}
	}
}

// TestCheckpointResumeRoundTrip verifies that an online checkpoint is self-contained
// (bundles vocab.json) and that loadModelAndBPE can resume it — so continual learning
// persists across restarts and load path == save path.
func TestCheckpointResumeRoundTrip(t *testing.T) {
	bpe := tokenizer.New()
	bpe.Train(seedCorpus(), 256)
	cfg := &model.TransformerConfig{VocabSize: bpe.VocabLen(), DModel: 16, NHeads: 2, NLayers: 1, DFF: 32, MaxSeqLen: 32}
	m := model.NewTransformerModel(cfg)
	tr := train.NewOnlineTrainer(m, bpe, train.DefaultOnlineConfig())

	dir := t.TempDir()
	if err := tr.SaveLatest(dir); err != nil {
		t.Fatalf("SaveLatest: %v", err)
	}

	// The checkpoint must bundle the vocabulary, not rely on data/vocab.json.
	if _, err := os.Stat(filepath.Join(dir, "vocab.json")); err != nil {
		t.Fatalf("checkpoint did not bundle vocab.json: %v", err)
	}

	m2, bpe2, err := loadModelAndBPE(dir)
	if err != nil {
		t.Fatalf("loadModelAndBPE: %v", err)
	}
	if bpe2.VocabLen() != bpe.VocabLen() {
		t.Errorf("vocab size mismatch after resume: %d vs %d", bpe2.VocabLen(), bpe.VocabLen())
	}
	if m2.Config.VocabSize != cfg.VocabSize {
		t.Errorf("model vocab size mismatch after resume: %d vs %d", m2.Config.VocabSize, cfg.VocabSize)
	}
}
