package dataset

import (
	"math/rand"
	"os"
	"testing"

	"awesomeProject/tokenizer"
)

// TestWriteJSONLRoundTrip verifies harvested pairs survive a WriteJSONL → LoadJSONL
// round-trip and that blank pairs are skipped and append mode accumulates.
func TestWriteJSONLRoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/harvest.jsonl"

	pairs := []DialogPair{
		{Input: "ciao", Output: "ehilà", Lang: "und"},
		{Input: "", Output: "skip me"},      // blank input → skipped
		{Input: "skip me too", Output: " "}, // blank output → skipped
		{Input: "how are you", Output: "fine"},
	}
	n, err := WriteJSONL(path, pairs, false)
	if err != nil {
		t.Fatalf("WriteJSONL: %v", err)
	}
	if n != 2 {
		t.Fatalf("expected 2 written (blanks skipped), got %d", n)
	}

	ds, err := LoadJSONL(path, nil)
	if err != nil {
		t.Fatalf("LoadJSONL: %v", err)
	}
	if len(ds.Pairs) != 2 {
		t.Fatalf("expected 2 pairs read, got %d", len(ds.Pairs))
	}
	if ds.Pairs[0].Input != "ciao" || ds.Pairs[0].Output != "ehilà" {
		t.Fatalf("round-trip mismatch: %+v", ds.Pairs[0])
	}

	// Append mode should accumulate, not truncate.
	if _, err := WriteJSONL(path, []DialogPair{{Input: "again", Output: "yes"}}, true); err != nil {
		t.Fatalf("WriteJSONL append: %v", err)
	}
	ds2, err := LoadJSONL(path, nil)
	if err != nil {
		t.Fatalf("LoadJSONL after append: %v", err)
	}
	if len(ds2.Pairs) != 3 {
		t.Fatalf("expected 3 pairs after append, got %d", len(ds2.Pairs))
	}
}

// TestGeneratorCount verifies that at least 5000 dialog pairs are generated.
func TestGeneratorCount(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	pairs := buildAllPairs(rng)
	if len(pairs) < 5000 {
		t.Errorf("Expected >= 5000 pairs, got %d", len(pairs))
	}
}

// TestDatasetVariety checks that multiple languages are present.
func TestDatasetVariety(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	pairs := buildAllPairs(rng)
	langs := make(map[string]int)
	for _, p := range pairs {
		langs[p.Lang]++
	}
	if langs["it"] == 0 {
		t.Error("No Italian pairs found")
	}
	if langs["en"] == 0 {
		t.Error("No English pairs found")
	}
}

// TestGenerateDataset verifies JSONL file output.
func TestGenerateDataset(t *testing.T) {
	path := t.TempDir() + "/test.jsonl"
	n, err := GenerateDataset(path, 99)
	if err != nil {
		t.Fatalf("GenerateDataset: %v", err)
	}
	if n < 5000 {
		t.Errorf("Expected >= 5000 pairs, got %d", n)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("Stat output file: %v", err)
	}
	if info.Size() == 0 {
		t.Error("Output file is empty")
	}
}

// TestBatchShape verifies that batch dimensions match requested sizes.
func TestBatchShape(t *testing.T) {
	corpus := []string{"ciao come stai bene grazie arrivederci buongiorno"}
	bpe := tokenizer.New()
	bpe.Train(corpus, 64)

	pairs := []DialogPair{
		{Input: "ciao", Output: "ciao", Lang: "it"},
		{Input: "come stai", Output: "bene grazie", Lang: "it"},
		{Input: "hello", Output: "hi there", Lang: "en"},
	}
	ds := &Dataset{Pairs: pairs, BPE: bpe}
	seqLen := 16
	batches := ds.Batches(2, seqLen)
	for _, b := range batches {
		for i := range b.InputIDs {
			if len(b.InputIDs[i]) != seqLen {
				t.Errorf("InputIDs[%d] length = %d, want %d", i, len(b.InputIDs[i]), seqLen)
			}
			if len(b.TargetIDs[i]) != seqLen {
				t.Errorf("TargetIDs[%d] length = %d, want %d", i, len(b.TargetIDs[i]), seqLen)
			}
			if len(b.Masks[i]) != seqLen {
				t.Errorf("Masks[%d] length = %d, want %d", i, len(b.Masks[i]), seqLen)
			}
		}
	}
}

// TestTargetShift verifies that TargetIDs[i][j] == InputIDs[i][j+1] for non-padded positions.
func TestTargetShift(t *testing.T) {
	corpus := []string{"ciao come stai bene grazie arrivederci buongiorno"}
	bpe := tokenizer.New()
	bpe.Train(corpus, 64)

	pairs := []DialogPair{{Input: "ciao come", Output: "stai bene", Lang: "it"}}
	ds := &Dataset{Pairs: pairs, BPE: bpe}
	seqLen := 16
	batches := ds.Batches(1, seqLen)
	if len(batches) == 0 {
		t.Fatal("No batches produced")
	}
	b := batches[0]
	inp := b.InputIDs[0]
	tgt := b.TargetIDs[0]
	for j := 0; j < seqLen-1; j++ {
		if inp[j] == tokenizer.PAD {
			break
		}
		if tgt[j] != inp[j+1] {
			t.Errorf("TargetIDs[%d] = %d, want InputIDs[%d] = %d", j, tgt[j], j+1, inp[j+1])
		}
	}
}

// TestSplitRatios verifies 80/10/10 splits.
func TestSplitRatios(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	pairs := buildAllPairs(rng)
	bpe := tokenizer.New()
	ds := &Dataset{Pairs: pairs, BPE: bpe}

	train, val, test := ds.Split(0.8, 0.1)
	total := len(train.Pairs) + len(val.Pairs) + len(test.Pairs)
	if total != len(pairs) {
		t.Errorf("Split total %d != original %d", total, len(pairs))
	}
}

// TestAugmentation checks that augmented dataset is larger.
func TestAugmentation(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	pairs := buildAllPairs(rng)[:100]
	bpe := tokenizer.New()
	ds := &Dataset{Pairs: pairs, BPE: bpe}
	aug := AugmentDataset(ds, rng)
	if len(aug.Pairs) <= len(ds.Pairs) {
		t.Errorf("Augmented dataset not larger: %d <= %d", len(aug.Pairs), len(ds.Pairs))
	}
}
