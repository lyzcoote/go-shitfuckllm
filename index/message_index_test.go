package index

import (
	"fmt"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"awesomeProject/model"
	"awesomeProject/tokenizer"
)

// testIndex builds a MessageIndex backed by a temp SQLite DB and a tiny model/BPE.
func testIndex(t *testing.T) (*MessageIndex, func()) {
	t.Helper()
	db, err := NewMessageDB(filepath.Join(t.TempDir(), "test.db"))
	if err != nil {
		t.Fatalf("open db: %v", err)
	}
	bpe := tokenizer.New()
	bpe.Train([]string{"message number hello world ciao come stai bene grazie turn content here foo bar baz"}, 80)
	cfg := &model.TransformerConfig{VocabSize: bpe.VocabLen(), DModel: 16, NHeads: 2, NLayers: 1, DFF: 32, MaxSeqLen: 32}
	model.SetSeed(1)
	m := model.NewTransformerModel(cfg)
	idx := New(db, bpe, m, &sync.Mutex{}, nil)
	return idx, func() { _ = db.Close() }
}

// TestAddAndRetrieve indexes 100 messages and checks GetContext returns the most recent.
func TestAddAndRetrieve(t *testing.T) {
	idx, cleanup := testIndex(t)
	defer cleanup()

	base := time.Now()
	for i := 0; i < 100; i++ {
		idx.Add(MessageRecord{
			ID:        fmt.Sprintf("m%d", i),
			ChannelID: "c1",
			AuthorID:  "u",
			Content:   fmt.Sprintf("message number %d hello", i),
			Timestamp: base.Add(time.Duration(i) * time.Second),
		})
	}

	ctx := idx.GetContext("c1", 10)
	if len(ctx) != 10 {
		t.Fatalf("GetContext returned %d messages, want 10", len(ctx))
	}
	if ctx[len(ctx)-1].ID != "m99" {
		t.Errorf("most recent = %s, want m99", ctx[len(ctx)-1].ID)
	}
	if ctx[0].ID != "m90" {
		t.Errorf("oldest of window = %s, want m90", ctx[0].ID)
	}
	// Verify chronological ordering.
	for i := 1; i < len(ctx); i++ {
		if ctx[i].Timestamp.Before(ctx[i-1].Timestamp) {
			t.Errorf("messages not ordered by timestamp at %d", i)
		}
	}
}

// TestCosineSimilarity verifies similar vectors score high and opposite vectors score low.
func TestCosineSimilarity(t *testing.T) {
	similar := CosineSimilarity([]float64{1, 1, 0, 0}, []float64{1, 1, 0.1, 0})
	if similar <= 0.7 {
		t.Errorf("similar vectors cosine = %.4f, want > 0.7", similar)
	}

	opposite := CosineSimilarity([]float64{1, 0, 0}, []float64{-1, 0, 0})
	if opposite >= 0.3 {
		t.Errorf("opposite vectors cosine = %.4f, want < 0.3", opposite)
	}

	orthogonal := CosineSimilarity([]float64{1, 0}, []float64{0, 1})
	if orthogonal >= 0.3 {
		t.Errorf("orthogonal vectors cosine = %.4f, want < 0.3", orthogonal)
	}

	if z := CosineSimilarity([]float64{0, 0}, []float64{1, 1}); z != 0 {
		t.Errorf("zero vector cosine = %.4f, want 0", z)
	}
}

// TestExportPairs verifies consecutive pairs emerge from a thread of 20 messages.
func TestExportPairs(t *testing.T) {
	idx, cleanup := testIndex(t)
	defer cleanup()

	base := time.Now()
	for i := 0; i < 20; i++ {
		author := "A"
		if i%2 == 1 {
			author = "B"
		}
		idx.Add(MessageRecord{
			ID:        fmt.Sprintf("m%d", i),
			ChannelID: "c1",
			AuthorID:  author,
			Content:   fmt.Sprintf("turn %d content here", i),
			Timestamp: base.Add(time.Duration(i) * time.Second),
		})
	}

	pairs := idx.ExportPairs(2)
	if len(pairs) < 9 {
		t.Errorf("ExportPairs returned %d pairs, want >= 9", len(pairs))
	}
	// A channel with fewer than minMessages must yield nothing.
	idx.Add(MessageRecord{ID: "solo", ChannelID: "c2", AuthorID: "A", Content: "lonely message", Timestamp: base})
	only := 0
	for _, p := range idx.ExportPairs(5) {
		_ = p
		only++
	}
	// c1 has 20 (>=5) → 19 pairs; c2 has 1 (<5) → 0 pairs.
	if only != 19 {
		t.Errorf("ExportPairs(min=5) returned %d pairs, want 19 (c2 excluded)", only)
	}
}
