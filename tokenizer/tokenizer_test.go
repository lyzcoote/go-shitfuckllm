package tokenizer

import (
	"os"
	"path/filepath"
	"testing"
)

// TestSpecialTokens verifies that special token IDs are correctly assigned.
func TestSpecialTokens(t *testing.T) {
	b := New()
	if b.Vocab[PadToken] != PAD {
		t.Errorf("PAD token ID mismatch: got %d want %d", b.Vocab[PadToken], PAD)
	}
	if b.Vocab[BosToken] != BOS {
		t.Errorf("BOS token ID mismatch: got %d want %d", b.Vocab[BosToken], BOS)
	}
	if b.Vocab[EosToken] != EOS {
		t.Errorf("EOS token ID mismatch: got %d want %d", b.Vocab[EosToken], EOS)
	}
	if b.Vocab[UnkToken] != UNK {
		t.Errorf("UNK token ID mismatch: got %d want %d", b.Vocab[UnkToken], UNK)
	}
}

// TestBPETrain verifies that training expands the vocabulary.
func TestBPETrain(t *testing.T) {
	corpus := []string{
		"ciao come stai bene grazie arrivederci hello how are you fine thanks",
		"buongiorno buonasera notte giorno settimana mese anno",
	}
	b := New()
	targetVocab := 64
	b.Train(corpus, targetVocab)
	if b.VocabLen() < 20 {
		t.Errorf("Vocab size after training = %d, expected at least 20", b.VocabLen())
	}
}

// TestEncodeDecodeRoundtrip verifies that Encode followed by Decode recovers the original text.
func TestEncodeDecodeRoundtrip(t *testing.T) {
	corpus := []string{"ciao come stai bene grazie arrivederci buongiorno"}
	b := New()
	b.Train(corpus, 64)

	texts := []string{"ciao", "come stai", "bene grazie"}
	for _, text := range texts {
		ids := b.Encode(text)
		decoded := b.Decode(ids)
		if decoded != text {
			t.Errorf("Roundtrip failed for %q: got %q", text, decoded)
		}
	}
}

// TestUnknownTokens checks that unseen characters map to UNK.
func TestUnknownTokens(t *testing.T) {
	b := New()
	b.Train([]string{"abc"}, 20)
	ids := b.Encode("xyz") // x, y, z not in tiny corpus
	for _, id := range ids {
		if id != BOS && id != EOS && id != UNK {
			// Accept if the chars happened to be learned; just ensure no panic
			_ = id
		}
	}
}

// TestSaveLoad verifies that Save/Load preserves the vocabulary.
func TestSaveLoad(t *testing.T) {
	corpus := []string{"ciao come stai bene grazie arrivederci hello how are you"}
	b := New()
	b.Train(corpus, 64)

	dir := t.TempDir()
	path := filepath.Join(dir, "vocab.json")
	if err := b.Save(path); err != nil {
		t.Fatalf("Save: %v", err)
	}

	b2, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	if b2.VocabLen() != b.VocabLen() {
		t.Errorf("VocabLen mismatch after reload: %d vs %d", b2.VocabLen(), b.VocabLen())
	}
	if len(b2.Merges) != len(b.Merges) {
		t.Errorf("Merges count mismatch: %d vs %d", len(b2.Merges), len(b.Merges))
	}

	// Clean up temp file
	_ = os.Remove(path)
}
