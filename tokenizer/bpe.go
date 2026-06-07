// Package tokenizer implements a Byte-Pair Encoding tokenizer trained from scratch.
package tokenizer

import (
	"container/heap"
	"strings"
	"unicode/utf8"
)

// Special token IDs.
const (
	PAD = 0
	BOS = 1
	EOS = 2
	UNK = 3
)

// Special token strings.
const (
	PadToken = "[PAD]"
	BosToken = "[BOS]"
	EosToken = "[EOS]"
	UnkToken = "[UNK]"
)

// Merge records one BPE merge rule: A + B → Result.
type Merge struct {
	A, B, Result string
}

// BPE is a Byte-Pair Encoding tokenizer.
type BPE struct {
	Vocab     map[string]int // token string → ID
	InvVocab  map[int]string // ID → token string
	MergeRank map[[2]string]int // pair → merge rank (lower = applied first)
	Merges    []Merge
	VocabSize int
}

// New returns an empty BPE with the four special tokens pre-registered.
func New() *BPE {
	b := &BPE{
		Vocab:     make(map[string]int),
		InvVocab:  make(map[int]string),
		MergeRank: make(map[[2]string]int),
	}
	for id, tok := range []string{PadToken, BosToken, EosToken, UnkToken} {
		b.Vocab[tok] = id
		b.InvVocab[id] = tok
	}
	return b
}

// ────────────────────────────────────────────────────────────────────────────
// Training
// ────────────────────────────────────────────────────────────────────────────

// wordSplit represents one word as a list of symbol strings.
type wordSplit struct {
	symbols []string
	count   int
}

// Train learns BPE merge rules from corpus until vocabSize is reached.
// corpus is a slice of raw text strings.
func (b *BPE) Train(corpus []string, vocabSize int) {
	b.VocabSize = vocabSize

	// Step 1: collect character vocabulary from corpus
	charFreq := make(map[string]int)
	for _, text := range corpus {
		for _, word := range strings.Fields(text) {
			chars := splitWord(word)
			for _, ch := range chars {
				charFreq[ch]++
			}
		}
	}
	// Seed vocab with special tokens (IDs 0-3 already set) then characters
	nextID := 4
	for ch := range charFreq {
		if _, exists := b.Vocab[ch]; !exists {
			b.Vocab[ch] = nextID
			b.InvVocab[nextID] = ch
			nextID++
		}
	}

	// Step 2: build word frequency table with character splits
	wordFreq := make(map[string]*wordSplit)
	for _, text := range corpus {
		for _, word := range strings.Fields(text) {
			key := word
			if _, exists := wordFreq[key]; !exists {
				wordFreq[key] = &wordSplit{symbols: splitWord(word), count: 0}
			}
			wordFreq[key].count++
		}
	}

	// Step 3: iteratively merge most-frequent pairs
	for nextID < vocabSize {
		pairFreq := countPairs(wordFreq)
		if len(pairFreq) == 0 {
			break
		}

		// Find best pair using priority queue
		best := bestPair(pairFreq)
		merged := best[0] + best[1]

		// Register new token
		if _, exists := b.Vocab[merged]; !exists {
			b.Vocab[merged] = nextID
			b.InvVocab[nextID] = merged
			nextID++
		}

		rank := len(b.Merges)
		b.Merges = append(b.Merges, Merge{A: best[0], B: best[1], Result: merged})
		b.MergeRank[[2]string{best[0], best[1]}] = rank

		// Apply merge to all words
		for _, ws := range wordFreq {
			ws.symbols = applyMerge(ws.symbols, best[0], best[1], merged)
		}
	}
	b.VocabSize = len(b.Vocab)
}

// splitWord converts a word to a list of UTF-8 characters, appending </w> to the last.
func splitWord(word string) []string {
	chars := make([]string, 0, len(word)+1)
	for len(word) > 0 {
		r, size := utf8.DecodeRuneInString(word)
		chars = append(chars, string(r))
		word = word[size:]
	}
	if len(chars) > 0 {
		chars[len(chars)-1] += "</w>"
	}
	return chars
}

// countPairs tallies adjacent symbol pairs weighted by word frequency.
func countPairs(wordFreq map[string]*wordSplit) map[[2]string]int {
	freq := make(map[[2]string]int)
	for _, ws := range wordFreq {
		syms := ws.symbols
		for i := 0; i < len(syms)-1; i++ {
			freq[[2]string{syms[i], syms[i+1]}] += ws.count
		}
	}
	return freq
}

// pairHeap implements heap.Interface for pair-frequency items.
type pairItem struct {
	pair  [2]string
	count int
}
type pairHeap []pairItem

func (h pairHeap) Len() int            { return len(h) }
func (h pairHeap) Less(i, j int) bool  { return h[i].count > h[j].count } // max-heap
func (h pairHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *pairHeap) Push(x interface{}) { *h = append(*h, x.(pairItem)) }
func (h *pairHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

// bestPair returns the most frequent adjacent pair.
func bestPair(pairFreq map[[2]string]int) [2]string {
	h := make(pairHeap, 0, len(pairFreq))
	for pair, cnt := range pairFreq {
		h = append(h, pairItem{pair, cnt})
	}
	heap.Init(&h)
	return heap.Pop(&h).(pairItem).pair
}

// applyMerge replaces all adjacent (a, b) occurrences in syms with merged.
func applyMerge(syms []string, a, b, merged string) []string {
	out := make([]string, 0, len(syms))
	i := 0
	for i < len(syms) {
		if i < len(syms)-1 && syms[i] == a && syms[i+1] == b {
			out = append(out, merged)
			i += 2
		} else {
			out = append(out, syms[i])
			i++
		}
	}
	return out
}

// ────────────────────────────────────────────────────────────────────────────
// Encode / Decode
// ────────────────────────────────────────────────────────────────────────────

// Encode converts text to a token ID sequence wrapped in [BOS] and [EOS].
func (b *BPE) Encode(text string) []int {
	ids := []int{BOS}
	for _, word := range strings.Fields(text) {
		syms := splitWord(word)
		syms = b.applyMerges(syms)
		for _, sym := range syms {
			id, ok := b.Vocab[sym]
			if !ok {
				id = UNK
			}
			ids = append(ids, id)
		}
	}
	ids = append(ids, EOS)
	return ids
}

// EncodeNoSpecial encodes text without BOS/EOS tokens.
func (b *BPE) EncodeNoSpecial(text string) []int {
	var ids []int
	for _, word := range strings.Fields(text) {
		syms := splitWord(word)
		syms = b.applyMerges(syms)
		for _, sym := range syms {
			id, ok := b.Vocab[sym]
			if !ok {
				id = UNK
			}
			ids = append(ids, id)
		}
	}
	return ids
}

// applyMerges applies all learned merge rules to a symbol list in rank order.
func (b *BPE) applyMerges(syms []string) []string {
	for _, merge := range b.Merges {
		syms = applyMerge(syms, merge.A, merge.B, merge.Result)
		if len(syms) == 1 {
			break
		}
	}
	return syms
}

// Decode converts token IDs back to text, stripping special tokens and </w> markers.
func (b *BPE) Decode(ids []int) string {
	var parts []string
	for _, id := range ids {
		if id == PAD || id == BOS || id == EOS {
			continue
		}
		tok, ok := b.InvVocab[id]
		if !ok {
			tok = UnkToken
		}
		parts = append(parts, tok)
	}
	// Join, then replace </w> markers with spaces
	joined := strings.Join(parts, "")
	joined = strings.ReplaceAll(joined, "</w>", " ")
	return strings.TrimSpace(joined)
}

// VocabLen returns the current vocabulary size.
func (b *BPE) VocabLen() int { return len(b.Vocab) }
