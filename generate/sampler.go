// Package generate provides human-like text generation on top of the transformer model.
package generate

import (
	"fmt"
	"hash/fnv"
	"math"
	"math/rand"
	"sort"
	"strings"
	"time"

	"awesomeProject/model"
	"awesomeProject/tokenizer"
)

// Config controls sampling behaviour and human-like post-processing.
type Config struct {
	Temperature       float64 // logit temperature (1.0 = no change)
	TopK              int     // 0 = disabled
	TopP              float64 // 1.0 = disabled
	RepetitionPenalty float64 // 1.0 = disabled, >1 reduces repetition
	MaxNewTokens      int     // max tokens to generate
	TypoRate          float64 // probability of introducing a typo per character
	UncertaintyRate   float64 // probability of prepending uncertainty phrase
}

// DefaultConfig returns human-like generation defaults.
func DefaultConfig() Config {
	return Config{
		Temperature:       0.8,
		TopK:              40,
		TopP:              0.9,
		RepetitionPenalty: 1.2,
		MaxNewTokens:      100,
		TypoRate:          0.02,
		UncertaintyRate:   0.05,
	}
}

// Sampler wraps a TransformerModel with sampling strategies and human-like effects.
type Sampler struct {
	Model  *model.TransformerModel
	BPE    *tokenizer.BPE
	Config Config
	RNG    *rand.Rand
}

// New returns a Sampler with the given configuration.
func New(m *model.TransformerModel, bpe *tokenizer.BPE, cfg Config) *Sampler {
	return &Sampler{
		Model:  m,
		BPE:    bpe,
		Config: cfg,
		RNG:    rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

// Generate produces a response to prompt, including thinking delay and post-processing.
func (s *Sampler) Generate(prompt string) (string, error) {
	// Encode prompt
	inputIDs := s.BPE.Encode(prompt)

	// Auto-regressive generation
	context := make([]int, len(inputIDs))
	copy(context, inputIDs)

	maxSeq := s.Model.Config.MaxSeqLen
	for range s.Config.MaxNewTokens {
		// Trim context to model's max seq len
		ctx := context
		if len(ctx) > maxSeq {
			ctx = ctx[len(ctx)-maxSeq:]
		}

		logits := s.Model.Forward(ctx)
		// Take logits at the last position
		lastRow := make([]float64, s.Model.Config.VocabSize)
		lastPos := logits.Rows - 1
		copy(lastRow, logits.Data[lastPos*logits.Cols:(lastPos+1)*logits.Cols])

		// Sample next token
		nextTok := s.sampleToken(lastRow, context)
		context = append(context, nextTok)

		if nextTok == tokenizer.EOS {
			break
		}
	}

	// Decode (skip input tokens)
	genTokens := context[len(inputIDs):]
	text := s.BPE.Decode(genTokens)

	// Post-processing pipeline
	text = s.injectUncertainty(text, hashString(prompt))
	text = s.insertEmoji(text)
	text = s.applyTypo(text)

	return text, nil
}

// sampleToken applies repetition penalty, temperature, top-k, top-p and samples.
func (s *Sampler) sampleToken(logits []float64, context []int) int {
	vocab := len(logits)

	// 1. Repetition penalty
	if s.Config.RepetitionPenalty != 1.0 {
		seen := make(map[int]bool)
		for _, id := range context {
			seen[id] = true
		}
		for id := range seen {
			if id < vocab {
				if logits[id] > 0 {
					logits[id] /= s.Config.RepetitionPenalty
				} else {
					logits[id] *= s.Config.RepetitionPenalty
				}
			}
		}
	}

	// 2. Temperature
	if s.Config.Temperature != 1.0 && s.Config.Temperature > 0 {
		for i := range logits {
			logits[i] /= s.Config.Temperature
		}
	}

	// 3. Top-K
	if s.Config.TopK > 0 && s.Config.TopK < vocab {
		// Find the K-th largest logit
		sorted := make([]float64, vocab)
		copy(sorted, logits)
		sort.Slice(sorted, func(i, j int) bool { return sorted[i] > sorted[j] })
		threshold := sorted[s.Config.TopK-1]
		for i, v := range logits {
			if v < threshold {
				logits[i] = -1e9
			}
		}
	}

	// 4. Softmax
	probs := softmax(logits)

	// 5. Top-P (nucleus)
	if s.Config.TopP < 1.0 {
		// Sort indices by probability descending
		indices := make([]int, vocab)
		for i := range indices {
			indices[i] = i
		}
		sort.Slice(indices, func(i, j int) bool { return probs[indices[i]] > probs[indices[j]] })

		cumsum := 0.0
		cutoff := -1
		for _, idx := range indices {
			cumsum += probs[idx]
			if cumsum >= s.Config.TopP {
				cutoff = idx
				break
			}
		}
		if cutoff >= 0 {
			threshold := probs[cutoff]
			for i, p := range probs {
				if p < threshold {
					probs[i] = 0
				}
			}
			// Renormalise
			sum := 0.0
			for _, p := range probs {
				sum += p
			}
			if sum > 0 {
				for i := range probs {
					probs[i] /= sum
				}
			}
		}
	}

	// 6. Sample via inverse CDF
	r := s.RNG.Float64()
	cumsum := 0.0
	for i, p := range probs {
		cumsum += p
		if r <= cumsum {
			return i
		}
	}
	return vocab - 1
}

// softmax converts logits to probabilities.
func softmax(logits []float64) []float64 {
	maxV := logits[0]
	for _, v := range logits {
		if v > maxV {
			maxV = v
		}
	}
	probs := make([]float64, len(logits))
	sum := 0.0
	for i, v := range logits {
		probs[i] = math.Exp(v - maxV)
		sum += probs[i]
	}
	for i := range probs {
		probs[i] /= sum
	}
	return probs
}

// ────────────────────────────────────────────────────────────────────────────
// Human-like post-processing
// ────────────────────────────────────────────────────────────────────────────

// qwertyAdjacent maps each key to its QWERTY neighbours for typo simulation.
var qwertyAdjacent = map[rune]string{
	'a': "sqwz", 'b': "vghn", 'c': "xdfv", 'd': "ersfxc", 'e': "rwsd",
	'f': "rtdgvc", 'g': "ytfhbv", 'h': "yugjbn", 'i': "uojk", 'j': "uihknm",
	'k': "iojlm", 'l': "opk", 'm': "njk", 'n': "bhjm", 'o': "ipkl",
	'p': "ol", 'q': "wa", 'r': "etdf", 's': "qwedxza", 't': "yrfg",
	'u': "yijh", 'v': "cfgb", 'w': "qase", 'x': "zsdc", 'y': "tugh",
	'z': "asx",
}

// applyTypo introduces random QWERTY typos to text.
func (s *Sampler) applyTypo(text string) string {
	runes := []rune(text)
	for i, r := range runes {
		if s.RNG.Float64() < s.Config.TypoRate {
			lower := rune(strings.ToLower(string(r))[0])
			if adj, ok := qwertyAdjacent[lower]; ok && len(adj) > 0 {
				replacement := rune(adj[s.RNG.Intn(len(adj))])
				runes[i] = replacement
			}
		}
	}
	return string(runes)
}

// italianUncertainty and englishUncertainty are uncertainty phrases injected at ~5%.
var italianUncertainty = []string{
	"forse ", "boh, ", "non so, ", "mah ", "tipo ", "diciamo ", "probabilmente ",
}
var englishUncertainty = []string{
	"maybe ", "idk, ", "not sure, ", "kinda ", "like ", "probably ", "I think ",
}

// injectUncertainty prepends an uncertainty phrase based on hash+rate.
func (s *Sampler) injectUncertainty(text string, hash uint64) string {
	if s.RNG.Float64() > s.Config.UncertaintyRate {
		return text
	}
	// Detect language by checking for Italian words
	lower := strings.ToLower(text)
	isItalian := strings.ContainsAny(lower, "àèéìíîòóùú") ||
		strings.Contains(lower, " il ") || strings.Contains(lower, " la ") ||
		strings.Contains(lower, " che ") || strings.Contains(lower, " un ")

	var phrases []string
	if isItalian {
		phrases = italianUncertainty
	} else {
		phrases = englishUncertainty
	}
	idx := int(hash) % len(phrases)
	if idx < 0 {
		idx = -idx
	}
	return phrases[idx] + text
}

// sentimentEmoji maps sentiment signals to emoji.
var sentimentEmoji = map[string]string{
	"felice": "😊", "happy": "😊", "bello": "✨", "great": "🎉",
	"triste": "😢", "sad": "😢", "grazie": "🙏", "thanks": "🙏",
	"amore": "❤️", "love": "❤️", "ridere": "😂", "funny": "😂",
	"lol": "😂", "lmao": "😂", "ottimo": "👍", "perfect": "👍",
	"pizza": "🍕", "cibo": "🍔", "food": "🍔", "musica": "🎵",
	"music": "🎵", "game": "🎮", "gioco": "🎮", "fuoco": "🔥",
}

// insertEmoji appends a contextual emoji based on sentiment keywords.
func (s *Sampler) insertEmoji(text string) string {
	lower := strings.ToLower(text)
	for keyword, emoji := range sentimentEmoji {
		if strings.Contains(lower, keyword) {
			if s.RNG.Float64() < 0.4 {
				return text + " " + emoji
			}
			break
		}
	}
	return text
}

// hashString computes a stable hash for a string (used for variability seeding).
func hashString(s string) uint64 {
	h := fnv.New64a()
	_, _ = h.Write([]byte(s))
	return h.Sum64()
}

// ThinkingDelay returns a simulated typing delay proportional to response length.
// Call time.Sleep on the returned duration before sending the response.
func ThinkingDelay(responseLen int) time.Duration {
	// 30–80ms per word (simulates typing at ~100 WPM)
	words := responseLen / 5
	if words < 1 {
		words = 1
	}
	ms := words*50 + 500 // base 500ms + 50ms per word
	if ms > 4000 {
		ms = 4000
	}
	return time.Duration(ms) * time.Millisecond
}

// ResponseVariability ensures no identical response is sent twice in a session.
// seenHashes tracks hashes of already-sent responses.
func ResponseVariability(response string, seenHashes map[uint64]bool) (string, bool) {
	h := hashString(response)
	if seenHashes[h] {
		return response, false // duplicate
	}
	seenHashes[h] = true
	return response, true
}

// Format ensures responses end with punctuation.
func Format(text string) string {
	text = strings.TrimSpace(text)
	if len(text) == 0 {
		return text
	}
	last := rune(text[len(text)-1])
	if last != '.' && last != '!' && last != '?' && last != '~' {
		// Add ellipsis for casual Discord style sometimes
		text += "."
	}
	return fmt.Sprintf("%s", text)
}
