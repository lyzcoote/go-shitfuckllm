package index

import (
	"log"
	"math"
	"sort"
	"sync"
	"time"

	"awesomeProject/model"
	"awesomeProject/tokenizer"
	"awesomeProject/train"
)

// minTrainTokens is the minimum token length for a message to become a training signal.
const minTrainTokens = 5

// defaultRingSize is the capacity of the in-memory recent-message ring buffer.
const defaultRingSize = 1000

// MessageRecord is a single indexed Discord message with its derived NLP artifacts.
type MessageRecord struct {
	ID        string    // Discord snowflake
	ChannelID string    // channel the message belongs to
	AuthorID  string    // author snowflake
	Content   string    // raw text
	Tokens    []int     // BPE token IDs
	Timestamp time.Time // creation time
	ReplyToID string    // referenced message ID, if this is a reply
	Embedding []float64 // semantic vector (dModel dims)
}

// MessageIndex maintains a ring buffer of recent messages plus a persistent DB,
// computes semantic embeddings, and feeds valid messages to the online trainer.
type MessageIndex struct {
	db        *MessageDB
	recent    []MessageRecord
	maxRecent int

	bpe     *tokenizer.BPE
	model   *model.TransformerModel
	modelMu *sync.Mutex // guards model weights shared with the online trainer

	trainCh chan<- train.TrainingPair // optional; nil disables training emission

	mu sync.RWMutex
}

// New constructs a MessageIndex. modelMu must be the same mutex the online trainer
// uses to guard weight updates; trainCh may be nil to disable training emission.
func New(db *MessageDB, bpe *tokenizer.BPE, m *model.TransformerModel, modelMu *sync.Mutex, trainCh chan<- train.TrainingPair) *MessageIndex {
	return &MessageIndex{
		db:        db,
		recent:    make([]MessageRecord, 0, defaultRingSize),
		maxRecent: defaultRingSize,
		bpe:       bpe,
		model:     m,
		modelMu:   modelMu,
		trainCh:   trainCh,
	}
}

// Add tokenises and embeds a message, stores it in the ring buffer and DB, and—when
// the message carries enough signal—emits a conversational training pair to the online
// trainer. Use this for live messages.
func (idx *MessageIndex) Add(msg MessageRecord) { idx.add(msg, true) }

// Ingest indexes a message (tokenise + embed + persist) WITHOUT emitting an online
// training pair. Use this for bulk backfill of channel history, where you train in one
// batch afterwards instead of streaming thousands of pairs through the online channel.
func (idx *MessageIndex) Ingest(msg MessageRecord) { idx.add(msg, false) }

// add is the shared implementation behind Add (emit=true) and Ingest (emit=false).
func (idx *MessageIndex) add(msg MessageRecord, emit bool) {
	if len(msg.Tokens) == 0 {
		msg.Tokens = idx.bpe.EncodeNoSpecial(msg.Content)
	}
	if msg.Timestamp.IsZero() {
		msg.Timestamp = time.Now()
	}
	if len(msg.Embedding) == 0 {
		msg.Embedding = idx.Embed(msg.Tokens)
	}

	// Find the previous message in the same channel (for conversational pairing)
	// before inserting this one, then append to the ring buffer.
	idx.mu.Lock()
	prevContent := idx.lastInChannelLocked(msg.ChannelID)
	idx.recent = append(idx.recent, msg)
	if len(idx.recent) > idx.maxRecent {
		idx.recent = idx.recent[len(idx.recent)-idx.maxRecent:]
	}
	idx.mu.Unlock()

	// Persist (best-effort: log but never crash the bot on a DB hiccup).
	if idx.db != nil {
		if err := idx.db.Insert(msg); err != nil {
			log.Printf("index: persist message %s failed: %v", msg.ID, err)
		}
	}

	// Emit a training pair only when there is real conversational context and the
	// message is substantial. (prev → curr) is genuine human dialogue signal.
	if emit && idx.trainCh != nil && prevContent != "" && len(msg.Tokens) > minTrainTokens {
		select {
		case idx.trainCh <- train.TrainingPair{Input: prevContent, Output: msg.Content}:
		default: // trainer busy/full — drop rather than block the handler
		}
	}
}

// lastInChannelLocked returns the most recent buffered message content for channelID.
// Caller must hold idx.mu.
func (idx *MessageIndex) lastInChannelLocked(channelID string) string {
	for i := len(idx.recent) - 1; i >= 0; i-- {
		if idx.recent[i].ChannelID == channelID {
			return idx.recent[i].Content
		}
	}
	return ""
}

// GetContext returns the last n messages of a channel ordered oldest→newest.
func (idx *MessageIndex) GetContext(channelID string, n int) []MessageRecord {
	idx.mu.RLock()
	var filtered []MessageRecord
	for _, r := range idx.recent {
		if r.ChannelID == channelID {
			filtered = append(filtered, r)
		}
	}
	idx.mu.RUnlock()

	sort.Slice(filtered, func(i, j int) bool {
		return filtered[i].Timestamp.Before(filtered[j].Timestamp)
	})
	if len(filtered) > n {
		filtered = filtered[len(filtered)-n:]
	}
	return filtered
}

// FindSimilar returns the topK buffered messages most similar to the query embedding
// (cosine similarity), highest first. Used for RAG-style context retrieval.
func (idx *MessageIndex) FindSimilar(embedding []float64, topK int) []MessageRecord {
	idx.mu.RLock()
	snapshot := make([]MessageRecord, len(idx.recent))
	copy(snapshot, idx.recent)
	idx.mu.RUnlock()

	type scored struct {
		rec   MessageRecord
		score float64
	}
	scoredRecs := make([]scored, 0, len(snapshot))
	for _, r := range snapshot {
		if len(r.Embedding) == 0 {
			continue
		}
		scoredRecs = append(scoredRecs, scored{r, CosineSimilarity(embedding, r.Embedding)})
	}
	sort.Slice(scoredRecs, func(i, j int) bool {
		return scoredRecs[i].score > scoredRecs[j].score
	})
	if len(scoredRecs) > topK {
		scoredRecs = scoredRecs[:topK]
	}
	out := make([]MessageRecord, len(scoredRecs))
	for i, s := range scoredRecs {
		out[i] = s.rec
	}
	return out
}

// ExportPairs extracts consecutive (input, output) message pairs from every channel
// thread that contains at least minMessages messages. Reads the full DB history when
// available, falling back to the in-memory ring buffer.
func (idx *MessageIndex) ExportPairs(minMessages int) []train.TrainingPair {
	var all []MessageRecord
	if idx.db != nil {
		if recs, err := idx.db.GetAll(); err == nil {
			all = recs
		} else {
			log.Printf("index: ExportPairs GetAll failed: %v", err)
		}
	}
	if len(all) == 0 {
		idx.mu.RLock()
		all = make([]MessageRecord, len(idx.recent))
		copy(all, idx.recent)
		idx.mu.RUnlock()
	}

	// Group by channel, preserving chronological order.
	byChannel := make(map[string][]MessageRecord)
	for _, r := range all {
		byChannel[r.ChannelID] = append(byChannel[r.ChannelID], r)
	}

	var pairs []train.TrainingPair
	for _, msgs := range byChannel {
		if len(msgs) < minMessages {
			continue
		}
		sort.Slice(msgs, func(i, j int) bool {
			return msgs[i].Timestamp.Before(msgs[j].Timestamp)
		})
		for i := 0; i+1 < len(msgs); i++ {
			pairs = append(pairs, train.TrainingPair{
				Input:  msgs[i].Content,
				Output: msgs[i+1].Content,
			})
		}
	}
	return pairs
}

// Embed computes a semantic vector for a token sequence as the mean of its token
// embeddings (a bag-of-embeddings sentence vector of dimension dModel). It reads the
// shared embedding table under modelMu so it stays consistent with online updates.
func (idx *MessageIndex) Embed(tokens []int) []float64 {
	d := idx.model.Embed.DModel
	vocab := idx.model.Embed.VocabSize
	emb := make([]float64, d)
	if len(tokens) == 0 {
		return emb
	}

	idx.lockModel()
	te := idx.model.Embed.TokenEmbed.Data
	for _, tok := range tokens {
		if tok < 0 || tok >= vocab {
			tok = tokenizer.UNK
		}
		base := tok * d
		for k := 0; k < d; k++ {
			emb[k] += te[base+k]
		}
	}
	idx.unlockModel()

	inv := 1.0 / float64(len(tokens))
	for k := range emb {
		emb[k] *= inv
	}
	return emb
}

// EmbedText is a convenience wrapper that tokenises text and embeds it.
func (idx *MessageIndex) EmbedText(text string) []float64 {
	return idx.Embed(idx.bpe.EncodeNoSpecial(text))
}

// Warm preloads up to limit most-recent messages from the DB into the ring buffer,
// so RAG and context retrieval work immediately after a restart.
func (idx *MessageIndex) Warm(limit int) error {
	if idx.db == nil {
		return nil
	}
	all, err := idx.db.GetAll()
	if err != nil {
		return err
	}
	if len(all) > limit {
		all = all[len(all)-limit:]
	}
	idx.mu.Lock()
	idx.recent = all
	if cap(idx.recent) < idx.maxRecent {
		grown := make([]MessageRecord, len(all), idx.maxRecent)
		copy(grown, all)
		idx.recent = grown
	}
	idx.mu.Unlock()
	return nil
}

// Forget clears the in-memory ring buffer and wipes the persistent store.
func (idx *MessageIndex) Forget() error {
	idx.mu.Lock()
	idx.recent = idx.recent[:0]
	idx.mu.Unlock()
	if idx.db != nil {
		return idx.db.Clear()
	}
	return nil
}

// Count returns the number of persisted messages (0 if there is no DB).
func (idx *MessageIndex) Count() (int, error) {
	if idx.db == nil {
		return idx.RecentLen(), nil
	}
	return idx.db.Count()
}

// RecentLen returns the number of messages currently in the ring buffer.
func (idx *MessageIndex) RecentLen() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return len(idx.recent)
}

// lockModel / unlockModel guard the shared model weights; they no-op if no mutex was set.
func (idx *MessageIndex) lockModel() {
	if idx.modelMu != nil {
		idx.modelMu.Lock()
	}
}

func (idx *MessageIndex) unlockModel() {
	if idx.modelMu != nil {
		idx.modelMu.Unlock()
	}
}

// CosineSimilarity returns the cosine similarity of two equal-length vectors in [-1, 1].
// Returns 0 when either vector is zero-length or has zero magnitude.
func CosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, na, nb float64
	for i := range a {
		dot += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}
