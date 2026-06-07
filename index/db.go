// Package index provides persistent message storage and semantic retrieval so the
// bot can learn continuously from live Discord chat instead of a static dataset.
package index

import (
	"database/sql"
	"encoding/binary"
	"fmt"
	"math"
	"time"

	// Pure-Go SQLite driver (no CGo). Registers the "sqlite" driver.
	_ "modernc.org/sqlite"
)

// dbTimeLayout is a fixed-width timestamp layout so DATETIME string ordering is correct.
const dbTimeLayout = "2006-01-02 15:04:05.000000000"

// MessageDB is the persistent SQLite-backed store for indexed chat messages.
type MessageDB struct {
	db *sql.DB
}

// NewMessageDB opens (or creates) the SQLite database at path and ensures the schema exists.
func NewMessageDB(path string) (*MessageDB, error) {
	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, fmt.Errorf("open sqlite %s: %w", path, err)
	}
	// SQLite with database/sql: a single connection avoids "database is locked" under
	// concurrent writers from the bot and online trainer goroutines.
	db.SetMaxOpenConns(1)

	mdb := &MessageDB{db: db}
	if err := mdb.migrate(); err != nil {
		_ = db.Close()
		return nil, err
	}
	return mdb, nil
}

// migrate creates the messages table and index if they do not already exist.
func (m *MessageDB) migrate() error {
	const schema = `
	CREATE TABLE IF NOT EXISTS messages (
		id          TEXT PRIMARY KEY,
		channel_id  TEXT NOT NULL,
		author_id   TEXT NOT NULL,
		content     TEXT NOT NULL,
		tokens      BLOB,
		embedding   BLOB,
		reply_to    TEXT,
		created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
	);
	CREATE INDEX IF NOT EXISTS idx_channel ON messages(channel_id, created_at);`
	if _, err := m.db.Exec(schema); err != nil {
		return fmt.Errorf("migrate schema: %w", err)
	}
	return nil
}

// Insert stores (or replaces) a message record.
func (m *MessageDB) Insert(r MessageRecord) error {
	ts := r.Timestamp
	if ts.IsZero() {
		ts = time.Now()
	}
	_, err := m.db.Exec(
		`INSERT OR REPLACE INTO messages
		 (id, channel_id, author_id, content, tokens, embedding, reply_to, created_at)
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
		r.ID, r.ChannelID, r.AuthorID, r.Content,
		intsToBlob(r.Tokens), floatsToBlob(r.Embedding), r.ReplyToID,
		ts.UTC().Format(dbTimeLayout),
	)
	if err != nil {
		return fmt.Errorf("insert message %s: %w", r.ID, err)
	}
	return nil
}

// GetByChannel returns up to limit most-recent messages for a channel, ordered oldest→newest.
func (m *MessageDB) GetByChannel(channelID string, limit int) ([]MessageRecord, error) {
	rows, err := m.db.Query(
		`SELECT id, channel_id, author_id, content, tokens, embedding, reply_to, created_at
		 FROM messages WHERE channel_id = ?
		 ORDER BY created_at DESC LIMIT ?`,
		channelID, limit,
	)
	if err != nil {
		return nil, fmt.Errorf("query channel %s: %w", channelID, err)
	}
	defer rows.Close()

	recs, err := scanRecords(rows)
	if err != nil {
		return nil, err
	}
	// Reverse to ascending chronological order.
	for i, j := 0, len(recs)-1; i < j; i, j = i+1, j-1 {
		recs[i], recs[j] = recs[j], recs[i]
	}
	return recs, nil
}

// Count returns the total number of stored messages.
func (m *MessageDB) Count() (int, error) {
	var n int
	if err := m.db.QueryRow(`SELECT COUNT(*) FROM messages`).Scan(&n); err != nil {
		return 0, fmt.Errorf("count: %w", err)
	}
	return n, nil
}

// GetAll returns every stored message ordered oldest→newest.
func (m *MessageDB) GetAll() ([]MessageRecord, error) {
	rows, err := m.db.Query(
		`SELECT id, channel_id, author_id, content, tokens, embedding, reply_to, created_at
		 FROM messages ORDER BY created_at ASC`,
	)
	if err != nil {
		return nil, fmt.Errorf("query all: %w", err)
	}
	defer rows.Close()
	return scanRecords(rows)
}

// Clear deletes all stored messages (used by the /forget admin command).
func (m *MessageDB) Clear() error {
	if _, err := m.db.Exec(`DELETE FROM messages`); err != nil {
		return fmt.Errorf("clear: %w", err)
	}
	return nil
}

// Close releases the underlying database handle.
func (m *MessageDB) Close() error {
	return m.db.Close()
}

// scanRecords materialises SQL rows into MessageRecord values.
func scanRecords(rows *sql.Rows) ([]MessageRecord, error) {
	var out []MessageRecord
	for rows.Next() {
		var (
			r          MessageRecord
			tokensBlob []byte
			embedBlob  []byte
			replyTo    sql.NullString
			createdAt  string
		)
		if err := rows.Scan(&r.ID, &r.ChannelID, &r.AuthorID, &r.Content,
			&tokensBlob, &embedBlob, &replyTo, &createdAt); err != nil {
			return nil, fmt.Errorf("scan row: %w", err)
		}
		r.Tokens = blobToInts(tokensBlob)
		r.Embedding = blobToFloats(embedBlob)
		if replyTo.Valid {
			r.ReplyToID = replyTo.String
		}
		if t, err := time.Parse(dbTimeLayout, createdAt); err == nil {
			r.Timestamp = t
		}
		out = append(out, r)
	}
	return out, rows.Err()
}

// ────────────────────────────────────────────────────────────────────────────
// BLOB (de)serialisation — little-endian, length-prefixed
// ────────────────────────────────────────────────────────────────────────────

// intsToBlob encodes a slice of ints as int64 little-endian bytes.
func intsToBlob(xs []int) []byte {
	buf := make([]byte, 8*len(xs))
	for i, x := range xs {
		binary.LittleEndian.PutUint64(buf[i*8:], uint64(int64(x)))
	}
	return buf
}

// blobToInts decodes bytes produced by intsToBlob.
func blobToInts(b []byte) []int {
	n := len(b) / 8
	xs := make([]int, n)
	for i := 0; i < n; i++ {
		xs[i] = int(int64(binary.LittleEndian.Uint64(b[i*8:])))
	}
	return xs
}

// floatsToBlob encodes a slice of float64 as IEEE-754 little-endian bytes.
func floatsToBlob(xs []float64) []byte {
	buf := make([]byte, 8*len(xs))
	for i, x := range xs {
		binary.LittleEndian.PutUint64(buf[i*8:], math.Float64bits(x))
	}
	return buf
}

// blobToFloats decodes bytes produced by floatsToBlob.
func blobToFloats(b []byte) []float64 {
	n := len(b) / 8
	xs := make([]float64, n)
	for i := 0; i < n; i++ {
		xs[i] = math.Float64frombits(binary.LittleEndian.Uint64(b[i*8:]))
	}
	return xs
}
