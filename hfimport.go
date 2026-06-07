package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"

	"awesomeProject/dataset"
)

// hfRowsEndpoint is the public Hugging Face datasets-server pagination API. It serves any
// public dataset as JSON rows (no Python / datasets library needed), 100 rows per request.
const hfRowsEndpoint = "https://datasets-server.huggingface.co/rows"

// hfPageSize is the max rows the datasets-server returns per request.
const hfPageSize = 100

// hfRowsResponse is the subset of the datasets-server /rows payload we consume.
type hfRowsResponse struct {
	Rows []struct {
		Row map[string]json.RawMessage `json:"row"`
	} `json:"rows"`
	NumRowsTotal int `json:"num_rows_total"`
}

// runImportHF downloads a Hugging Face dataset over the public datasets-server HTTP API
// and converts it into a JSONL training dataset on disk (same format as harvest/synthetic).
//
// Two shapes are supported:
//   - Single text column (textCol set): consecutive rows become (input, output) pairs, so
//     the model learns to continue the text — ideal for a corpus like the Divina Commedia.
//   - Paired columns (inputCol & outputCol set): each row becomes one pair directly — for
//     Q&A / instruction datasets.
//
// Afterwards you train fully offline: `--mode=train --data=<out>` (or `--mode=replay`).
//
// startOffset lets you resume an interrupted import (combine with --append). A short delay
// is inserted between pages to stay under the Hugging Face rate limit; on a hard fetch
// failure the rows collected so far are still written so progress is never lost.
func runImportHF(dsName, config, split, textCol, inputCol, outputCol, outPath string, limit, startOffset, delayMs int, appendMode bool) {
	if dsName == "" {
		log.Fatal("--hf-dataset is required, e.g. --hf-dataset=maiurilorenzo/divina-commedia")
	}
	if os.Getenv("HF_TOKEN") == "" {
		log.Println("Tip: anonymous Hugging Face requests are heavily rate-limited. Set HF_TOKEN " +
			"(a free token from https://huggingface.co/settings/tokens) for a reliable import.")
	}
	delay := time.Duration(delayMs) * time.Millisecond
	paired := inputCol != "" && outputCol != ""

	client := &http.Client{Timeout: 60 * time.Second}
	var (
		texts     []string             // single-text-column mode buffer
		pairs     []dataset.DialogPair // paired mode (or final consecutive pairs)
		total     = -1
		offset    = startOffset
		fetchErr  error
		nextStart int // where a resume should pick up if this run is interrupted
	)
	if startOffset > 0 {
		log.Printf("Resuming import at offset %d (append=%v)", startOffset, appendMode)
	}

	for {
		if limit > 0 && offset >= startOffset+limit {
			break
		}
		length := hfPageSize
		if limit > 0 {
			if rem := startOffset + limit - offset; rem < length {
				length = rem
			}
		}

		resp, err := fetchHFRows(client, dsName, config, split, offset, length)
		if err != nil {
			// Don't discard what we already have — save it and tell the user how to resume.
			fetchErr = err
			nextStart = offset
			break
		}
		if total < 0 {
			total = resp.NumRowsTotal
			log.Printf("Dataset %s [%s/%s]: %d rows total", dsName, config, split, total)
		}
		if len(resp.Rows) == 0 {
			break
		}

		for _, r := range resp.Rows {
			if paired {
				in := hfString(r.Row, inputCol)
				out := hfString(r.Row, outputCol)
				if strings.TrimSpace(in) != "" && strings.TrimSpace(out) != "" {
					pairs = append(pairs, dataset.DialogPair{Input: in, Output: out, Lang: "it"})
				}
			} else {
				if t := strings.TrimSpace(hfString(r.Row, textCol)); t != "" {
					texts = append(texts, t)
				}
			}
		}

		offset += len(resp.Rows)
		log.Printf("Fetched %d/%d rows...", min(offset, total), total)
		if offset >= total {
			break
		}
		time.Sleep(delay) // be polite to the datasets-server rate limit
	}

	// Single-text-column mode: turn the ordered lines into consecutive (line, nextLine) pairs.
	if !paired {
		for i := 0; i+1 < len(texts); i++ {
			pairs = append(pairs, dataset.DialogPair{Input: texts[i], Output: texts[i+1], Lang: "it"})
		}
	}

	if len(pairs) == 0 {
		if fetchErr != nil {
			log.Fatalf("Fetch failed at offset %d with no rows collected: %v", nextStart, fetchErr)
		}
		log.Fatal("No usable pairs (check --hf-text / --hf-input / --hf-output column names).")
	}

	written, werr := dataset.WriteJSONL(outPath, pairs, appendMode)
	if werr != nil {
		log.Fatalf("Write dataset %s: %v", outPath, werr)
	}

	if fetchErr != nil {
		log.Printf("⚠️  Fetch interrupted at offset %d (%v).", nextStart, fetchErr)
		log.Printf("⚠️  Saved %d pairs so far → %s. Resume the rest with --append:", written, outPath)
		log.Printf("    go run . --mode=import-hf --hf-dataset=%s --hf-offset=%d --out=%s --append",
			dsName, nextStart, outPath)
		if os.Getenv("HF_TOKEN") == "" {
			log.Printf("    (set HF_TOKEN first to avoid the rate limit, or add --hf-delay=1000 to go slower)")
		}
		return
	}

	log.Printf("✅ Imported %s → %d pairs written to %s", dsName, written, outPath)
	log.Printf("Train offline (no network) with:")
	log.Printf("    go run . --mode=train  --data=%s --workers=0", outPath)
	log.Printf("    go run . --mode=replay --data=%s --workers=0   # fine-tune an existing checkpoint", outPath)
}

// fetchHFRows requests one page from the datasets-server, retrying transient 5xx errors
// (the server may briefly return "dataset is being processed"). An optional HF_TOKEN env
// var is sent as a bearer token so gated/private datasets you have access to also work.
func fetchHFRows(client *http.Client, dsName, config, split string, offset, length int) (*hfRowsResponse, error) {
	q := url.Values{}
	q.Set("dataset", dsName)
	q.Set("config", config)
	q.Set("split", split)
	q.Set("offset", strconv.Itoa(offset))
	q.Set("length", strconv.Itoa(length))
	endpoint := hfRowsEndpoint + "?" + q.Encode()

	const maxAttempts = 6
	var lastErr error
	for attempt := 0; attempt < maxAttempts; attempt++ {
		req, err := http.NewRequest(http.MethodGet, endpoint, nil)
		if err != nil {
			return nil, err
		}
		if tok := os.Getenv("HF_TOKEN"); tok != "" {
			req.Header.Set("Authorization", "Bearer "+tok)
		}
		res, err := client.Do(req)
		if err != nil {
			lastErr = err
			time.Sleep(backoff(attempt, 0))
			continue
		}
		body, _ := io.ReadAll(res.Body)
		retryAfter := parseRetryAfter(res.Header.Get("Retry-After"))
		res.Body.Close()

		if res.StatusCode == http.StatusOK {
			var parsed hfRowsResponse
			if err := json.Unmarshal(body, &parsed); err != nil {
				return nil, fmt.Errorf("parse response: %w", err)
			}
			return &parsed, nil
		}
		lastErr = fmt.Errorf("HTTP %d", res.StatusCode)

		// 429 (rate limit) and 5xx are transient → wait and retry. Other 4xx are fatal.
		if res.StatusCode != http.StatusTooManyRequests && res.StatusCode < 500 {
			return nil, fmt.Errorf("HTTP %d: %s", res.StatusCode, strings.TrimSpace(string(body)))
		}
		wait := backoff(attempt, retryAfter)
		log.Printf("Rate-limited/transient (HTTP %d) at offset %d; retrying in %s...", res.StatusCode, offset, wait)
		time.Sleep(wait)
	}
	return nil, fmt.Errorf("giving up after %d attempts: %w", maxAttempts, lastErr)
}

// backoff returns the wait before the next retry: the server's Retry-After if provided,
// otherwise exponential (2s, 4s, 8s, …) capped at 30s.
func backoff(attempt int, retryAfter time.Duration) time.Duration {
	if retryAfter > 0 {
		return retryAfter
	}
	d := time.Duration(2<<attempt) * time.Second
	if d > 30*time.Second {
		d = 30 * time.Second
	}
	return d
}

// parseRetryAfter reads a Retry-After header expressed in seconds (the form the HF API
// uses). Returns 0 if absent or unparseable.
func parseRetryAfter(h string) time.Duration {
	h = strings.TrimSpace(h)
	if h == "" {
		return 0
	}
	if secs, err := strconv.Atoi(h); err == nil && secs > 0 {
		return time.Duration(secs) * time.Second
	}
	return 0
}

// hfString extracts a column value as text. String cells are returned as-is; non-string
// cells (numbers, bools) are rendered from their raw JSON. Missing columns yield "".
func hfString(row map[string]json.RawMessage, col string) string {
	raw, ok := row[col]
	if !ok {
		return ""
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}
	return strings.Trim(string(raw), "\"")
}
