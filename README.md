# Discord LLM Bot — Pure Go Transformer

A Discord bot powered by a transformer decoder-only LLM implemented **entirely from scratch in Go** — no PyTorch, no TensorFlow, no HuggingFace. Everything — autograd, BPE tokeniser, attention, training loop — is hand-rolled using only the Go standard library.

**The bot learns directly from chat, in real time.** It does **not** need a pre-made dataset: start it with no prior training and it bootstraps its own tokeniser and an untrained model, then learns its language behaviour entirely from the messages it sees. The synthetic dataset and `--mode=train` are an **optional** offline head-start, not a requirement.

## Architecture

| Component | Location | Description |
|-----------|----------|-------------|
| Autograd engine | `model/tensor.go`, `model/backprop.go` | Micrograd-style dynamic graph with topological backward pass |
| Multi-Head Attention | `model/attention.go` | Causal self-attention, Q/K/V projections, scaled dot-product |
| Feed-Forward Network | `model/feedforward.go` | Linear → GELU → Linear |
| Layer Norm | `model/layernorm.go` | Pre-norm architecture with gamma/beta |
| Positional Encoding | `model/embeddings.go` | Sinusoidal fixed PE + learnable token embeddings |
| Transformer | `model/transformer.go` | Decoder-only, NLayers stacked blocks, Save/Load checkpoints |
| Optimiser | `model/optimizer.go` | AdamW + warmup + cosine decay + gradient clipping |
| BPE Tokeniser | `tokenizer/bpe.go` | Byte-Pair Encoding trained from scratch, heap-based merging |
| Message Index | `index/message_index.go` | Ring buffer, bag-of-embeddings, cosine RAG retrieval, pair export |
| Message Store | `index/db.go` | Persistent SQLite store (pure-Go), tokens/embeddings as BLOBs |
| Online Trainer | `train/online_trainer.go` | **Continual learning from chat** — background AdamW steps, EWC, checkpoints |
| Dataset (optional) | `dataset/` | Synthetic IT/EN dialog pairs for *optional* offline pre-training |
| Training Loop | `train/trainer.go` | Offline epoch/batch loop, cross-entropy, checkpoints, BLEU, perplexity |
| Sampler | `generate/sampler.go` | Temperature · Top-K · Top-P · repetition penalty · human effects |
| Discord Bot | `bot/discord.go` | discordgo, message indexing, RAG, online pairs, slash commands |

**Hyperparameters**: dModel=256, nHeads=4, nLayers=4, dFF=1024, seqLen=128, vocabSize=8192, lr=3e-4, epochs=20

## Prerequisites

- Go 1.22+
- A Discord bot token ([Discord Developer Portal](https://discord.com/developers/applications))

## Setup

```bash
git clone <repo>
cd awesomeProject

# Install dependencies
go mod download

# Copy env template and fill in your values
cp .env.example .env
# Edit .env: set DISCORD_TOKEN and GUILD_ID
```

## Building

`go run .` works for everything below. To produce optimized binaries (one per mode),
cross-compile, or use the `Makefile` targets, see **[BUILD.md](BUILD.md)**:

```bash
make all          # binary + one executable per mode in bin/
./bin/bot --workers=8
```

## Running the Bot (learns from chat — no dataset needed)

```bash
# Start from scratch: bootstraps a tokeniser + fresh model, then learns from chat.
# No --mode=train required. Just set DISCORD_TOKEN in .env.
go run . --mode=bot

# Force a specific checkpoint (skips auto-resume):
go run . --mode=bot --checkpoint=checkpoints/best

# Bot responds to all messages in channels it can see.
# Slash commands: /reset, /status, /stats, /learn, /replay [n], /forget
```

**Checkpoints & auto-resume.** Without an explicit `--checkpoint`, the bot resumes the
previous online session automatically: it loads `checkpoints/online_latest/` if present,
then `checkpoints/best/`, otherwise it bootstraps fresh. On shutdown it saves to
`checkpoints/online_latest/` — so **load path == save path** and continual learning
persists across restarts. Online checkpoints are self-contained (they bundle their own
`vocab.json`), so the tokeniser can never drift out of sync with the model.

On first run with no `data/vocab.json`, the bot builds a bootstrap tokeniser from a
built-in **character** seed (Italian/English letters, digits, punctuation, the `→`
separator, common emoji — *not* dialog data) so live chat encodes cleanly, then learns
everything else online.

> ⚠️ **An untrained model replies with gibberish — this is expected.** A randomly
> initialised transformer samples tokens at random until it has learned. For coherent
> replies, either run `--mode=train` first (offline pre-training on the synthetic dataset)
> and start the bot from that checkpoint, or let it learn online for a long time. Online
> learning alone (`lr=1e-5`, 256-dim model) is slow — pre-training is strongly recommended.

## Performance (multi-core)

Two independent layers of parallelism:

**1. Intra-op (matrix multiply).** `rawMatMul` (`model/tensor.go`) parallelises large
products across cores by partitioning the rows of the left operand — each worker writes a
disjoint row range, so it is race-free. Below `m*k*n = 32768` ops it stays single-threaded.
This speeds up *everything* (generation + training). ≈ **2.9×** forward pass on 8 cores.

```go
model.SetMatMulWorkers(n) // n cores; n=1 disables (default = NumCPU)
```

**2. Data-parallel training.** For *all* training modes, set `--workers` to split each
mini-batch across goroutines, each with its own model replica; gradients are reduced into
the master before one AdamW step. It is numerically identical to sequential accumulation
(verified to ~1e-15) and race-free.

```bash
go run . --mode=train  --workers=8     # data-parallel offline pre-training
go run . --mode=replay --workers=8     # data-parallel replay / backfill
go run . --mode=bot    --workers=8     # data-parallel online learning
go run . --mode=train  --workers=1     # disable (sequential)
# --workers=0 (default) = auto = NumCPU
```

Measured on 8 cores (256-dim / 4-layer model), end-to-end:

| Strategy | `train` (batch=32) | `replay` (128 pairs) |
|----------|-------------------:|---------------------:|
| Single-core | 1.00× | 1.00× |
| Matmul-parallel only (`--workers=1`) | ~1.8× | 1.79× |
| **Data-parallel `--workers=8`** | **3.59×** | **3.42×** |

When data parallelism is active, intra-op matmul parallelism is disabled for the duration
to avoid oversubscription (W workers × 1 core each). Cost: `--workers=N` keeps `N` copies of
the model in memory (≈ N × model size), so lower it for very large models. Generation and
training share `ModelMu`, so only one parallel region runs at a time.

## Continual (Online) Learning

In bot mode the model is **not** frozen — it learns from live chat in real time, and can do
so starting from random weights:

- **Indexing**: every message is tokenised, embedded, and stored in `data/messages.db`
  (SQLite, pure-Go via `modernc.org/sqlite`) plus an in-memory ring buffer of the last 1000.
- **RAG**: before replying, the 3 most semantically similar past messages (cosine
  similarity over bag-of-embeddings vectors) are prepended to the prompt.
- **Online trainer**: a background goroutine accumulates conversational pairs and applies
  one small AdamW step per 16-pair buffer (`lr=1e-5`, grad-clip `0.5`), checkpointing to
  `checkpoints/online_N/` every 500 steps and `checkpoints/online_latest/` on shutdown.
- **Thread-safety**: generation, training, and embedding all share the model through a
  single `ModelMu` mutex — verified race-free (`go test -race ./train/... ./index/...`).
- **EWC** (optional): Elastic Weight Consolidation anchors weights to mitigate catastrophic
  forgetting (`loss += λ·Σ Fᵢ(θᵢ−θ*ᵢ)²`).

### Learning from past messages (backfill)

By default the bot only learns from messages seen *while it is running*. To make it learn
from a channel's **existing history**, use `/learn` — it fetches old messages from the
Discord API, indexes them, and trains on the resulting pairs:

```
/learn                       # last 1000 messages of this channel, 2 epochs
/learn messages:5000         # last 5000 messages
/learn messages:0 epochs:3   # as many as possible (capped at 10000), 3 epochs
```

`/learn` is **admin only** (it does many API calls + training). It trains per mini-batch,
releasing the model lock between batches, so the bot stays responsive (just slower) while
catching up on history.

| Command | Source | Fetches from Discord? |
|---------|--------|-----------------------|
| `/learn [messages] [epochs]` | channel history via Discord API | ✅ yes |
| `--mode=harvest` | channel history → JSONL dataset on disk | ✅ once |
| `/replay [n]` | messages already in the local DB | ❌ no |
| `--mode=replay --n=N` | same as `/replay`, from the CLI | ❌ no |
| `--mode=train --data=FILE` / `--mode=replay --data=FILE` | a JSONL dataset on disk | ❌ no |

```bash
# Replay training on already-stored history (offline batch over the message DB):
go run . --mode=replay --checkpoint=checkpoints/best --n=500
```

### Harvesting a channel to a reusable dataset (download once, train forever)

`/learn` re-downloads history every time. To pay the Discord API cost **once** and keep a
dataset on disk you can re-train from repeatedly, use `--mode=harvest`. It connects over
REST (no gateway), pulls the requested channels' history, builds consecutive
conversational pairs, and writes them as JSONL — the same format as the synthetic dataset.

```bash
# Specific channel(s), comma-separated; --n caps messages per channel (default 20000):
go run . --mode=harvest --channel=123456789012345678,234567890123456789 \
         --out=data/harvest.jsonl --n=5000

# No --channel → all text channels of GUILD_ID (from .env). --append accumulates:
go run . --mode=harvest --out=data/harvest.jsonl --append
```

Then re-train **offline, multi-core, never touching the API again** — as many times as you
like:

```bash
go run . --mode=train  --data=data/harvest.jsonl --workers=8   # fresh model, full schedule
go run . --mode=replay --data=data/harvest.jsonl --workers=8   # fine-tune a checkpoint
```

> Requires the **Message Content Intent** (privileged) enabled in the Developer Portal —
> the same one the live bot needs. `--mode=train --data=...` reuses `data/vocab.json` if it
> exists; delete it first to rebuild the BPE vocabulary from the harvested corpus.

### Importing a Hugging Face dataset (e.g. to learn basic Italian)

`--mode=import-hf` downloads any public Hugging Face dataset over the datasets-server HTTP
API (no Python / `datasets` library) and converts it to the same JSONL format. For a
single-text-column corpus like the Divina Commedia it builds **consecutive-line pairs** so
the model learns to continue Italian text:

```bash
# 14k verses of the Divine Comedy → consecutive (line, next-line) pairs
go run . --mode=import-hf --hf-dataset=maiurilorenzo/divina-commedia \
         --hf-text=text --out=data/divina.jsonl

# Then train offline, multi-core (delete data/vocab.json first for a fresh Italian vocab):
rm -f data/vocab.json
go run . --mode=train --data=data/divina.jsonl --workers=8
```

For paired (Q&A / instruction) datasets, map the two columns instead:

```bash
go run . --mode=import-hf --hf-dataset=owner/qa-dataset \
         --hf-input=question --hf-output=answer --out=data/qa.jsonl
```

Flags: `--hf-dataset` (id, required), `--hf-config` (default `default`), `--hf-split`
(default `train`), `--hf-text` (default `text`), `--hf-input`/`--hf-output` (paired mode),
`--out`, `--append`, `--n` (row cap, default = all), `--hf-offset` (resume), `--hf-delay`
(ms between pages, default 1000).

> **Rate limits.** Anonymous Hugging Face requests are throttled (HTTP 429). The import
> paginates 100 rows at a time with a 1s delay (enough for the anonymous limit) and
> **retries** transient errors. If it's still interrupted it **saves what it fetched** and
> prints a `--hf-offset=N … --append` command to resume. For large datasets, set a free
> `HF_TOKEN` (https://huggingface.co/settings/tokens) in the environment for a much higher
> limit, then you can lower `--hf-delay`.

`/forget` (admin only) wipes the message DB and re-initialises the weights, requiring
`confirm:true` within 30 seconds of the first invocation.

## Optional: Offline Pre-training

Pre-training on the synthetic dataset is **optional** — use it only if you want the bot to
start with some baseline language ability before learning from chat.

```bash
# Generate dataset + train BPE + train model (all automatic)
go run . --mode=train

# Training log example:
# Epoch 1 [=====>    ] 45/100 loss=6.2341 ppl=507.2 lr=3.00e-04 gnorm=0.812
# Epoch 1/20 — train_loss=5.8431  val_loss=5.7821  val_ppl=324.1  lr=2.97e-04
# Checkpoint saved → checkpoints/epoch_01
```

Data and vocabulary are saved to `data/`; checkpoints go to `checkpoints/` (best at
`checkpoints/best/`). Loss typically drops from ~8–9 to <3 within 3–5 epochs. Then run the
bot with `--checkpoint=checkpoints/best` to continue learning online from there.

## Text Generation (CLI)

```bash
go run . --mode=generate --checkpoint=checkpoints/best --prompt="Ciao, come stai?"
```

## Smoke Test

```bash
go run . --mode=test
```

Runs forward pass, loss backward, and BPE round-trip without requiring a trained model.

## Running Tests

```bash
go test ./...

# With verbose output
go test ./model/... -v

# Benchmark forward pass
go test ./model/... -bench=BenchmarkForward -benchtime=5s
```

## Project Structure

```
.
├── main.go              # Entry point, mode dispatch
├── go.mod / go.sum
├── .env.example         # Environment template
├── model/
│   ├── tensor.go        # Tensor type + raw matrix ops
│   ├── backprop.go      # Autograd ops with backward closures
│   ├── attention.go     # Multi-head causal self-attention
│   ├── feedforward.go   # FFN (Linear → GELU → Linear)
│   ├── embeddings.go    # Token embeds + sinusoidal pos encoding
│   ├── layernorm.go     # LayerNorm with residual
│   ├── transformer.go   # Full decoder-only transformer, checkpoint I/O
│   ├── optimizer.go     # AdamW + LR schedule + grad clip
│   ├── rand.go          # Global RNG
│   └── model_test.go    # Unit + numerical gradient tests
├── tokenizer/
│   ├── bpe.go           # BPE training + encode/decode
│   ├── vocab.go         # JSON save/load
│   └── tokenizer_test.go
├── dataset/
│   ├── generator.go     # 5000+ IT/EN dialog pairs
│   ├── loader.go        # JSONL → batches with padding
│   ├── augment.go       # Token dropout, synonym swap
│   └── dataset_test.go
├── train/
│   ├── trainer.go          # Offline training loop with BLEU + perplexity
│   ├── online_trainer.go   # Continual background learning (lr 1e-5, EWC, checkpoints)
│   └── online_trainer_test.go
├── index/
│   ├── message_index.go    # Ring buffer, embeddings, RAG retrieval, pair export
│   ├── db.go               # SQLite message store (pure-Go modernc.org/sqlite)
│   └── message_index_test.go
├── generate/
│   ├── sampler.go       # Sampling strategies + human-like effects
│   └── generate_test.go
├── bot/
│   └── discord.go       # Discord integration, indexing, RAG, slash commands
├── data/                # Generated: train.jsonl, vocab.json, messages.db
└── checkpoints/         # Saved model checkpoints (incl. online_N, online_latest)
```

## Dependencies

External packages beyond the Go standard library:

- [`github.com/bwmarrin/discordgo`](https://github.com/bwmarrin/discordgo) — Discord WebSocket API
- [`github.com/joho/godotenv`](https://github.com/joho/godotenv) — `.env` file loading
- [`modernc.org/sqlite`](https://modernc.org/sqlite) — pure-Go SQLite (no CGo) for message persistence

## Human-Like Response Pipeline

Each generated response passes through:
1. Repetition penalty on recent context tokens
2. Temperature scaling → Top-K filter → Top-P (nucleus) filter → sample
3. Uncertainty injection ("forse", "boh", "maybe") at 5% probability
4. Contextual emoji insertion based on sentiment keywords
5. QWERTY typo simulation at 2% per character
6. Response variability hash check (avoids identical replies)
7. Simulated typing delay (proportional to response length)




# 1. Importa i 14.233 versi → coppie di righe consecutive (verso N → verso N+1)
go run . --mode=import-hf --hf-dataset=maiurilorenzo/divina-commedia --out=data/divina.jsonl

# 2. Pulisci il vocab così il BPE si ricostruisce sull'italiano della Commedia
rm -f data/vocab.json checkpoints/

# 3. Allena offline, multi-core (nessuna rete)
go run . --mode=train --data=data/divina.jsonl --workers=8

