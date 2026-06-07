# CLAUDE.md — Discord LLM Bot in Pure Go

## Panoramica

Questo progetto implementa un bot Discord con un LLM transformer decoder-only scritto **interamente in Go**, senza librerie ML esterne. Ogni componente — autograd, tokenizer BPE, attention, training loop — è implementato a mano usando solo la standard library.

**Il bot impara direttamente dalla chat in tempo reale.** Il dataset sintetico (`--mode=train`) è **opzionale**: serve solo per un pre-training offline che dà al modello un punto di partenza. In `--mode=bot` il modello può partire da zero — se non esiste `data/vocab.json` viene generato un tokenizer di bootstrap da un seed di caratteri (non dati di dialogo) — e apprende il linguaggio **interamente dai messaggi** che riceve.

## Comandi essenziali

```bash
# Build
go build ./...

# Test (tutti i package)
go test ./...

# Smoke test end-to-end (no checkpoint richiesto)
go run . --mode=test

# Bot Discord — impara dalla chat in tempo reale (richiede .env con DISCORD_TOKEN)
# Senza --checkpoint riprende AUTOMATICAMENTE l'ultima sessione online (online_latest),
# poi best, poi fresh bootstrap. L'apprendimento online persiste tra i riavvii.
go run . --mode=bot
# Forzare un checkpoint specifico (ignora l'auto-resume):
go run . --mode=bot --checkpoint=checkpoints/best

# (Opzionale) Pre-training offline sul dataset sintetico per un head-start
go run . --mode=train

# Generazione testo da checkpoint
go run . --mode=generate --checkpoint=checkpoints/best --prompt="Ciao!"

# Replay: ri-allena sui messaggi già salvati in data/messages.db
go run . --mode=replay --checkpoint=checkpoints/best --n=500

# Harvest: scarica UNA VOLTA la cronologia di uno o più canali in un dataset su disco
# (richiede DISCORD_TOKEN). Poi ri-alleni quante volte vuoi senza più toccare l'API.
go run . --mode=harvest --channel=123,456 --out=data/harvest.jsonl
go run . --mode=harvest --out=data/harvest.jsonl   # senza --channel: tutti i canali testuali di GUILD_ID

# Ri-allenamento OFFLINE dal dataset raccolto (nessuna chiamata a Discord):
go run . --mode=train  --data=data/harvest.jsonl --workers=0   # modello fresco, schedule completo
go run . --mode=replay --data=data/harvest.jsonl --workers=0   # fine-tune di un checkpoint esistente

# Import da Hugging Face → dataset JSONL (via datasets-server HTTP API, no Python).
# Colonna testo singola → coppie di righe consecutive (es. corpus/poesia):
go run . --mode=import-hf --hf-dataset=maiurilorenzo/divina-commedia --hf-text=text --out=data/divina.jsonl
# Dataset già a coppie (Q&A/instruction):
go run . --mode=import-hf --hf-dataset=owner/qa --hf-input=question --hf-output=answer --out=data/qa.jsonl
# Poi: go run . --mode=train --data=data/divina.jsonl --workers=0
```

## Struttura del progetto

```
awesomeProject/
├── main.go                 # Entry point: --mode=train|bot|generate|replay|harvest|test
├── model/
│   ├── tensor.go           # Tipo Tensor (flat row-major float64), raw matrix ops
│   ├── backprop.go         # Operatori autograd (Add, MatMul, GELU, Softmax, …) + Backward()
│   ├── attention.go        # Multi-Head Self-Attention causale
│   ├── feedforward.go      # FFN: Linear → GELU → Linear
│   ├── embeddings.go       # Token embedding + sinusoidal positional encoding
│   ├── layernorm.go        # LayerNorm pre-norm con gamma/beta
│   ├── transformer.go      # TransformerConfig, TransformerModel, Save/Load checkpoint
│   ├── optimizer.go        # AdamW + warmup lineare + cosine decay + grad clipping
│   └── rand.go             # RNG globale thread-safe
├── tokenizer/
│   ├── bpe.go              # BPE: Train (heap-based), Encode, Decode
│   └── vocab.go            # JSON save/load del vocabolario
├── dataset/
│   ├── generator.go        # 5000+ coppie dialogo IT/EN generate programmaticamente
│   ├── loader.go           # JSONL → Batch con padding, split 80/10/10
│   └── augment.go          # TokenDropout, SynonymSwap, BackTranslationSim
├── train/
│   ├── trainer.go          # Loop training offline, cross-entropy, checkpoint, BLEU, perplexity
│   └── online_trainer.go   # OnlineTrainer: apprendimento continuo in background (lr 1e-5)
├── index/
│   ├── message_index.go    # MessageRecord, ring buffer 1000, Add/GetContext/FindSimilar/ExportPairs, RAG
│   └── db.go               # MessageDB SQLite (modernc.org/sqlite, pure-Go) — persistenza messaggi
├── generate/
│   └── sampler.go          # Temperature, Top-K, Top-P, repetition penalty, effetti human-like
├── bot/
│   └── discord.go          # discordgo, indicizzazione, RAG, pair online, /reset /status /stats /replay /forget
├── data/                   # Generato: train.jsonl, vocab.json, messages.db  [gitignored]
└── checkpoints/            # Checkpoint modello (incl. online_N, online_latest)  [gitignored]
```

## Iperparametri di default

| Parametro | Valore |
|-----------|--------|
| `dModel` | 256 |
| `nHeads` | 4 |
| `nLayers` | 4 |
| `dFF` | 1024 |
| `seqLen` | 128 |
| `vocabSize` | 8192 |
| `lr` | 3e-4 |
| `epochs` | 20 |
| `warmupSteps` | 500 |
| `batchSize` | 32 |
| `gradClip` | 1.0 |
| `weightDecay` | 0.01 |
| `earlyStopPatience` | 3 |

Modificabili in `train.DefaultConfig()` (`train/trainer.go`).

## Architettura autograd

Il motore autograd è **dynamic graph micrograd-style**:

- Ogni `*Tensor` porta `Data []float64`, `Grad []float64`, `children []*Tensor`, `backwardFn func()`.
- Le funzioni operatore (`Add`, `MatMul`, `GELU`, `SoftmaxRows`, `LayerNormForward`, `MaskFill`, `SumAll`, …) calcolano il forward e impostano la `backwardFn` come closure che cattura gli input.
- `tensor.Backward()` costruisce l'ordinamento topologico con DFS post-order e itera al contrario chiamando ogni `backwardFn`.
- Il grafo viene **ricostruito ad ogni forward pass** — non ci sono grafi statici.
- Le funzioni raw (`rawMatMul`, `rawTranspose`, …) in `tensor.go` operano su `[]float64` direttamente: **usarle sempre nelle backward closures** per evitare di costruire un secondo grafo.

**Backward su output non-scalare:** `Backward()` assume input scalare (1×1). Per loss multi-elemento usare sempre `SumAll(tensor)` prima di chiamare `.Backward()`.

## Tokenizer BPE

- Token speciali: `[PAD]=0, [BOS]=1, [EOS]=2, [UNK]=3` (costanti in `tokenizer/bpe.go`).
- Vocabolario iniziale: tutti i caratteri UTF-8 del corpus a partire da ID 4.
- Training: heap `container/heap` per O(log V) best-pair lookup; frequenze aggiornate in modo incrementale.
- `Encode(text)` restituisce `[BOS] + token_ids + [EOS]`.
- `EncodeNoSpecial(text)` senza token speciali (usare nel bot per il prompt di contesto).
- Vocabolario salvato/caricato da `data/vocab.json`.

## Gestione dati

- `dataset.GenerateDataset(path, seed)` scrive il JSONL; usato **solo** dal pre-training offline (`--mode=train`), non dal bot.
- `tokenizer.Load(path)` carica il BPE da `data/vocab.json`.
- **Bot da zero (no dataset):** se `data/vocab.json` manca, `freshModel()` in `main.go` chiama `bpe.Train(seedCorpus(), bootstrapVocabSize)` — `seedCorpus()` è un seed di **caratteri** (lettere IT/EN, cifre, punteggiatura, separatore `→`, emoji), non dati di dialogo. Garantisce copertura caratteri per codificare la chat con pochi `[UNK]`; il comportamento linguistico viene appreso interamente online. Test: `TestSeedCorpusEncodesChat`.
- Il batch target è il batch input spostato di 1 posizione a sinistra; le posizioni padding usano `tokenizer.PAD` come sentinel per ignorare la loss.

## Checkpoint

- Offline: `checkpoints/epoch_NN/` (ogni epoca) e `checkpoints/best/` (miglior val loss).
- Online: `checkpoints/online_N/` ogni 500 step, `checkpoints/online_latest/` allo shutdown.
- Formato: `config.json` + `model.bin` (gob di `[][]float64`). I checkpoint **online** includono anche `vocab.json` (auto-contenuti → niente mismatch vocab/modello tra run).
- Caricamento: `model.LoadConfig(dir)` → `model.NewTransformerModel(cfg)` → `m.Load(dir)`. `loadModelAndBPE` in `main.go` preferisce il `vocab.json` dentro il checkpoint, con fallback a `data/vocab.json`, e verifica `bpe.VocabLen() == cfg.VocabSize`.
- **Auto-resume del bot (`runBot`/`resolveBotCheckpoint`):** senza `--checkpoint` esplicito riprende da `checkpoints/online_latest` → poi `checkpoints/best` → infine fresh bootstrap. Con `--checkpoint X` usa esattamente `X`. Load path == save path, quindi l'apprendimento online persiste tra i riavvii.

## Apprendimento online / continuo (real-time learning)

In `--mode=bot` il bot **non** usa un dataset statico: impara in tempo reale dalla chat. Può partire completamente da zero (modello a pesi random + tokenizer di bootstrap) e costruire la competenza linguistica solo dai messaggi. Un eventuale checkpoint (`--mode=train`) serve solo come head-start opzionale.

**Flusso (`bot/discord.go` → `messageCreate`):**
1. **Ogni** messaggio viene indicizzato (`index.MessageIndex.Add`, in goroutine) — anche quelli a cui non risponde.
2. RAG: prima di generare, recupera i 3 messaggi più simili (`FindSimilar` su cosine similarity) e li antepone al prompt.
3. Se il bot risponde, emette una coppia di training `(user, bot)` su `OnlineTrainer.Submit`.
4. `index.Add` emette anche coppie conversazionali reali `(prev, curr)` per lo stesso canale.

**OnlineTrainer (`train/online_trainer.go`):**
- Goroutine `Start(ctx)`: legge da `trainCh`, accumula in un buffer da 16 coppie, poi esegue **uno** step.
- Iperparametri online: `lr=1e-5` (costante: warmup=1, horizon enorme), `gradClip=0.5`, `weightDecay=0`.
- Un `flush` = un update AdamW; checkpoint ogni 500 step; `ForceFlush()` allo shutdown.
- EWC opzionale (`EnableEWC`): `loss += λ·Σ Fᵢ(θᵢ−θ*ᵢ)²` per mitigare il catastrophic forgetting.

**⚠️ Concorrenza (invariante critica):** il modello è condiviso tra generazione (bot), training online e calcolo embedding. Tutti gli accessi ai pesi passano per **lo stesso** `*sync.Mutex` esposto come `OnlineTrainer.ModelMu`:
- bot: `b.lockModel()/unlockModel()` attorno a `Sampler.Generate`
- index: `Embed` legge `TokenEmbed.Data` sotto `modelMu`
- trainer: `trainBatch` tiene `ModelMu` per tutto Forward+Backward+Step

Verificato race-free con `go test -race ./train/... ./index/...` (vedi `TestConcurrentTrainAndRead`). Se aggiungi nuovi accessi al modello, **devi** acquisire `ModelMu`.

**Persistenza:** `index/db.go` usa `database/sql` + `modernc.org/sqlite` (pure-Go, no CGo). `SetMaxOpenConns(1)` evita "database is locked". Token ed embedding salvati come BLOB little-endian.

**Backfill dalla cronologia (`/learn`):** `bot.backfillAndLearn` usa `Session.ChannelMessages` (paginato a 100, cursore `before`) per scaricare fino a `maxBackfill=10000` messaggi del canale, li indicizza con `index.Ingest` (no emissione su `trainCh`), costruisce coppie consecutive e chiama `ReplayBatch`. `ReplayBatch` blocca `ModelMu` **per mini-batch** (non per tutta la run) → il bot resta responsivo durante il catch-up. Richiede il MESSAGE CONTENT INTENT anche via REST.

**Modalità replay:** `--mode=replay` (e `/replay [n]`) ri-allenano sui messaggi **già nel DB locale** (`ExportPairs` → `ReplayBatch`), senza scaricare nulla da Discord. Differenza chiave: `/learn` *scarica* la cronologia, `/replay` no.

**Harvest → dataset su disco (`--mode=harvest`):** in `main.go`, `runHarvest` apre una sessione discordgo **solo REST** (niente `s.Open()`/gateway), enumera i canali (`--channel=ID,ID` oppure tutti i text channel di `GUILD_ID` via `GuildChannels`), pagina la cronologia con `fetchChannelHistory` (uguale al backfill del bot ma filtra i messaggi `Author.Bot`), costruisce coppie consecutive e le scrive in JSONL con `dataset.WriteJSONL` (formato `DialogPair`, `--append` per accumulare, `--n` cap per canale, default `defaultHarvestPerChannel=20000`). **Scopo:** pagare il costo dell'API Discord **una sola volta** → poi ri-allenare offline N volte. Cap messaggi: `--n` (default 20000/canale). Richiede MESSAGE CONTENT INTENT anche via REST.

**Re-training offline da JSONL (`--data`):** il flag `--data=path.jsonl` fa sì che `--mode=train` usi quel file come `Config.DataPath` (modello fresco, schedule completo, multi-core) e `--mode=replay` carichi le coppie da file invece che dal DB (`loadPairsJSONL` → `ReplayBatch`, fine-tune di un checkpoint). Nessuno dei due tocca l'API. **Nota vocab:** `--mode=train --data=...` riusa `data/vocab.json` se esiste; cancellalo per ricostruire il BPE dal corpus raccolto.

**Le tre vie a confronto:** `/learn` (slash, scarica+allena subito, transitorio) · `--mode=harvest` (scarica → dataset su disco riutilizzabile) · `--mode=replay`/`--mode=train --data=` (allena da disco, zero API).

**Import da Hugging Face (`--mode=import-hf`):** in `hfimport.go`, `runImportHF` scarica un dataset pubblico via la **datasets-server HTTP API** (`https://datasets-server.huggingface.co/rows`, pagina 100 righe, niente libreria Python) e lo converte in JSONL `DialogPair`. Due forme: **(a)** colonna testo singola (`--hf-text`, default `text`) → coppie di righe **consecutive** (impara a continuare il testo, ideale per corpora come la Divina Commedia); **(b)** colonne appaiate (`--hf-input`/`--hf-output`) → una coppia per riga (Q&A/instruction). Flag: `--hf-dataset` (id, obbligatorio), `--hf-config` (default `default`), `--hf-split` (default `train`), `--out`, `--append`, `--n` (cap righe, 0=tutte). `HF_TOKEN` opzionale via env per dataset gated. Retry su 5xx. Output ri-allenabile con `--mode=train --data=`.

**Embedding semantico:** media dei token-embedding (bag-of-embeddings, dim `DModel`) — niente grafo autograd, letto sotto `ModelMu`.

## Performance / multi-core

- `rawMatMul` (`model/tensor.go`) parallelizza i prodotti matriciali grandi (`m*k*n ≥ 32768`) su più core, partizionando le **righe** di A: ogni goroutine scrive righe disgiunte di C → race-free. Sotto soglia gira single-thread (overhead > guadagno).
- Worker di default = `runtime.NumCPU()`. Configurabile con `model.SetMatMulWorkers(n)` (`n=1` disabilita); `model.MatMulWorkers()` legge il valore.
- Speedup misurato ~2.9x su forward dModel=256/4 layer/seq=48 con 8 core (non 8x: LayerNorm/softmax/GELU/embedding restano seriali, le matmul per-head sono sotto soglia).
- **Niente oversubscription:** generazione e training tengono `ModelMu`, quindi al massimo una regione matmul-parallela è attiva → usa tutti i core senza contesa.
- **Data-parallel training** (flag `--workers`, vale per `train`, `replay` e `bot`): helper condiviso `parallelAccumulate` in `train/parallel.go`, usato sia dal `Trainer` offline (`trainEpoch`) sia dall'`OnlineTrainer` (`trainBatchParallel`). Ogni worker processa una fetta del batch su una **replica** del modello (`replicas = workers-1`, worker 0 = master, costruite da `makeReplicas`). Pesi sincronizzati dal master a ogni step, gradienti **sommati** nel master prima dell'unico `Step`. Numericamente **identico** al sequenziale (verificato a ~1e-15/1e-17: `TestDataParallelMatchesSequential`, `TestParallelAccumulateMatchesSequential`), race-free. Durante la fase parallela la matmul intra-op è forzata a 1 (`SetMatMulWorkers(1)`) per non oversubscribere. `--workers=0` (default) = auto = `NumCPU`, `1` = disabilitato. Costo: N copie del modello in RAM.
- Speedup end-to-end su 8 core (modello 256/4-layer): single-core 1.0x → solo matmul ~1.8x → **data-parallel x8: `train` 3.59x, `replay` 3.42x**.
- L'offline e l'online sommano i gradienti (no media) per esempio nel batch; coerente con `ClipGradients` a valle. (L'online poi fa `scaleGrads(1/n)`; l'offline no, come l'originale.)
- Generazione resta intra-op (matmul) perché è autoregressiva (un token alla volta).

## Comportamento del modello non addestrato

Un modello a pesi random (bot avviato senza checkpoint) **genera rumore** — campiona token a caso dal vocabolario di bootstrap. È atteso, non un bug. Per output coerenti: `--mode=train` (pre-training offline) poi `--mode=bot --checkpoint=checkpoints/best`, oppure molto apprendimento online. Il `seedCorpus()` copre i caratteri **uno per token** (no parole-alfabeto/`xNx` che generavano token-spazzatura).

## Dipendenze esterne

| Pacchetto | Scopo |
|-----------|-------|
| `github.com/bwmarrin/discordgo v0.28.1` | WebSocket Discord API |
| `github.com/joho/godotenv v1.5.1` | Caricamento `.env` |
| `modernc.org/sqlite v1.51.0` | SQLite pure-Go (no CGo) per la persistenza messaggi |

Non aggiungere dipendenze ML, algebra lineare o deep learning. SQLite pure-Go è ammesso solo come storage.

## Convenzioni di codice

- **Error handling esplicito**: niente `panic()` salvo `main()` e inizializzazioni fatali.
- **Commenti godoc** su tutte le funzioni pubbliche.
- **Nessun commento ovvio**: commentare solo il WHY non-ovvio (invarianti nascoste, workaround specifici).
- **Niente batch reale**: il batch size è simulato con gradient accumulation (loop su esempi singoli).
- Le funzioni helper `max`/`min` sono già built-in in Go 1.21+ — non ridefinirle.
- `range over int` (es. `for i := range n`) è disponibile da Go 1.22+ e già usato nel codebase.

## Test

```bash
# Test specifici con verbose
go test ./model/... -v
go test ./tokenizer/... -v
go test ./dataset/... -v
go test ./generate/... -v

# Benchmark forward pass
go test ./model/... -bench=BenchmarkForward -benchtime=5s

# Test con race detector
go test -race ./...
```

Test critici da non rompere:
- `TestBackwardMatMulNumerical` — verifica gradient check numerico (tolleranza 1e-3 relativa)
- `TestGELUGrad` — verifica derivata GELU vs finite differences
- `TestBPERoundTrip` — encode → decode deve restituire testo originale
- `TestGeneratorCount` — dataset deve avere ≥ 5000 coppie
- `TestOnlineStepReducesLoss` — 2 flush su coppie identiche → la loss scende
- `TestNoExplodingGradients` — dopo 100 step i pesi restano in [-10, 10]
- `TestCheckpointSaveLoad` — save/load online bit-identico
- `TestConcurrentTrainAndRead` — training + lettura modello concorrenti, race-free (`-race`)
- `index`: `TestAddAndRetrieve`, `TestCosineSimilarity`, `TestExportPairs`

Slash command runtime: `/reset`, `/status`, `/stats` (msg indicizzati, step, loss, DB size),
`/learn [messages] [epochs]` (admin; **scarica la cronologia del canale da Discord**, la indicizza e ci allena),
`/replay [n]` (replay sui msg già nel DB locale, niente fetch), `/forget` (admin, reset DB+pesi, conferma `confirm:true` entro 30s).

## Configurazione bot Discord

Creare `.env` (non committare mai):

```env
DISCORD_TOKEN=il_tuo_token_qui
GUILD_ID=id_del_tuo_server
```

Il token è nel [Discord Developer Portal](https://discord.com/developers/applications) → Bot → Reset Token.

## Pipeline di generazione (human-like)

In `generate/sampler.go`, ogni risposta passa attraverso:
1. Repetition penalty sui token del contesto
2. Temperature scaling → Top-K → Top-P (nucleus) → campionamento
3. Uncertainty injection (`"forse"`, `"boh"`, `"maybe"`) al 5% di probabilità
4. Emoji contestuale basata su keyword di sentiment
5. Typo simulator QWERTY al 2% per carattere
6. Hash-based variability check (evita risposte identiche)
7. Thinking delay proporzionale alla lunghezza risposta

## Estendere il progetto

**Aggiungere nuove categorie di dialogo:** editare `dataset/generator.go`, aggiungere una funzione `nomeCategoria(rng) []DialogPair` e registrarla in `buildAllPairs`.

**Modificare l'architettura:** cambiare `TransformerConfig` in `model/transformer.go`; ricordarsi di cancellare `data/vocab.json` e `checkpoints/` prima di rifare il training.

**Aggiungere un nuovo operatore autograd:** in `model/backprop.go`, seguire lo schema degli operatori esistenti — calcola forward, imposta `out.children` e `out.backwardFn` come closure.

**Nuovi comandi slash Discord:** aggiungere la definizione in `registerCommands()` e il handler in `interactionCreate()` in `bot/discord.go`.
