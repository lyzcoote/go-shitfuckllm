# BUILD.md — Build & run

Il progetto è un **singolo `package main`** con dispatch via `--mode`, quindi `go build`
produce **un solo binario**. Per avere "eseguibili diretti per ogni mode" si usa il binario
unico + wrapper leggeri (uno per mode), oppure il `Makefile` incluso.

---

## Quick start

```bash
make all          # build del binario + un eseguibile per ogni mode in bin/
./bin/bot --workers=8
```

---

## Makefile

| Comando | Cosa fa |
|---------|---------|
| `make build` | Solo il binario ottimizzato → `bin/discord-llm` |
| `make wrappers` | Binario + un eseguibile per ogni mode → `bin/train`, `bin/bot`, … |
| `make all` | `build` + `wrappers` |
| `make clean` | Rimuove `bin/` |
| `make test` | `go test ./...` |
| `make vet` | `go vet ./...` |
| `make <mode> ARGS="…"` | Build + run di quel mode (vedi sotto) |

### Shortcut "build + run"

```bash
make train ARGS="--workers=8 --data=data/divina.jsonl"
make bot   ARGS="--workers=8"
make generate ARGS='--checkpoint=checkpoints/best --prompt="Ciao"'
```

> ⚠️ `make test` esegue `go test ./...`, **non** il mode `test`. Per lo smoke test
> end-to-end usa `./bin/test` (oppure `go run . --mode=test`).

---

## Opzione manuale (senza Makefile)

```bash
# 1. Binario core ottimizzato (-s -w rimuove tabella simboli e info di debug)
go build -ldflags="-s -w" -o bin/discord-llm .

# 2. Un eseguibile per ogni mode (wrapper che inoltrano tutti i flag con "$@")
mkdir -p bin
for m in train bot generate replay harvest import-hf test; do
  printf '#!/usr/bin/env bash\nexec "$(dirname "$0")/discord-llm" --mode=%s "$@"\n' "$m" > "bin/$m"
  chmod +x "bin/$m"
done
```

---

## Eseguibili diretti per ogni mode

Dopo `make all` (o lo script manuale):

```bash
./bin/train     --workers=8 --data=data/divina.jsonl     # pre-training offline (data-parallel)
./bin/bot       --workers=8                               # bot Discord, apprende dalla chat
./bin/generate  --checkpoint=checkpoints/best --prompt="Nel mezzo del cammin"
./bin/replay    --data=data/divina.jsonl                 # ri-allena da un JSONL su disco
./bin/harvest   --channel=123 --out=data/harvest.jsonl   # scarica una chat → dataset JSONL
./bin/import-hf --hf-dataset=maiurilorenzo/divina-commedia --out=data/divina.jsonl
./bin/test                                               # smoke test end-to-end
```

I flag specifici di ogni mode restano disponibili: i wrapper inoltrano tutto con `"$@"`.

---

## Cross-compilation

Go compila per altre piattaforme senza toolchain esterne:

```bash
GOOS=linux   GOARCH=amd64 go build -ldflags="-s -w" -o bin/discord-llm-linux-amd64 .
GOOS=linux   GOARCH=arm64 go build -ldflags="-s -w" -o bin/discord-llm-linux-arm64 .
GOOS=windows GOARCH=amd64 go build -ldflags="-s -w" -o bin/discord-llm.exe .
GOOS=darwin  GOARCH=arm64 go build -ldflags="-s -w" -o bin/discord-llm-mac-arm64 .
```

> Tutte le dipendenze sono pure-Go (incluso SQLite via `modernc.org/sqlite`), quindi la
> cross-compilation **non richiede CGo** né un cross-compiler C.

---

## Note

- **Perché un solo binario**: `package main` fa il dispatch su `--mode`. I "7 eseguibili"
  sono wrapper shell di ~70 byte che chiamano `bin/discord-llm --mode=<x> "$@"`. Leggeri,
  inoltrano ogni flag.
- **Dimensione**: il binario core pesa ~11 MB perché include SQLite pure-Go. `-ldflags="-s -w"`
  lo riduce togliendo simboli e info di debug.
- **`.gitignore`**: aggiungi `bin/` (come `checkpoints/` e `data/`) per non committare i binari.
- **Modes disponibili**: `train | bot | generate | replay | harvest | import-hf | test`.
  Dettagli sui flag di ciascuno nel `README.md` / `CLAUDE.md`.
