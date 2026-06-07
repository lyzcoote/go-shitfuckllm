# Makefile — build helpers for the pure-Go Discord LLM bot.
# The app is a single binary dispatched by --mode; `make wrappers` also emits one small
# executable per mode under bin/ for convenience (./bin/train, ./bin/bot, ...).

BIN      := bin
APP      := $(BIN)/discord-llm
LDFLAGS  := -s -w
# MODES = every wrapper produced under bin/. RUN_MODES = the `make <mode>` shortcuts
# (excludes "test", which is reserved below for `go test`).
MODES    := train bot generate replay harvest import-hf test
RUN_MODES := train bot generate replay harvest import-hf

.PHONY: all build wrappers clean test vet $(RUN_MODES)

## build the optimized single binary
build:
	@mkdir -p $(BIN)
	go build -ldflags="$(LDFLAGS)" -o $(APP) .
	@echo "Built $(APP)"

## build the binary + one wrapper executable per mode
all: build wrappers

wrappers: build
	@for m in $(MODES); do \
		printf '#!/usr/bin/env bash\nexec "$$(dirname "$$0")/discord-llm" --mode=%s "$$@"\n' "$$m" > $(BIN)/$$m; \
		chmod +x $(BIN)/$$m; \
		echo "Wrote $(BIN)/$$m"; \
	done

## convenience: `make bot`, `make train`, ... build then run that mode (pass ARGS="--workers=8")
$(RUN_MODES): build
	./$(APP) --mode=$@ $(ARGS)

test:
	go test ./...

vet:
	go vet ./...

clean:
	rm -rf $(BIN)
