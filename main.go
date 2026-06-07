// discord-llm: a Discord bot with an embedded transformer LLM written in pure Go.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/bwmarrin/discordgo"
	"github.com/joho/godotenv"

	"awesomeProject/bot"
	"awesomeProject/dataset"
	"awesomeProject/generate"
	"awesomeProject/index"
	"awesomeProject/model"
	"awesomeProject/tokenizer"
	"awesomeProject/train"
)

// dbPath is the SQLite store for indexed chat messages.
const dbPath = "data/messages.db"

// onlineLatestDir is where the bot saves (and, by default, resumes) its continuously
// learned weights between sessions.
const onlineLatestDir = "checkpoints/online_latest"

func main() {
	mode := flag.String("mode", "bot", "Execution mode: train | bot | generate | replay | harvest | import-hf | test")
	checkpoint := flag.String("checkpoint", "checkpoints/best", "Path to checkpoint directory")
	prompt := flag.String("prompt", "Ciao! Come stai?", "Prompt for generate mode")
	replayN := flag.Int("n", 500, "Number of recent messages to replay/harvest (replay & harvest modes)")
	workers := flag.Int("workers", 0, "Data-parallel training workers (0=auto=NumCPU, 1=disable)")
	channel := flag.String("channel", "", "Harvest mode: comma-separated channel IDs (empty = all text channels of GUILD_ID)")
	out := flag.String("out", "data/harvest.jsonl", "Harvest mode: output JSONL dataset path")
	appendOut := flag.Bool("append", false, "Harvest mode: append to --out instead of overwriting")
	data := flag.String("data", "", "train/replay modes: JSONL dataset to use instead of the synthetic/DB source")
	hfDataset := flag.String("hf-dataset", "", "import-hf mode: Hugging Face dataset id (e.g. maiurilorenzo/divina-commedia)")
	hfConfig := flag.String("hf-config", "default", "import-hf mode: dataset config/subset name")
	hfSplit := flag.String("hf-split", "train", "import-hf mode: dataset split")
	hfText := flag.String("hf-text", "text", "import-hf mode: single text column → consecutive-line pairs")
	hfInput := flag.String("hf-input", "", "import-hf mode: input column (paired datasets; with --hf-output)")
	hfOutput := flag.String("hf-output", "", "import-hf mode: output column (paired datasets; with --hf-input)")
	hfOffset := flag.Int("hf-offset", 0, "import-hf mode: resume from this row offset (combine with --append)")
	hfDelay := flag.Int("hf-delay", 1000, "import-hf mode: milliseconds to wait between pages (lower with HF_TOKEN set)")
	flag.Parse()

	// Detect whether the user explicitly set --checkpoint (vs. the default). In bot mode
	// this decides between honouring the exact path vs. auto-resuming the online session.
	checkpointExplicit := false
	nExplicit := false
	flag.Visit(func(f *flag.Flag) {
		switch f.Name {
		case "checkpoint":
			checkpointExplicit = true
		case "n":
			nExplicit = true
		}
	})

	// In harvest mode --n caps messages per channel; without it, default to a large
	// cap (disk is cheap and no training happens here) rather than replay's small 500.
	harvestN := *replayN
	if !nExplicit {
		harvestN = defaultHarvestPerChannel
	}

	// import-hf treats --n as a row cap; without it, import the whole dataset (0 = all).
	importN := 0
	if nExplicit {
		importN = *replayN
	}

	// Load .env if present
	if err := godotenv.Load(); err != nil && *mode == "bot" {
		log.Println("Warning: .env file not found, falling back to environment variables.")
	}

	nWorkers := resolveWorkers(*workers)

	switch *mode {
	case "train":
		runTrain(nWorkers, *data)
	case "bot":
		runBot(*checkpoint, checkpointExplicit, nWorkers)
	case "generate":
		runGenerate(*checkpoint, *prompt)
	case "replay":
		runReplay(*checkpoint, *replayN, nWorkers, *data)
	case "harvest":
		runHarvest(*channel, *out, harvestN, *appendOut)
	case "import-hf":
		runImportHF(*hfDataset, *hfConfig, *hfSplit, *hfText, *hfInput, *hfOutput, *out, importN, *hfOffset, *hfDelay, *appendOut)
	case "test":
		runTest()
	default:
		fmt.Fprintf(os.Stderr, "Unknown mode: %s\nValid modes: train, bot, generate, replay, harvest, import-hf, test\n", *mode)
		os.Exit(1)
	}
}

// ────────────────────────────────────────────────────────────────────────────
// Mode: train
// ────────────────────────────────────────────────────────────────────────────

func runTrain(workers int, dataPath string) {
	cfg := train.DefaultConfig()
	cfg.Workers = workers
	// When a custom dataset is supplied (e.g. a harvested channel), train on it instead
	// of the synthetic generator. Delete data/vocab.json first if you want the BPE
	// rebuilt from the new corpus rather than reusing the existing vocabulary.
	if dataPath != "" {
		cfg.DataPath = dataPath
		log.Printf("Training on custom dataset: %s", dataPath)
	}
	t, err := train.New(cfg)
	if err != nil {
		log.Fatalf("Init trainer: %v", err)
	}
	if err := t.Run(); err != nil {
		log.Fatalf("Training failed: %v", err)
	}
}

// ────────────────────────────────────────────────────────────────────────────
// Mode: bot
// ────────────────────────────────────────────────────────────────────────────

// resolveWorkers turns the --workers flag into a concrete count: 0 → NumCPU (auto),
// otherwise the value (clamped to ≥1).
func resolveWorkers(n int) int {
	if n <= 0 {
		return runtime.NumCPU()
	}
	return n
}

func runBot(ckptDir string, explicit bool, workers int) {
	token := os.Getenv("DISCORD_TOKEN")
	if token == "" {
		log.Fatal("DISCORD_TOKEN not set. Add it to .env or the environment.")
	}
	guildID := os.Getenv("GUILD_ID")

	// 1. Decide which checkpoint to start from, then load it — or bootstrap a fresh model.
	//    Without an explicit --checkpoint, resume the previous online session automatically
	//    so continual learning persists across restarts (load path == save path).
	start := resolveBotCheckpoint(ckptDir, explicit)
	var (
		m   *model.TransformerModel
		bpe *tokenizer.BPE
		err error
	)
	if start != "" {
		if m, bpe, err = loadModelAndBPE(start); err != nil {
			log.Printf("Could not load checkpoint %s (%v); initialising a fresh model.", start, err)
			start = ""
		} else {
			log.Printf("Resuming from checkpoint: %s", start)
		}
	}
	if start == "" {
		log.Println("No usable checkpoint; initialising a fresh model (learning will come from chat).")
		if m, bpe, err = freshModel(); err != nil {
			log.Fatalf("Init fresh model: %v", err)
		}
	}

	// 2. Open / create the persistent message store.
	db, err := index.NewMessageDB(dbPath)
	if err != nil {
		log.Fatalf("Open message DB: %v", err)
	}

	// 3. Online trainer (shares the model with the bot via ModelMu).
	onlineCfg := train.DefaultOnlineConfig()
	onlineCfg.SeqLen = m.Config.MaxSeqLen
	onlineCfg.Workers = workers
	trainer := train.NewOnlineTrainer(m, bpe, onlineCfg)

	// 4. Message index, wired to embed via the model and to feed the trainer.
	idx := index.New(db, bpe, m, trainer.ModelMu, trainer.Channel())
	if err := idx.Warm(1000); err != nil {
		log.Printf("Warm index from DB: %v", err)
	}

	// 5. Start the background learning loop.
	ctx, cancel := context.WithCancel(context.Background())
	go trainer.Start(ctx)

	// 6. Connect the Discord bot with learning enabled.
	cfg := bot.Config{Token: token, GuildID: guildID}
	b, err := bot.New(cfg, m, bpe, generate.DefaultConfig())
	if err != nil {
		cancel()
		_ = db.Close()
		log.Fatalf("Create bot: %v", err)
	}
	b.EnableLearning(idx, trainer, dbPath)

	// Start blocks until SIGINT/SIGTERM is received.
	if err := b.Start(); err != nil {
		log.Printf("Bot error: %v", err)
	}

	// 7. Graceful shutdown: stop trainer, flush remaining buffer, save, close DB.
	cancel()
	trainer.ForceFlush()
	if err := trainer.SaveLatest(onlineLatestDir); err != nil {
		log.Printf("Save %s: %v", onlineLatestDir, err)
	} else {
		log.Printf("Saved %s (auto-resumed on next `--mode=bot`)", onlineLatestDir)
	}
	if err := db.Close(); err != nil {
		log.Printf("Close DB: %v", err)
	}
}

// resolveBotCheckpoint chooses where the bot starts from. With an explicit --checkpoint
// the exact path is honoured; otherwise it prefers the last online session, then a
// pre-trained "best" checkpoint, then falls back to a fresh model (empty string).
func resolveBotCheckpoint(ckptDir string, explicit bool) string {
	if explicit {
		return ckptDir
	}
	for _, dir := range []string{onlineLatestDir, ckptDir} {
		if _, err := os.Stat(filepath.Join(dir, "config.json")); err == nil {
			return dir
		}
	}
	return "" // nothing usable → fresh bootstrap
}

// bootstrapVocabSize is the target vocabulary size when bootstrapping a tokenizer from
// the character seed (no dialog dataset involved).
const bootstrapVocabSize = 1024

// freshModel builds an untrained model so the bot can learn entirely from live chat,
// with NO pre-made dialog dataset. If data/vocab.json exists it is reused; otherwise a
// tokenizer is bootstrapped from a built-in character/alphabet seed (not dialog data).
func freshModel() (*model.TransformerModel, *tokenizer.BPE, error) {
	bpe, err := tokenizer.Load("data/vocab.json")
	if err != nil {
		log.Println("No vocabulary found; bootstrapping a tokenizer from a character seed (learning will come from chat).")
		bpe = tokenizer.New()
		bpe.Train(seedCorpus(), bootstrapVocabSize)
		if mkErr := os.MkdirAll("data", 0o755); mkErr != nil {
			return nil, nil, fmt.Errorf("mkdir data: %w", mkErr)
		}
		if saveErr := bpe.Save("data/vocab.json"); saveErr != nil {
			log.Printf("Warning: could not persist bootstrap vocab: %v", saveErr)
		}
		log.Printf("Bootstrapped tokenizer: vocab size = %d", bpe.VocabLen())
	}
	cfg := train.DefaultConfig()
	mc := &model.TransformerConfig{
		VocabSize: bpe.VocabLen(),
		DModel:    cfg.DModel,
		NHeads:    cfg.NHeads,
		NLayers:   cfg.NLayers,
		DFF:       cfg.DFF,
		MaxSeqLen: cfg.SeqLen,
		Dropout:   0.1,
	}
	return model.NewTransformerModel(mc), bpe, nil
}

// seedCorpus returns a built-in character/alphabet seed used to bootstrap the BPE
// tokenizer when the bot starts with no prior dataset. It covers Italian/English letters,
// digits, punctuation, the "→" pair separator, and common emoji so live chat text encodes
// without an explosion of [UNK] tokens. It is NOT dialog training data — the model still
// learns its language behaviour entirely from chat.
func seedCorpus() []string {
	// Each character as its OWN space-separated token. This guarantees character coverage
	// without creating long junk merge-tokens (an earlier version fed the whole alphabet as
	// a single "word", which BPE merged into garbage like "NOPQRSTUVWXYZ").
	var chars []string
	addChars := func(s string) {
		for _, r := range s {
			chars = append(chars, string(r))
		}
	}
	addChars("abcdefghijklmnopqrstuvwxyz")
	addChars("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
	addChars("àèéìíîòóùúÀÈÉÌÒÙ")
	addChars("0123456789")
	addChars(".,!?;:'\"()[]{}-_/\\@#&%+=*<>|~^$→")
	charLine := strings.Join(chars, " ")

	emoji := "😊 😂 😢 😎 🤔 ❤️ 🔥 👍 🎉 😴 🙏 🍕 🎮 🎵 🌿 ✨ 😅 🙂 😉 🥳 😭 👀"
	wordsIT := "ciao come stai bene grazie prego buongiorno buonasera salve arrivederci " +
		"si no forse certo davvero perché che cosa quando dove chi mai sempre niente tutto " +
		"bello brutto grande piccolo amico oggi domani ieri adesso io tu lui lei noi voi loro"
	wordsEN := "hello hi hey how are you fine thanks please good morning evening night " +
		"yes no maybe sure really why what when where who always never nothing everything " +
		"nice ugly big small friend today tomorrow now i me we they it the and or but with"

	return []string{
		charLine, // single-character coverage (each char is its own token)
		emoji,
		wordsIT,
		wordsEN,
		wordsIT + " → " + wordsEN, // exercise the pair separator and natural bigrams
	}
}

// ────────────────────────────────────────────────────────────────────────────
// Mode: replay — re-train on history already stored in the message DB
// ────────────────────────────────────────────────────────────────────────────

func runReplay(ckptDir string, n, workers int, dataPath string) {
	m, bpe, err := loadModelAndBPE(ckptDir)
	if err != nil {
		log.Fatalf("Replay needs a trained checkpoint (%s): %v", ckptDir, err)
	}

	onlineCfg := train.DefaultOnlineConfig()
	onlineCfg.SeqLen = m.Config.MaxSeqLen
	onlineCfg.Workers = workers
	trainer := train.NewOnlineTrainer(m, bpe, onlineCfg)

	// Pairs come either from a JSONL dataset on disk (e.g. a harvested channel) or, by
	// default, from the locally stored message DB — neither touches the Discord API.
	var pairs []train.TrainingPair
	if dataPath != "" {
		pairs, err = loadPairsJSONL(dataPath)
		if err != nil {
			log.Fatalf("Load dataset %s: %v", dataPath, err)
		}
		log.Printf("Loaded %d pairs from %s", len(pairs), dataPath)
	} else {
		db, dberr := index.NewMessageDB(dbPath)
		if dberr != nil {
			log.Fatalf("Open message DB: %v", dberr)
		}
		defer db.Close()
		idx := index.New(db, bpe, m, trainer.ModelMu, nil)
		pairs = idx.ExportPairs(2)
	}

	if n > 0 && len(pairs) > n {
		pairs = pairs[len(pairs)-n:]
	}
	if len(pairs) == 0 {
		log.Println("No pairs to replay. Harvest a channel (--mode=harvest) or chat with the bot first.")
		return
	}

	log.Printf("Replaying %d pairs for 3 epochs...", len(pairs))
	loss := trainer.ReplayBatch(pairs, 3)
	log.Printf("Replay complete. Final loss=%.4f", loss)

	if err := m.Save(ckptDir); err != nil {
		log.Printf("Save updated weights: %v", err)
	} else {
		log.Printf("Saved updated weights → %s", ckptDir)
	}
}

// ────────────────────────────────────────────────────────────────────────────
// Mode: harvest — download channel history once and save it as a JSONL dataset
// ────────────────────────────────────────────────────────────────────────────

// defaultHarvestPerChannel is the per-channel message cap when --n is not given. Harvest
// only writes to disk (no training), so this is deliberately large.
const defaultHarvestPerChannel = 20000

// runHarvest connects to Discord over REST (no gateway), pulls message history from the
// requested channels, builds consecutive conversational pairs, and writes them to a JSONL
// dataset on disk. The point is to pay the Discord API cost ONCE: afterwards you re-train
// as often as you like with `--mode=train --data=<out>` or `--mode=replay --data=<out>`,
// never hitting the API again.
func runHarvest(channelArg, outPath string, perChannel int, appendMode bool) {
	token := os.Getenv("DISCORD_TOKEN")
	if token == "" {
		log.Fatal("DISCORD_TOKEN not set. Add it to .env or the environment.")
	}
	if perChannel <= 0 {
		perChannel = defaultHarvestPerChannel
	}

	s, err := discordgo.New("Bot " + token)
	if err != nil {
		log.Fatalf("Create Discord session: %v", err)
	}
	// Message content over REST requires the (privileged) Message Content intent enabled
	// for the application in the Developer Portal.
	s.Identify.Intents = discordgo.IntentsGuildMessages | discordgo.IntentMessageContent

	channels := resolveHarvestChannels(s, channelArg)
	if len(channels) == 0 {
		log.Fatal("No channels to harvest. Pass --channel=ID[,ID...] or set GUILD_ID to harvest all text channels.")
	}

	var pairs []dataset.DialogPair
	totalMsgs := 0
	for _, ch := range channels {
		msgs, ferr := fetchChannelHistory(s, ch, perChannel)
		if ferr != nil {
			log.Printf("Channel %s: skipped (%v)", ch, ferr)
			continue
		}
		cp := buildHarvestPairs(msgs)
		totalMsgs += len(msgs)
		pairs = append(pairs, cp...)
		log.Printf("Channel %s: %d messages → %d pairs", ch, len(msgs), len(cp))
	}

	if len(pairs) == 0 {
		log.Fatal("Harvested no usable pairs (channels empty, or bot lacks read/Message-Content access).")
	}

	written, err := dataset.WriteJSONL(outPath, pairs, appendMode)
	if err != nil {
		log.Fatalf("Write dataset %s: %v", outPath, err)
	}
	log.Printf("✅ Harvested %d messages from %d channel(s) → %d pairs written to %s",
		totalMsgs, len(channels), written, outPath)
	log.Printf("Re-train offline (no Discord API) with:")
	log.Printf("    go run . --mode=train  --data=%s --workers=0   # fresh model, full schedule", outPath)
	log.Printf("    go run . --mode=replay --data=%s --workers=0   # fine-tune an existing checkpoint", outPath)
}

// resolveHarvestChannels turns the --channel argument (comma-separated IDs) into a slice.
// When empty, it enumerates all text channels of GUILD_ID via REST.
func resolveHarvestChannels(s *discordgo.Session, channelArg string) []string {
	channelArg = strings.TrimSpace(channelArg)
	if channelArg != "" {
		var ids []string
		for _, c := range strings.Split(channelArg, ",") {
			if c = strings.TrimSpace(c); c != "" {
				ids = append(ids, c)
			}
		}
		return ids
	}

	guildID := os.Getenv("GUILD_ID")
	if guildID == "" {
		return nil
	}
	chans, err := s.GuildChannels(guildID)
	if err != nil {
		log.Printf("List guild channels: %v", err)
		return nil
	}
	var ids []string
	for _, c := range chans {
		if c.Type == discordgo.ChannelTypeGuildText {
			ids = append(ids, c.ID)
		}
	}
	log.Printf("No --channel given; harvesting %d text channel(s) from guild %s", len(ids), guildID)
	return ids
}

// fetchChannelHistory pulls up to limit messages from a channel over REST (newest first),
// paginating in pages of 100, then returns them in chronological order with bot-authored
// messages removed. Unlike the bot's live fetch it does not need an open gateway/session.
func fetchChannelHistory(s *discordgo.Session, channelID string, limit int) ([]*discordgo.Message, error) {
	var all []*discordgo.Message
	beforeID := ""
	for len(all) < limit {
		page := 100
		if rem := limit - len(all); rem < page {
			page = rem
		}
		batch, err := s.ChannelMessages(channelID, page, beforeID, "", "")
		if err != nil {
			return nil, fmt.Errorf("fetch history: %w", err)
		}
		if len(batch) == 0 {
			break
		}
		all = append(all, batch...)
		beforeID = batch[len(batch)-1].ID
		if len(batch) < page {
			break // reached the start of the channel
		}
	}

	// API returns newest→oldest; reverse to chronological order.
	for x, y := 0, len(all)-1; x < y; x, y = x+1, y-1 {
		all[x], all[y] = all[y], all[x]
	}

	// Drop bot messages so the dataset is human conversation only.
	filtered := all[:0]
	for _, m := range all {
		if m.Author != nil && m.Author.Bot {
			continue
		}
		filtered = append(filtered, m)
	}
	return filtered, nil
}

// buildHarvestPairs turns a chronological message slice into consecutive (input, output)
// dialog pairs, skipping blank messages.
func buildHarvestPairs(msgs []*discordgo.Message) []dataset.DialogPair {
	var pairs []dataset.DialogPair
	prev := ""
	for _, m := range msgs {
		content := strings.TrimSpace(m.Content)
		if content == "" {
			continue
		}
		if prev != "" {
			pairs = append(pairs, dataset.DialogPair{Input: prev, Output: content, Lang: "und"})
		}
		prev = content
	}
	return pairs
}

// loadPairsJSONL reads a JSONL dialog dataset from disk and converts it into the
// trainer's pair type. Used by replay mode's --data path.
func loadPairsJSONL(path string) ([]train.TrainingPair, error) {
	ds, err := dataset.LoadJSONL(path, nil) // BPE not needed just to parse pairs
	if err != nil {
		return nil, err
	}
	pairs := make([]train.TrainingPair, 0, len(ds.Pairs))
	for _, p := range ds.Pairs {
		if strings.TrimSpace(p.Input) == "" || strings.TrimSpace(p.Output) == "" {
			continue
		}
		pairs = append(pairs, train.TrainingPair{Input: p.Input, Output: p.Output})
	}
	return pairs, nil
}

// ────────────────────────────────────────────────────────────────────────────
// Mode: generate
// ────────────────────────────────────────────────────────────────────────────

func runGenerate(ckptDir, prompt string) {
	m, bpe, err := loadModelAndBPE(ckptDir)
	if err != nil {
		// If no checkpoint, generate from a freshly-initialised (untrained) model
		log.Printf("No checkpoint found (%v), generating from untrained model.\n", err)
		cfg := train.DefaultConfig()
		modelCfg := &model.TransformerConfig{
			VocabSize: 500,
			DModel:    cfg.DModel,
			NHeads:    cfg.NHeads,
			NLayers:   cfg.NLayers,
			DFF:       cfg.DFF,
			MaxSeqLen: cfg.SeqLen,
		}
		m = model.NewTransformerModel(modelCfg)
		bpe = tokenizer.New()
		bpe.Train([]string{prompt, "Ciao come stai bene grazie arrivederci buongiorno"}, 200)
	}

	samplerCfg := generate.DefaultConfig()
	s := generate.New(m, bpe, samplerCfg)

	fmt.Printf("Prompt: %s\n\n", prompt)
	resp, err := s.Generate(prompt)
	if err != nil {
		log.Fatalf("Generate: %v", err)
	}
	fmt.Printf("Response: %s\n", resp)
}

// ────────────────────────────────────────────────────────────────────────────
// Mode: test
// ────────────────────────────────────────────────────────────────────────────

func runTest() {
	// Quick smoke test: generate dataset, train BPE, verify token count
	log.Println("Running smoke tests...")

	n, err := dataset.GenerateDataset("data/smoke_test.jsonl", 99)
	if err != nil {
		log.Fatalf("GenerateDataset: %v", err)
	}
	log.Printf("Dataset: %d pairs\n", n)

	bpe := tokenizer.New()
	bpe.Train([]string{"ciao come stai bene grazie hello how are you fine thanks"}, 64)
	ids := bpe.Encode("ciao")
	text := bpe.Decode(ids)
	log.Printf("BPE encode/decode: 'ciao' → %v → '%s'\n", ids, text)

	cfg := &model.TransformerConfig{
		VocabSize: bpe.VocabLen(),
		DModel:    32,
		NHeads:    2,
		NLayers:   1,
		DFF:       64,
		MaxSeqLen: 16,
	}
	m := model.NewTransformerModel(cfg)
	logits := m.Forward(ids[:min(len(ids), 8)])
	log.Printf("Forward pass: logits shape [%d × %d] ✓\n", logits.Rows, logits.Cols)

	loss := model.CrossEntropyLoss(logits, ids[:min(len(ids), 8)], tokenizer.PAD)
	loss.Backward()
	log.Printf("Loss: %.4f, backward ✓\n", loss.Data[0])

	log.Println("All smoke tests passed ✓")
}

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────

// loadModelAndBPE loads a model from a checkpoint directory together with its tokenizer.
// It prefers a vocab.json bundled inside the checkpoint (self-contained, written by the
// online trainer), falling back to the shared data/vocab.json.
func loadModelAndBPE(ckptDir string) (*model.TransformerModel, *tokenizer.BPE, error) {
	cfg, err := model.LoadConfig(ckptDir)
	if err != nil {
		return nil, nil, fmt.Errorf("load config from %s: %w", ckptDir, err)
	}
	m := model.NewTransformerModel(cfg)
	if err := m.Load(ckptDir); err != nil {
		return nil, nil, fmt.Errorf("load weights: %w", err)
	}

	vocabPath := filepath.Join(ckptDir, "vocab.json")
	if _, statErr := os.Stat(vocabPath); statErr != nil {
		vocabPath = "data/vocab.json"
	}
	bpe, err := tokenizer.Load(vocabPath)
	if err != nil {
		return nil, nil, fmt.Errorf("load vocab from %s: %w", vocabPath, err)
	}
	if bpe.VocabLen() != cfg.VocabSize {
		return nil, nil, fmt.Errorf("vocab/model mismatch: vocab has %d tokens, model expects %d (checkpoint %s)", bpe.VocabLen(), cfg.VocabSize, ckptDir)
	}
	return m, bpe, nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
