// Package bot provides a Discord bot that uses the transformer LLM to generate responses.
package bot

import (
	"fmt"
	"log"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/bwmarrin/discordgo"

	"awesomeProject/generate"
	"awesomeProject/index"
	"awesomeProject/model"
	"awesomeProject/tokenizer"
	"awesomeProject/train"
)

// Config holds Discord bot configuration loaded from .env.
type Config struct {
	Token   string
	GuildID string
}

// ChannelContext maintains the sliding window of messages for one channel.
type ChannelContext struct {
	Messages  []string
	LastReply time.Time
	mu        sync.Mutex
}

// addMessage appends a message to the sliding window, keeping at most 10 entries.
func (c *ChannelContext) addMessage(msg string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.Messages = append(c.Messages, msg)
	if len(c.Messages) > 10 {
		c.Messages = c.Messages[len(c.Messages)-10:]
	}
}

// buildPrompt returns the last N messages joined as a conversation prompt.
func (c *ChannelContext) buildPrompt(newMsg string) string {
	c.mu.Lock()
	defer c.mu.Unlock()
	var sb strings.Builder
	for _, m := range c.Messages {
		sb.WriteString(m)
		sb.WriteString("\n")
	}
	sb.WriteString("User: ")
	sb.WriteString(newMsg)
	sb.WriteString("\nBot:")
	return sb.String()
}

// Bot is the main Discord bot struct.
type Bot struct {
	Session    *discordgo.Session
	Model      *model.TransformerModel
	BPE        *tokenizer.BPE
	Sampler    *generate.Sampler
	Contexts   map[string]*ChannelContext
	ctxMu      sync.RWMutex
	Cfg        Config
	StartTime  time.Time
	SeenHashes map[uint64]bool
	seenMu     sync.Mutex
	commands   []*discordgo.ApplicationCommand

	// Online learning (optional, wired via EnableLearning). When nil, the bot
	// behaves like the original static-model bot.
	Index   *index.MessageIndex
	Trainer *train.OnlineTrainer
	DBPath  string

	// forgetPending tracks armed /forget confirmations (userID → arm time).
	forgetPending map[string]time.Time
	forgetMu      sync.Mutex
}

// EnableLearning attaches a message index and online trainer so the bot indexes every
// message, learns from conversations in real time, and exposes learning slash commands.
func (b *Bot) EnableLearning(idx *index.MessageIndex, tr *train.OnlineTrainer, dbPath string) {
	b.Index = idx
	b.Trainer = tr
	b.DBPath = dbPath
	b.forgetPending = make(map[string]time.Time)
}

// lockModel / unlockModel serialise access to the shared model weights with the online
// trainer. They no-op when learning is disabled.
func (b *Bot) lockModel() {
	if b.Trainer != nil {
		b.Trainer.ModelMu.Lock()
	}
}

func (b *Bot) unlockModel() {
	if b.Trainer != nil {
		b.Trainer.ModelMu.Unlock()
	}
}

// New creates a Bot and opens the Discord WebSocket session.
func New(cfg Config, m *model.TransformerModel, bpe *tokenizer.BPE, samplerCfg generate.Config) (*Bot, error) {
	s, err := discordgo.New("Bot " + cfg.Token)
	if err != nil {
		return nil, fmt.Errorf("create session: %w", err)
	}
	s.Identify.Intents = discordgo.IntentsGuildMessages | discordgo.IntentMessageContent

	b := &Bot{
		Session:    s,
		Model:      m,
		BPE:        bpe,
		Sampler:    generate.New(m, bpe, samplerCfg),
		Contexts:   make(map[string]*ChannelContext),
		Cfg:        cfg,
		StartTime:  time.Now(),
		SeenHashes: make(map[uint64]bool),
	}

	s.AddHandler(b.messageCreate)
	s.AddHandler(b.interactionCreate)

	return b, nil
}

// Start opens the WebSocket connection and blocks until an OS signal is received.
func (b *Bot) Start() error {
	if err := b.Session.Open(); err != nil {
		return fmt.Errorf("open session: %w", err)
	}
	defer b.Session.Close()

	if err := b.registerCommands(); err != nil {
		log.Printf("Warning: could not register slash commands: %v\n", err)
	}

	log.Println("Bot is running. Press CTRL+C to exit.")
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)
	<-stop

	b.removeCommands()
	log.Println("Bot shutting down.")
	return nil
}

// Stop closes the Discord session.
func (b *Bot) Stop() {
	b.removeCommands()
	_ = b.Session.Close()
}

// ────────────────────────────────────────────────────────────────────────────
// Message handler
// ────────────────────────────────────────────────────────────────────────────

// messageCreate is called for every message visible to the bot. Every message is
// indexed for continual learning; the bot then optionally replies (subject to rate
// limiting) and turns the exchange into a fresh training pair.
func (b *Bot) messageCreate(s *discordgo.Session, m *discordgo.MessageCreate) {
	// Ignore messages from the bot itself
	if m.Author.ID == s.State.User.ID {
		return
	}

	channelID := m.ChannelID
	content := strings.TrimSpace(m.Content)
	if content == "" {
		return
	}

	// 1. Index every message — even ones we won't reply to — so the model keeps
	//    learning from the full chat stream. Done off-handler to avoid blocking.
	if b.Index != nil {
		record := index.MessageRecord{
			ID:        m.ID,
			ChannelID: channelID,
			AuthorID:  m.Author.ID,
			Content:   content,
			Timestamp: messageTimestamp(m),
			ReplyToID: referencedMessageID(m),
		}
		go b.Index.Add(record)
	}

	ctx := b.getOrCreateContext(channelID)

	// 2. Rate limiting: at most one reply per 2 seconds per channel. The message is
	//    already indexed above, so learning is unaffected by the reply throttle.
	ctx.mu.Lock()
	if time.Since(ctx.LastReply) < 2*time.Second {
		ctx.mu.Unlock()
		return
	}
	ctx.mu.Unlock()

	// 3. RAG-style retrieval: prepend the most similar past messages as context.
	ragPrefix := ""
	if b.Index != nil {
		queryEmb := b.Index.EmbedText(content)
		similar := b.Index.FindSimilar(queryEmb, 3)
		ragPrefix = buildRAGContext(similar)
	}
	prompt := ragPrefix + ctx.buildPrompt(content)

	// Fire response generation in a goroutine to avoid blocking the handler
	go func() {
		// Send "Bot is typing..." indicator
		_ = s.ChannelTyping(channelID)

		// Generation reads the model weights; lock out concurrent online updates.
		b.lockModel()
		response, err := b.Sampler.Generate(prompt)
		b.unlockModel()
		if err != nil {
			log.Printf("Generation error: %v\n", err)
			return
		}
		response = generate.Format(response)

		// Simulate thinking delay proportional to response length
		delay := generate.ThinkingDelay(len(response))
		time.Sleep(delay)

		// Response variability: skip identical replies
		b.seenMu.Lock()
		response, ok := generate.ResponseVariability(response, b.SeenHashes)
		b.seenMu.Unlock()
		if !ok {
			response = response + "..." // minor variation to break duplicates
		}

		if _, err := s.ChannelMessageSend(channelID, response); err != nil {
			log.Printf("Send error: %v\n", err)
			return
		}

		// 4. Turn the exchange into a supervised training pair for online learning.
		if b.Trainer != nil {
			b.Trainer.Submit(train.TrainingPair{Input: content, Output: response})
		}

		// Update context and rate-limit timestamp
		ctx.addMessage("User: " + content)
		ctx.addMessage("Bot: " + response)
		ctx.mu.Lock()
		ctx.LastReply = time.Now()
		ctx.mu.Unlock()
	}()
}

// messageTimestamp returns the message creation time, falling back to now.
func messageTimestamp(m *discordgo.MessageCreate) time.Time {
	if !m.Timestamp.IsZero() {
		return m.Timestamp
	}
	return time.Now()
}

// referencedMessageID returns the ID of the message this one replies to, if any.
func referencedMessageID(m *discordgo.MessageCreate) string {
	if m.MessageReference != nil {
		return m.MessageReference.MessageID
	}
	return ""
}

// buildRAGContext formats retrieved similar messages as a context preamble.
func buildRAGContext(similar []index.MessageRecord) string {
	if len(similar) == 0 {
		return ""
	}
	var sb strings.Builder
	sb.WriteString("[Contesto rilevante]\n")
	for _, r := range similar {
		line := strings.TrimSpace(r.Content)
		if line == "" {
			continue
		}
		sb.WriteString("- ")
		sb.WriteString(line)
		sb.WriteString("\n")
	}
	sb.WriteString("[Fine contesto]\n")
	return sb.String()
}

// getOrCreateContext retrieves or initialises a ChannelContext for channelID.
func (b *Bot) getOrCreateContext(channelID string) *ChannelContext {
	b.ctxMu.RLock()
	ctx, exists := b.Contexts[channelID]
	b.ctxMu.RUnlock()
	if exists {
		return ctx
	}
	b.ctxMu.Lock()
	defer b.ctxMu.Unlock()
	ctx = &ChannelContext{}
	b.Contexts[channelID] = ctx
	return ctx
}

// ────────────────────────────────────────────────────────────────────────────
// Slash commands
// ────────────────────────────────────────────────────────────────────────────

// registerCommands creates slash commands in the target guild.
func (b *Bot) registerCommands() error {
	cmds := []*discordgo.ApplicationCommand{
		{
			Name:        "reset",
			Description: "Reset conversation context for this channel",
		},
		{
			Name:        "status",
			Description: "Show bot status, model info, and uptime",
		},
		{
			Name:        "stats",
			Description: "Show online-learning stats: indexed messages, steps, loss, DB size",
		},
		{
			Name:        "replay",
			Description: "Replay-train on recent indexed messages",
			Options: []*discordgo.ApplicationCommandOption{
				{
					Type:        discordgo.ApplicationCommandOptionInteger,
					Name:        "n",
					Description: "Number of recent messages to replay (default 200)",
					Required:    false,
				},
			},
		},
		{
			Name:        "learn",
			Description: "ADMIN: fetch this channel's history from Discord, index it, and train on it",
			Options: []*discordgo.ApplicationCommandOption{
				{
					Type:        discordgo.ApplicationCommandOptionInteger,
					Name:        "messages",
					Description: "How many recent messages to learn from (default 1000, 0 = as many as possible)",
					Required:    false,
				},
				{
					Type:        discordgo.ApplicationCommandOptionInteger,
					Name:        "epochs",
					Description: "Training passes over the fetched history (default 2)",
					Required:    false,
				},
			},
		},
		{
			Name:        "forget",
			Description: "ADMIN: erase all learned data and reset model weights",
			Options: []*discordgo.ApplicationCommandOption{
				{
					Type:        discordgo.ApplicationCommandOptionBoolean,
					Name:        "confirm",
					Description: "Set true within 30s of the first call to confirm",
					Required:    false,
				},
			},
		},
	}

	for _, cmd := range cmds {
		created, err := b.Session.ApplicationCommandCreate(b.Session.State.User.ID, b.Cfg.GuildID, cmd)
		if err != nil {
			return fmt.Errorf("create command %s: %w", cmd.Name, err)
		}
		b.commands = append(b.commands, created)
	}
	log.Printf("Registered %d slash commands\n", len(b.commands))
	return nil
}

// removeCommands deletes all registered slash commands on shutdown.
func (b *Bot) removeCommands() {
	for _, cmd := range b.commands {
		_ = b.Session.ApplicationCommandDelete(b.Session.State.User.ID, b.Cfg.GuildID, cmd.ID)
	}
}

// interactionCreate handles slash command interactions.
func (b *Bot) interactionCreate(s *discordgo.Session, i *discordgo.InteractionCreate) {
	if i.Type != discordgo.InteractionApplicationCommand {
		return
	}
	switch i.ApplicationCommandData().Name {
	case "reset":
		b.handleReset(s, i)
	case "status":
		b.handleStatus(s, i)
	case "stats":
		b.handleStats(s, i)
	case "replay":
		b.handleReplay(s, i)
	case "learn":
		b.handleLearn(s, i)
	case "forget":
		b.handleForget(s, i)
	}
}

// handleReset clears conversation context for the current channel.
func (b *Bot) handleReset(s *discordgo.Session, i *discordgo.InteractionCreate) {
	channelID := i.ChannelID
	b.ctxMu.Lock()
	b.Contexts[channelID] = &ChannelContext{}
	b.ctxMu.Unlock()

	_ = s.InteractionRespond(i.Interaction, &discordgo.InteractionResponse{
		Type: discordgo.InteractionResponseChannelMessageWithSource,
		Data: &discordgo.InteractionResponseData{
			Content: "Context reset! Starting fresh 🔄",
		},
	})
}

// handleStatus sends model info and uptime.
func (b *Bot) handleStatus(s *discordgo.Session, i *discordgo.InteractionCreate) {
	uptime := time.Since(b.StartTime).Round(time.Second)
	params := b.Model.NumParameters()
	vocab := b.BPE.VocabLen()

	msg := fmt.Sprintf(
		"**Bot Status**\n"+
			"Uptime: `%v`\n"+
			"Model parameters: `%d` (%.1fM)\n"+
			"Vocabulary size: `%d`\n"+
			"Layers: `%d` | Heads: `%d` | dModel: `%d`",
		uptime,
		params, float64(params)/1e6,
		vocab,
		b.Model.Config.NLayers, b.Model.Config.NHeads, b.Model.Config.DModel,
	)

	_ = s.InteractionRespond(i.Interaction, &discordgo.InteractionResponse{
		Type: discordgo.InteractionResponseChannelMessageWithSource,
		Data: &discordgo.InteractionResponseData{
			Content: msg,
		},
	})
}

// handleStats reports continual-learning metrics.
func (b *Bot) handleStats(s *discordgo.Session, i *discordgo.InteractionCreate) {
	if b.Index == nil || b.Trainer == nil {
		respondText(s, i, "Online learning is not enabled.")
		return
	}
	count, _ := b.Index.Count()
	st := b.Trainer.Stats()
	dbKB := float64(fileSize(b.DBPath)) / 1024.0

	msg := fmt.Sprintf(
		"**Learning Stats**\n"+
			"Indexed messages: `%d`\n"+
			"Online steps: `%d` (flushes `%d`)\n"+
			"Last loss: `%.4f` | Avg loss: `%.4f`\n"+
			"Buffered pairs: `%d`\n"+
			"DB size: `%.1f KB`",
		count, st.Steps, st.Flushes, st.LastLoss, st.AvgLoss, st.Buffered, dbKB,
	)
	respondText(s, i, msg)
}

// handleReplay re-trains the model on recent indexed messages.
func (b *Bot) handleReplay(s *discordgo.Session, i *discordgo.InteractionCreate) {
	if b.Index == nil || b.Trainer == nil {
		respondText(s, i, "Online learning is not enabled.")
		return
	}
	n := 200
	if opts := i.ApplicationCommandData().Options; len(opts) > 0 {
		if v := int(opts[0].IntValue()); v > 0 {
			n = v
		}
	}

	// Acknowledge immediately; training may take a while.
	_ = s.InteractionRespond(i.Interaction, &discordgo.InteractionResponse{
		Type: discordgo.InteractionResponseDeferredChannelMessageWithSource,
	})

	go func() {
		pairs := b.Index.ExportPairs(2)
		if len(pairs) > n {
			pairs = pairs[len(pairs)-n:]
		}
		loss := b.Trainer.ReplayBatch(pairs, 1)
		content := fmt.Sprintf("Replay complete: trained on `%d` pairs, final loss `%.4f`.", len(pairs), loss)
		_, _ = s.FollowupMessageCreate(i.Interaction, true, &discordgo.WebhookParams{Content: content})
	}()
}

// handleForget erases all learned data and resets the weights. Admin only, with a
// two-step confirmation that must complete within 30 seconds.
func (b *Bot) handleForget(s *discordgo.Session, i *discordgo.InteractionCreate) {
	if b.Index == nil || b.Trainer == nil {
		respondText(s, i, "Online learning is not enabled.")
		return
	}
	if !isAdmin(i) {
		respondText(s, i, "⛔ This command requires Administrator permission.")
		return
	}

	userID := interactionUserID(i)
	confirm := false
	if opts := i.ApplicationCommandData().Options; len(opts) > 0 {
		confirm = opts[0].BoolValue()
	}

	if !confirm {
		b.forgetMu.Lock()
		b.forgetPending[userID] = time.Now()
		b.forgetMu.Unlock()
		respondText(s, i, "⚠️ This will **erase all learned data and reset the model weights**.\n"+
			"Re-run `/forget confirm:true` within 30s to confirm ✅.")
		return
	}

	// confirm == true: verify a pending request armed within the last 30s.
	b.forgetMu.Lock()
	armed, ok := b.forgetPending[userID]
	if ok {
		delete(b.forgetPending, userID)
	}
	b.forgetMu.Unlock()

	if !ok || time.Since(armed) > 30*time.Second {
		respondText(s, i, "⏳ No active confirmation (or it expired). Run `/forget` first, then confirm within 30s.")
		return
	}

	if err := b.Index.Forget(); err != nil {
		respondText(s, i, fmt.Sprintf("Failed to clear data: %v", err))
		return
	}
	b.resetWeights()
	respondText(s, i, "🧹 Forgotten all indexed messages and reset model weights to random init.")
}

// resetWeights re-initialises the model parameters in place (random init) and zeroes
// gradients, holding the shared model mutex so no generation/training races with it.
func (b *Bot) resetWeights() {
	fresh := model.NewTransformerModel(b.Model.Config)
	b.lockModel()
	dst := b.Model.Parameters()
	src := fresh.Parameters()
	for i := range dst {
		copy(dst[i].Data, src[i].Data)
		for j := range dst[i].Grad {
			dst[i].Grad[j] = 0
		}
	}
	b.unlockModel()
}

// ────────────────────────────────────────────────────────────────────────────
// Interaction helpers
// ────────────────────────────────────────────────────────────────────────────

// respondText sends a simple text interaction response.
func respondText(s *discordgo.Session, i *discordgo.InteractionCreate, content string) {
	_ = s.InteractionRespond(i.Interaction, &discordgo.InteractionResponse{
		Type: discordgo.InteractionResponseChannelMessageWithSource,
		Data: &discordgo.InteractionResponseData{Content: content},
	})
}

// isAdmin reports whether the invoking member has the Administrator permission.
func isAdmin(i *discordgo.InteractionCreate) bool {
	if i.Member == nil {
		return false
	}
	return i.Member.Permissions&discordgo.PermissionAdministrator != 0
}

// interactionUserID returns the invoking user's ID (guild member or DM user).
func interactionUserID(i *discordgo.InteractionCreate) string {
	if i.Member != nil && i.Member.User != nil {
		return i.Member.User.ID
	}
	if i.User != nil {
		return i.User.ID
	}
	return ""
}

// fileSize returns the size in bytes of a file, or 0 if it cannot be stat'd.
func fileSize(path string) int64 {
	if path == "" {
		return 0
	}
	info, err := os.Stat(path)
	if err != nil {
		return 0
	}
	return info.Size()
}

// ────────────────────────────────────────────────────────────────────────────
// /learn — backfill from Discord history and train on it
// ────────────────────────────────────────────────────────────────────────────

// maxBackfill caps how many historical messages a single /learn can pull, so an
// accidental "whole channel" request can't run away. Raise it if you really need more.
const maxBackfill = 10000

// handleLearn fetches the current channel's recent history from Discord, indexes it, and
// trains the model on the resulting conversational pairs. Admin only (it is heavy).
func (b *Bot) handleLearn(s *discordgo.Session, i *discordgo.InteractionCreate) {
	if b.Index == nil || b.Trainer == nil {
		respondText(s, i, "Online learning is not enabled.")
		return
	}
	if !isAdmin(i) {
		respondText(s, i, "⛔ This command requires Administrator permission.")
		return
	}

	limit, epochs := 1000, 2
	for _, opt := range i.ApplicationCommandData().Options {
		switch opt.Name {
		case "messages":
			limit = int(opt.IntValue())
		case "epochs":
			if v := int(opt.IntValue()); v > 0 {
				epochs = v
			}
		}
	}
	if limit <= 0 || limit > maxBackfill {
		limit = maxBackfill // 0 ("as many as possible") and over-cap both clamp here
	}

	channelID := i.ChannelID

	// Acknowledge now; fetching + training can take a while (up to ~15 min window).
	_ = s.InteractionRespond(i.Interaction, &discordgo.InteractionResponse{
		Type: discordgo.InteractionResponseDeferredChannelMessageWithSource,
	})

	go func() {
		fetched, pairs, loss, err := b.backfillAndLearn(channelID, limit, epochs)
		var content string
		if err != nil {
			content = fmt.Sprintf("Learn failed: %v", err)
		} else if pairs == 0 {
			content = fmt.Sprintf("Fetched %d messages but found no usable pairs to train on.", fetched)
		} else {
			content = fmt.Sprintf("📚 Learned from **%d** messages → **%d** pairs × %d epochs, final loss `%.4f`.",
				fetched, pairs, epochs, loss)
		}
		_, _ = s.FollowupMessageCreate(i.Interaction, true, &discordgo.WebhookParams{Content: content})
	}()
}

// backfillAndLearn pulls up to limit recent messages from a channel, indexes them, builds
// consecutive conversational pairs, and replay-trains on them for the given epochs.
func (b *Bot) backfillAndLearn(channelID string, limit, epochs int) (fetched, pairs int, loss float64, err error) {
	msgs, err := b.fetchHistory(channelID, limit)
	if err != nil {
		return 0, 0, 0, err
	}
	fetched = len(msgs)

	var pairList []train.TrainingPair
	prev := ""
	for _, m := range msgs {
		content := strings.TrimSpace(m.Content)
		if content == "" {
			continue
		}
		// Index without flooding the online trainer; we train in one batch below.
		b.Index.Ingest(index.MessageRecord{
			ID:        m.ID,
			ChannelID: channelID,
			AuthorID:  authorID(m),
			Content:   content,
			Timestamp: m.Timestamp,
			ReplyToID: messageReferenceID(m),
		})
		if prev != "" {
			pairList = append(pairList, train.TrainingPair{Input: prev, Output: content})
		}
		prev = content
	}

	pairs = len(pairList)
	if pairs > 0 {
		// ReplayBatch locks ModelMu per mini-batch (not for the whole run), so the bot
		// stays responsive — just slower — while it learns from history.
		loss = b.Trainer.ReplayBatch(pairList, epochs)
	}
	return fetched, pairs, loss, nil
}

// fetchHistory pulls up to limit messages from a channel (newest first from the API),
// paginating in pages of 100, then returns them in chronological order with the bot's
// own messages removed.
func (b *Bot) fetchHistory(channelID string, limit int) ([]*discordgo.Message, error) {
	var all []*discordgo.Message
	beforeID := ""
	for len(all) < limit {
		page := 100
		if rem := limit - len(all); rem < page {
			page = rem
		}
		batch, err := b.Session.ChannelMessages(channelID, page, beforeID, "", "")
		if err != nil {
			return nil, fmt.Errorf("fetch history: %w", err)
		}
		if len(batch) == 0 {
			break
		}
		all = append(all, batch...)
		beforeID = batch[len(batch)-1].ID // oldest in this page → cursor for older history
		if len(batch) < page {
			break // reached the start of the channel
		}
	}

	// The API returns newest→oldest; reverse to chronological order.
	for x, y := 0, len(all)-1; x < y; x, y = x+1, y-1 {
		all[x], all[y] = all[y], all[x]
	}

	// Drop the bot's own messages so it doesn't train on its past replies.
	var selfID string
	if b.Session.State != nil && b.Session.State.User != nil {
		selfID = b.Session.State.User.ID
	}
	filtered := all[:0]
	for _, m := range all {
		if m.Author != nil && m.Author.ID == selfID {
			continue
		}
		filtered = append(filtered, m)
	}
	return filtered, nil
}

// authorID safely extracts a message author ID.
func authorID(m *discordgo.Message) string {
	if m.Author != nil {
		return m.Author.ID
	}
	return ""
}

// messageReferenceID returns the referenced message ID for a plain Message, if any.
func messageReferenceID(m *discordgo.Message) string {
	if m.MessageReference != nil {
		return m.MessageReference.MessageID
	}
	return ""
}
