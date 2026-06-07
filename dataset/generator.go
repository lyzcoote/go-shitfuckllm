// Package dataset provides dialog data generation, loading, and augmentation.
package dataset

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
)

// DialogPair is one input/output dialog example.
type DialogPair struct {
	Input  string `json:"input"`
	Output string `json:"output"`
	Lang   string `json:"lang"`
}

// GenerateDataset creates 5000+ synthetic Italian/English dialog pairs and writes
// them to outPath in JSONL format.
func GenerateDataset(outPath string, seed int64) (int, error) {
	rng := rand.New(rand.NewSource(seed))
	pairs := buildAllPairs(rng)

	if err := os.MkdirAll(filepath.Dir(outPath), 0o755); err != nil {
		return 0, fmt.Errorf("mkdir: %w", err)
	}
	f, err := os.Create(outPath)
	if err != nil {
		return 0, fmt.Errorf("create %s: %w", outPath, err)
	}
	defer f.Close()

	enc := json.NewEncoder(f)
	for _, p := range pairs {
		if err := enc.Encode(p); err != nil {
			return 0, err
		}
	}
	return len(pairs), nil
}

// ────────────────────────────────────────────────────────────────────────────
// Dialog pair generators
// ────────────────────────────────────────────────────────────────────────────

func buildAllPairs(rng *rand.Rand) []DialogPair {
	var pairs []DialogPair
	pairs = append(pairs, greetingsIT(rng)...)
	pairs = append(pairs, greetingsEN(rng)...)
	pairs = append(pairs, questionsIT(rng)...)
	pairs = append(pairs, questionsEN(rng)...)
	pairs = append(pairs, weatherIT(rng)...)
	pairs = append(pairs, weatherEN(rng)...)
	pairs = append(pairs, foodIT(rng)...)
	pairs = append(pairs, foodEN(rng)...)
	pairs = append(pairs, emotionsIT(rng)...)
	pairs = append(pairs, emotionsEN(rng)...)
	pairs = append(pairs, opinionsIT(rng)...)
	pairs = append(pairs, opinionsEN(rng)...)
	pairs = append(pairs, discordSlangIT(rng)...)
	pairs = append(pairs, discordSlangEN(rng)...)
	pairs = append(pairs, humorIT(rng)...)
	pairs = append(pairs, humorEN(rng)...)
	pairs = append(pairs, shortAnswersIT(rng)...)
	pairs = append(pairs, shortAnswersEN(rng)...)
	pairs = append(pairs, longAnswersIT(rng)...)
	pairs = append(pairs, longAnswersEN(rng)...)
	pairs = append(pairs, formalIT(rng)...)
	pairs = append(pairs, formalEN(rng)...)
	pairs = append(pairs, ironicIT(rng)...)
	pairs = append(pairs, ironicEN(rng)...)
	pairs = append(pairs, numbersIT(rng)...)
	pairs = append(pairs, colorsIT(rng)...)
	pairs = append(pairs, timeIT(rng)...)
	pairs = append(pairs, musicIT(rng)...)
	pairs = append(pairs, gamesIT(rng)...)
	pairs = append(pairs, techIT(rng)...)
	pairs = append(pairs, travelIT(rng)...)
	pairs = append(pairs, schoolIT(rng)...)
	pairs = append(pairs, sportsIT(rng)...)
	pairs = append(pairs, moviesIT(rng)...)
	pairs = append(pairs, complimentsIT(rng)...)
	// Additional combinatorial expansions to reach 5000+
	pairs = append(pairs, combinatorialIT(rng)...)
	pairs = append(pairs, combinatorialEN(rng)...)
	pairs = append(pairs, conversationIT(rng)...)
	pairs = append(pairs, conversationEN(rng)...)
	pairs = append(pairs, emojiChatIT(rng)...)
	pairs = append(pairs, askingAboutIT(rng)...)
	pairs = append(pairs, askingAboutEN(rng)...)
	// Shuffle for variety
	rng.Shuffle(len(pairs), func(i, j int) { pairs[i], pairs[j] = pairs[j], pairs[i] })
	return pairs
}

// pick returns a random element from a string slice.
func pick(rng *rand.Rand, ss []string) string {
	return ss[rng.Intn(len(ss))]
}

// combine generates all combinations of prefix + suffix.
func combine(prefixes, suffixes []string) []DialogPair {
	var out []DialogPair
	for _, p := range prefixes {
		for _, s := range suffixes {
			out = append(out, DialogPair{Input: p, Output: s})
		}
	}
	return out
}

// ────────────────────────────────────────────────────────────────────────────
// Italian categories
// ────────────────────────────────────────────────────────────────────────────

func greetingsIT(rng *rand.Rand) []DialogPair {
	_ = rng
	inputs := []string{
		"Ciao!", "Ciao, come stai?", "Buongiorno!", "Buonasera!", "Salve!",
		"Ehi, ciao!", "Hey!", "Ciao ciao!", "Come va?", "Tutto bene?",
		"Come stai oggi?", "Ciao, come stai amico?", "Buona giornata!",
		"Come va la vita?", "Come stai di questi tempi?", "Ciao bella/o!",
		"Ciao, che fai?", "Come stai, vecchio?", "Ehi, come butta?",
		"Buon pomeriggio!", "Salve, piacere!", "Ciao, ci sei?",
	}
	outputs := []string{
		"Ciao! Tutto bene, grazie!", "Sto benissimo, grazie! E tu?",
		"Buongiorno! Come stai?", "Bene, grazie mille!", "Ciao! Come stai tu?",
		"Tutto bene! Grazie per aver chiesto!", "Va bene, e tu come stai?",
		"Sto alla grande! E tu?", "Molto bene, grazie!", "Bene bene!",
		"Abbastanza bene, grazie!", "Non c'è male, e tu?", "Benissimo!",
		"Dai bene! Come stai tu?", "Bene grazie, e dalla tua parte?",
		"Hola! Sto proprio bene oggi!", "Perfetto! E tu come stai?",
		"Bene! Grazie, ci si vede!", "Alla grande! E tu?", "Bene grazie!",
		"Non male, non male. E tu?", "Meglio di ieri!",
	}
	var pairs []DialogPair
	for i, inp := range inputs {
		pairs = append(pairs, DialogPair{Input: inp, Output: outputs[i%len(outputs)], Lang: "it"})
	}
	// Add more variations
	extra := combine(inputs[:10], outputs[:10])
	for i := range extra {
		extra[i].Lang = "it"
	}
	return append(pairs, extra...)
}

func greetingsEN(rng *rand.Rand) []DialogPair {
	_ = rng
	inputs := []string{
		"Hello!", "Hey, how are you?", "Good morning!", "Good evening!", "Hi there!",
		"What's up?", "How's it going?", "Hey!", "How are you doing?", "Sup?",
		"How have you been?", "Hey buddy!", "Howdy!", "What's good?", "Greetings!",
		"How's everything?", "Long time no see!", "What's new?", "How's life?",
		"Good to see you!", "Hey, you okay?", "How's your day?",
	}
	outputs := []string{
		"Hello! How are you?", "I'm doing great, thanks!", "Good morning to you too!",
		"Hey! All good here.", "Hi! I'm fine, thanks.", "Not much, just chilling!",
		"Going well, thanks for asking!", "Hey! What's up?", "Doing well, thanks!",
		"All good! You?", "Been good, you?", "Hey! What's up buddy?",
		"Howdy! All good!", "Things are good!", "Greetings to you too!",
		"Everything's great!", "Yeah, been a while!", "Nothing much!",
		"Life's good, you?", "Good to see you too!", "Yeah all good, thanks!",
		"Pretty good! You?",
	}
	var pairs []DialogPair
	for i, inp := range inputs {
		pairs = append(pairs, DialogPair{Input: inp, Output: outputs[i%len(outputs)], Lang: "en"})
	}
	extra := combine(inputs[:10], outputs[:10])
	for i := range extra {
		extra[i].Lang = "en"
	}
	return append(pairs, extra...)
}

func questionsIT(rng *rand.Rand) []DialogPair {
	topics := []string{"musica", "film", "libri", "sport", "cibo", "viaggi", "tecnologia", "anime", "serie tv", "videogiochi"}
	questions := []string{
		"Qual è il tuo %s preferito?",
		"Ti piace il %s?",
		"Cosa ne pensi del %s?",
		"Hai mai provato %s?",
		"Mi consigli qualcosa di %s?",
	}
	answers := []string{
		"Bella domanda! Direi che mi piace molto il %s.",
		"Sì, il %s mi appassiona moltissimo!",
		"Mmm, sul %s ho opinioni miste...",
		"Non ne so molto di %s, onestamente.",
		"Per il %s, ti consiglio di esplorare!",
	}
	var pairs []DialogPair
	for _, topic := range topics {
		for _, q := range questions {
			for _, a := range answers {
				pairs = append(pairs, DialogPair{
					Input:  fmt.Sprintf(q, topic),
					Output: fmt.Sprintf(a, topic),
					Lang:   "it",
				})
			}
		}
	}
	_ = rng
	return pairs
}

func questionsEN(rng *rand.Rand) []DialogPair {
	topics := []string{"music", "movies", "books", "sports", "food", "travel", "tech", "anime", "TV shows", "games"}
	questions := []string{
		"What's your favorite %s?",
		"Do you like %s?",
		"What do you think about %s?",
		"Have you ever tried %s?",
		"Can you recommend some %s?",
	}
	answers := []string{
		"Great question! I'd say I really enjoy %s.",
		"Yes, I'm really into %s!",
		"Hmm, I have mixed feelings about %s...",
		"I don't know much about %s, honestly.",
		"For %s, I'd say just explore around!",
	}
	var pairs []DialogPair
	for _, topic := range topics {
		for _, q := range questions {
			for _, a := range answers {
				pairs = append(pairs, DialogPair{
					Input:  fmt.Sprintf(q, topic),
					Output: fmt.Sprintf(a, topic),
					Lang:   "en",
				})
			}
		}
	}
	_ = rng
	return pairs
}

func weatherIT(_ *rand.Rand) []DialogPair {
	inputs := []string{
		"Che tempo fa oggi?", "Piove?", "Fa caldo?", "Fa freddo oggi?",
		"Nevica?", "Com'è il tempo?", "Devo prendere l'ombrello?",
		"Che bel tempo!", "Che brutto tempo!", "Quanto fa caldo oggi?",
		"È nuvoloso?", "C'è il sole?", "Che giornata grigia!",
		"Che cielo limpido!", "Pioverà domani?",
	}
	outputs := []string{
		"Oggi c'è il sole, una bellissima giornata!", "Sì, piove un po', prendi l'ombrello!",
		"Sì, fa abbastanza caldo oggi!", "Sì, mettiti una giacca!",
		"No, non nevica ancora.", "Il tempo è variabile oggi.",
		"Meglio sì, potrebbe piovere!", "Sì, finalmente una bella giornata!",
		"Brutto sì, speriamo passi presto.", "Oggi ci sono quasi 30 gradi!",
		"Sì, un po' nuvoloso ma niente pioggia.", "Sì, splende il sole!",
		"Già, giornata uggiosa... prendiamo un caffè?", "Bellissimo cielo oggi!",
		"Le previsioni dicono sì, porta l'ombrello!",
	}
	var pairs []DialogPair
	for i, inp := range inputs {
		pairs = append(pairs, DialogPair{Input: inp, Output: outputs[i%len(outputs)], Lang: "it"})
	}
	return pairs
}

func weatherEN(_ *rand.Rand) []DialogPair {
	inputs := []string{
		"What's the weather like?", "Is it raining?", "Is it hot?", "Is it cold today?",
		"Is it snowing?", "How's the weather?", "Should I bring an umbrella?",
		"Beautiful day!", "Terrible weather!", "How hot is it today?",
		"Is it cloudy?", "Is it sunny?", "What a gloomy day!",
	}
	outputs := []string{
		"It's sunny today, a beautiful day!", "Yes, it's raining a bit, bring an umbrella!",
		"Yes, it's quite hot today!", "Yes, wear a jacket!",
		"No, not snowing yet.", "The weather is variable today.",
		"Better yes, it might rain!", "Yes, finally a nice day!",
		"Yeah, bad weather, hope it passes soon.", "It's almost 30 degrees today!",
		"Yes, a bit cloudy but no rain.", "Yes, the sun is shining!",
		"Yeah, gloomy day... let's get a coffee?",
	}
	var pairs []DialogPair
	for i, inp := range inputs {
		pairs = append(pairs, DialogPair{Input: inp, Output: outputs[i%len(outputs)], Lang: "en"})
	}
	return pairs
}

func foodIT(_ *rand.Rand) []DialogPair {
	inputs := []string{
		"Hai mangiato?", "Cosa mangi di solito?", "Ti piace la pizza?",
		"Qual è il tuo cibo preferito?", "Hai fame?", "Cosa hai mangiato oggi?",
		"Ti piace cucinare?", "Qual è il miglior ristorante che conosci?",
		"Ti piace il sushi?", "Preferisci dolce o salato?",
		"Mangi la pasta ogni giorno?", "Hai una ricetta preferita?",
		"Ti piace il caffè?", "Hai provato la cucina giapponese?",
		"Preferisci pizza o pasta?",
	}
	outputs := []string{
		"Sì! Ho mangiato poco fa, ero morto di fame!", "Di solito mangio pasta o pizza, classico italiano!",
		"Sì, la pizza è fantastica! Margherita forever!", "Il mio cibo preferito? La pasta al pomodoro!",
		"Un po' sì, potrei mangiare qualcosa...", "Oggi ho mangiato pasta, ovviamente!",
		"Sì, adoro cucinare! È rilassante.", "C'è un posto qui vicino incredibile!",
		"Sì, il sushi è delizioso!", "Dipende dall'umore, ma tendenzialmente salato!",
		"Non ogni giorno, ma spesso sì!", "La mia nonna ha una ricetta segreta per il ragù!",
		"Sì! Non posso vivere senza caffè al mattino!", "Sì, la cucina giapponese è fantastica!",
		"Pizza! Non c'è dubbio.",
	}
	var pairs []DialogPair
	for i, inp := range inputs {
		pairs = append(pairs, DialogPair{Input: inp, Output: outputs[i%len(outputs)], Lang: "it"})
	}
	return pairs
}

func foodEN(_ *rand.Rand) []DialogPair {
	inputs := []string{
		"Have you eaten?", "What do you usually eat?", "Do you like pizza?",
		"What's your favorite food?", "Are you hungry?", "What did you eat today?",
		"Do you like cooking?", "What's the best restaurant you know?",
		"Do you like sushi?", "Do you prefer sweet or savory?",
	}
	outputs := []string{
		"Yes! Just ate, I was starving!", "I usually eat pasta or pizza!",
		"Yes, pizza is amazing! Margherita forever!", "My favorite food? Pasta with tomato sauce!",
		"A little, I could eat something...", "I had pasta today, of course!",
		"Yes, I love cooking! It's relaxing.", "There's an amazing place nearby!",
		"Yes, sushi is delicious!", "Depends on the mood, but usually savory!",
	}
	var pairs []DialogPair
	for i, inp := range inputs {
		pairs = append(pairs, DialogPair{Input: inp, Output: outputs[i%len(outputs)], Lang: "en"})
	}
	return pairs
}

func emotionsIT(_ *rand.Rand) []DialogPair {
	inputs := []string{
		"Sono triste.", "Sono felice!", "Mi sento stanco.", "Sono arrabbiato.",
		"Sono preoccupato.", "Mi annoio.", "Sono eccitato!", "Mi sento solo.",
		"Sono stressato.", "Mi sento bene oggi!", "Sono depresso.",
		"Oggi non è la mia giornata.", "Mi sento fantastico!", "Sono nervoso.",
		"Sono emozionato!", "Mi sento giù.", "Sono contento!",
	}
	outputs := []string{
		"Mi dispiace, cosa è successo? Posso aiutarti?",
		"Che bello! Sono contento per te! Cosa è successo?",
		"Capita, prenditi una pausa e rilassati!",
		"Capisco, vuoi parlarne? A volte aiuta!",
		"Non preoccuparti troppo, le cose si sistemano!",
		"Noia? Proviamo a trovare qualcosa di divertente!",
		"Wow, raccontami! Perché sei così eccitato?",
		"Non sei solo, sono qui io! Parliamo.",
		"Lo stress fa male, cerca di rilassarti!",
		"Fantastico! Cosa ti ha reso felice oggi?",
		"Mi dispiace molto, ti sono vicino.",
		"Domani andrà meglio, vedrai!",
		"Ottimo! Condividi la buona energia!",
		"Respira, andrà tutto bene!",
		"Raccontami! Sono tutto orecchi!",
		"Ehi, che è successo? Parlami!",
		"Sono contento per te!",
	}
	var pairs []DialogPair
	for i, inp := range inputs {
		pairs = append(pairs, DialogPair{Input: inp, Output: outputs[i%len(outputs)], Lang: "it"})
	}
	return pairs
}

func emotionsEN(_ *rand.Rand) []DialogPair {
	inputs := []string{
		"I'm sad.", "I'm happy!", "I feel tired.", "I'm angry.",
		"I'm worried.", "I'm bored.", "I'm excited!", "I feel lonely.",
		"I'm stressed.", "I feel great today!", "I'm depressed.",
		"Today's not my day.", "I feel fantastic!", "I'm nervous.",
	}
	outputs := []string{
		"I'm sorry, what happened? Can I help?",
		"That's great! I'm happy for you! What happened?",
		"It happens, take a break and relax!",
		"I understand, want to talk about it? It sometimes helps!",
		"Don't worry too much, things will work out!",
		"Bored? Let's find something fun to do!",
		"Wow, tell me! Why are you so excited?",
		"You're not alone, I'm here! Let's talk.",
		"Stress is bad, try to relax!",
		"Fantastic! What made you happy today?",
		"I'm really sorry, I'm here for you.",
		"Tomorrow will be better, you'll see!",
		"Great! Spread the good energy!",
		"Breathe, everything will be okay!",
	}
	var pairs []DialogPair
	for i, inp := range inputs {
		pairs = append(pairs, DialogPair{Input: inp, Output: outputs[i%len(outputs)], Lang: "en"})
	}
	return pairs
}

func opinionsIT(_ *rand.Rand) []DialogPair {
	inputs := []string{
		"Cosa pensi dei social media?", "Qual è la tua opinione sulla politica?",
		"Pensi che l'AI cambierà il mondo?", "Cosa ne pensi del cambiamento climatico?",
		"Pensi che i videogiochi facciano male?", "Qual è la tua opinione sullo smart working?",
		"Cosa pensi degli influencer?", "Pensi che il futuro sarà migliore?",
		"Cosa ne pensi della globalizzazione?", "Hai un'opinione sulle criptovalute?",
	}
	outputs := []string{
		"I social media sono un'arma a doppio taglio, secondo me.",
		"La politica è complicata, preferisco non sbilanciarmi.",
		"Sì, l'AI sta già cambiando tutto, bello e spaventoso insieme!",
		"Il cambiamento climatico è una realtà urgente che dobbiamo affrontare.",
		"No, i videogiochi possono essere positivi se usati con moderazione.",
		"Lo smart working ha pro e contro, dipende dalla persona.",
		"Gli influencer sono il nuovo marketing, nel bene e nel male.",
		"Spero di sì! Ci sono tante cose da migliorare ma sono ottimista.",
		"La globalizzazione porta benefici ma anche disuguaglianze.",
		"Le criptovalute sono interessanti ma molto volatili e rischiose.",
	}
	var pairs []DialogPair
	for i, inp := range inputs {
		pairs = append(pairs, DialogPair{Input: inp, Output: outputs[i%len(outputs)], Lang: "it"})
	}
	return pairs
}

func opinionsEN(_ *rand.Rand) []DialogPair {
	inputs := []string{
		"What do you think about social media?", "What's your opinion on politics?",
		"Do you think AI will change the world?", "What about climate change?",
		"Do you think video games are harmful?", "What's your take on remote work?",
		"What do you think of influencers?", "Do you think the future will be better?",
	}
	outputs := []string{
		"Social media is a double-edged sword, in my opinion.",
		"Politics is complicated, I prefer not to take sides.",
		"Yes, AI is already changing everything, exciting and scary!",
		"Climate change is an urgent reality we need to address.",
		"No, video games can be positive if used in moderation.",
		"Remote work has pros and cons, depends on the person.",
		"Influencers are the new marketing, for better or worse.",
		"I hope so! There's a lot to improve but I'm optimistic.",
	}
	var pairs []DialogPair
	for i, inp := range inputs {
		pairs = append(pairs, DialogPair{Input: inp, Output: outputs[i%len(outputs)], Lang: "en"})
	}
	return pairs
}

func discordSlangIT(_ *rand.Rand) []DialogPair {
	inputs := []string{
		"lol", "lmao", "gg", "wp", "noob", "afk", "brb", "omg",
		"wtf", "irl", "tbh", "ngl", "fr fr", "based", "cringe",
		"pog", "Kappa", "pepega", "monkaS", "PauseChamp",
		"sei troppo forte", "sei una leggenda", "che lag!",
		"mi hai killato!", "ratio!", "L + ratio", "touch grass",
	}
	outputs := []string{
		"haha sì!", "lol esatto!", "gg wp bella partita!", "grazie, ci ho messo impegno!",
		"dai non sono così noob lol", "torno subito!", "vado e torno!", "omg davvero?!",
		"lol che succede?", "nella vita reale è diverso eh!", "onestamente sì!",
		"non mentirò, hai ragione!", "assolutamente vero!", "sì decisamente!",
		"un po' sì lol", "POGGGG!", "Kappa sì!", "pepega moment!",
		"monkaS che situazione!", "PauseChamp...", "grazie sei gentilissimo!",
		"sei tu la leggenda!", "il lag mi uccide!", "ti vendicherò!",
		"ratio + L + skill issue!", "nah sei tu a fare ratio fail!",
		"touch grass? mai sentito 🌿",
	}
	var pairs []DialogPair
	for i, inp := range inputs {
		pairs = append(pairs, DialogPair{Input: inp, Output: outputs[i%len(outputs)], Lang: "it"})
	}
	return pairs
}

func discordSlangEN(_ *rand.Rand) []DialogPair {
	inputs := []string{
		"lol", "lmao", "gg", "wp", "noob", "afk", "brb", "omg",
		"wtf just happened?", "irl though", "tbh you're right", "ngl that was crazy",
		"fr fr no cap", "that's based", "cringe moment", "POG",
		"you're so good!", "what a lag!", "you killed me!", "ratio!",
		"skill issue tbh", "touch grass dude", "W + no cap",
	}
	outputs := []string{
		"haha yeah!", "lmao ikr!", "gg wp good game!", "thanks, I tried!",
		"I'm not that much of a noob lol", "be right back!", "brb real quick!", "omg seriously?!",
		"lol what happened?", "irl it's different tho!", "honestly yeah!",
		"ngl you got me there!", "absolutely fr!", "yeah based!",
		"a little cringe yeah lol", "POGGERS!", "you killed it!", "lag is killing me!",
		"I'll get you back!", "ratio + L + skill issue!", "no YOU have a skill issue!",
		"touch grass? never heard of it 🌿", "W indeed!",
	}
	var pairs []DialogPair
	for i, inp := range inputs {
		pairs = append(pairs, DialogPair{Input: inp, Output: outputs[i%len(outputs)], Lang: "en"})
	}
	return pairs
}

func humorIT(_ *rand.Rand) []DialogPair {
	inputs := []string{
		"Dimmi una barzelletta!", "Fai ridere!", "Dì qualcosa di divertente!",
		"Hai delle barzellette?", "Raccontami qualcosa di buffo.",
		"Perché il pollo ha attraversato la strada?",
		"Cosa fa un pesce quando è annoiato?",
		"Come si chiama un cane senza zampe?",
	}
	outputs := []string{
		"Perché Babbo Natale è un bugiardo? Perché dice 'lo-ho-ho' ma non ride davvero!",
		"Un matematico entra in un bar. Chiede: 'Quanto fa 1+1?' Il barista: '2.' 'Bene, allora uno dei miei.'",
		"Cosa dice un muro a un altro muro? 'Ci vediamo all'angolo!'",
		"Sì! Come si chiama un pesce senza occhi? Un pessc!",
		"Ho sentito che gli spazzacamini di questi tempi sono molto cupi...",
		"Per arrivare dall'altra parte! (Classico ma sempreverde!)",
		"Nuota in tondo, ovviamente!",
		"Freddo! Puoi chiamarlo come vuoi, tanto non viene!",
	}
	var pairs []DialogPair
	for i, inp := range inputs {
		pairs = append(pairs, DialogPair{Input: inp, Output: outputs[i%len(outputs)], Lang: "it"})
	}
	return pairs
}

func humorEN(_ *rand.Rand) []DialogPair {
	inputs := []string{
		"Tell me a joke!", "Make me laugh!", "Say something funny!",
		"Do you know any jokes?", "Tell me something funny.",
		"Why did the chicken cross the road?",
		"What do you call a fish without eyes?",
	}
	outputs := []string{
		"Why don't scientists trust atoms? Because they make up everything!",
		"I told my wife she should embrace her mistakes... she gave me a hug.",
		"Why did the scarecrow win an award? Because he was outstanding in his field!",
		"Yes! What do you call a lazy kangaroo? A pouch potato!",
		"I asked my dog what 2 minus 2 is. He said nothing.",
		"To get to the other side! (Classic!)",
		"A fsh! (No eyes, no 'i'!)",
	}
	var pairs []DialogPair
	for i, inp := range inputs {
		pairs = append(pairs, DialogPair{Input: inp, Output: outputs[i%len(outputs)], Lang: "en"})
	}
	return pairs
}

func shortAnswersIT(_ *rand.Rand) []DialogPair {
	pairs := []DialogPair{
		{"Sì o no?", "Sì!", "it"},
		{"Sei d'accordo?", "Sì, assolutamente!", "it"},
		{"Sei sicuro?", "Sì, sicurissimo!", "it"},
		{"Ok?", "Ok!", "it"},
		{"Capito?", "Sì, capito!", "it"},
		{"Pronto?", "Pronto!", "it"},
		{"Ci sei?", "Sì, ci sono!", "it"},
		{"Sei lì?", "Sì, eccomi!", "it"},
		{"Vai?", "Sì, vado!", "it"},
		{"Fatto?", "Fatto!", "it"},
		{"Bene?", "Benissimo!", "it"},
		{"Male?", "No, per niente!", "it"},
		{"Dai?", "Sì, dai!", "it"},
		{"Vero?", "Verissimo!", "it"},
		{"Giusto?", "Giusto!", "it"},
	}
	return pairs
}

func shortAnswersEN(_ *rand.Rand) []DialogPair {
	pairs := []DialogPair{
		{"Yes or no?", "Yes!", "en"},
		{"Do you agree?", "Yes, absolutely!", "en"},
		{"Are you sure?", "Yes, definitely!", "en"},
		{"OK?", "OK!", "en"},
		{"Got it?", "Yes, got it!", "en"},
		{"Ready?", "Ready!", "en"},
		{"You there?", "Yes, I'm here!", "en"},
		{"Going?", "Yes, going!", "en"},
		{"Done?", "Done!", "en"},
		{"Good?", "Great!", "en"},
		{"Bad?", "Not at all!", "en"},
		{"Really?", "Yes, really!", "en"},
		{"True?", "Absolutely true!", "en"},
		{"Right?", "Right!", "en"},
	}
	return pairs
}

func longAnswersIT(_ *rand.Rand) []DialogPair {
	pairs := []DialogPair{
		{
			"Spiegami come funziona il machine learning.",
			"Il machine learning è un ramo dell'intelligenza artificiale che permette alle macchine di imparare dai dati senza essere esplicitamente programmate. Si basa su algoritmi che analizzano grandi quantità di dati, trovano pattern e fanno previsioni. Ci sono tre tipi principali: supervised learning (con dati etichettati), unsupervised learning (senza etichette) e reinforcement learning (apprendimento per rinforzo). È alla base di moltissime applicazioni moderne!",
			"it",
		},
		{
			"Cosa è la blockchain?",
			"La blockchain è una tecnologia di registro distribuito dove i dati sono organizzati in blocchi collegati tra loro crittograficamente. È decentralizzata, cioè non c'è un'autorità centrale che la controlla. Ogni transazione viene verificata da più nodi della rete. È alla base delle criptovalute come Bitcoin ed Ethereum, ma ha applicazioni anche in altri settori come la logistica, la sanità e il voto elettronico.",
			"it",
		},
		{
			"Come posso imparare a programmare?",
			"Per imparare a programmare, ti consiglio di iniziare con un linguaggio semplice come Python. Poi, pratica ogni giorno con piccoli progetti. Usa risorse online gratuite come freeCodeCamp, Khan Academy o YouTube. Quando ti senti pronto, prova a contribuire a progetti open source. La cosa più importante è non arrendersi mai e continuare a praticare! Il coding diventa naturale con il tempo.",
			"it",
		},
		{
			"Parlami dell'Italia.",
			"L'Italia è un paese meraviglioso nel centro del Mediterraneo, famoso per la sua storia millenaria, l'arte, la cucina e il design. Ha dato i natali a grandi artisti come Leonardo da Vinci, Michelangelo e Raffaello. La sua cucina è rinomata in tutto il mondo: pizza, pasta, gelato... L'Italia è anche il paese con il maggior numero di siti UNESCO al mondo. Da nord a sud, ogni regione ha la sua cultura, dialetto e tradizioni uniche.",
			"it",
		},
		{
			"Cosa ne pensi dell'intelligenza artificiale?",
			"L'intelligenza artificiale è una delle tecnologie più rivoluzionarie della nostra epoca. Sta trasformando settori come la medicina, i trasporti, l'educazione e l'intrattenimento. Da un lato offre opportunità incredibili: diagnosi mediche più accurate, auto a guida autonoma, assistenti personali. Dall'altro pone sfide etiche importanti: la privacy, il lavoro, i bias algoritmici. Penso che dobbiamo svilupparla in modo responsabile e inclusivo.",
			"it",
		},
	}
	return pairs
}

func longAnswersEN(_ *rand.Rand) []DialogPair {
	pairs := []DialogPair{
		{
			"Explain how machine learning works.",
			"Machine learning is a branch of AI that allows machines to learn from data without being explicitly programmed. It relies on algorithms that analyze large amounts of data, find patterns, and make predictions. There are three main types: supervised learning (with labeled data), unsupervised learning (without labels), and reinforcement learning. It's behind many modern applications like recommendation systems, image recognition, and natural language processing!",
			"en",
		},
		{
			"What is blockchain?",
			"Blockchain is a distributed ledger technology where data is organized in cryptographically linked blocks. It's decentralized, meaning no central authority controls it. Each transaction is verified by multiple network nodes. It's the foundation of cryptocurrencies like Bitcoin and Ethereum, but also has applications in logistics, healthcare, and electronic voting.",
			"en",
		},
		{
			"How can I learn to code?",
			"To learn coding, I recommend starting with a simple language like Python. Then practice daily with small projects. Use free online resources like freeCodeCamp, Khan Academy or YouTube. When you feel ready, try contributing to open source projects. The most important thing is never giving up and keep practicing! Coding becomes natural with time.",
			"en",
		},
	}
	return pairs
}

func formalIT(_ *rand.Rand) []DialogPair {
	inputs := []string{
		"Buongiorno, come posso aiutarla?", "La ringrazio per la sua risposta.",
		"Potrebbe ripetere, per favore?", "Scusi il disturbo.",
		"Come posso essere utile?", "La prego di attendere un momento.",
		"Grazie mille per il suo tempo.", "Con piacere, prego.",
	}
	outputs := []string{
		"Buongiorno! Sono qui per assisterla in tutto ciò di cui ha bisogno.",
		"Prego, è stato un piacere!", "Certo, con molto piacere!",
		"Non si preoccupi, sono qui apposta!", "Sono a sua completa disposizione.",
		"Certo, prendo nota immediatamente.", "Grazie a lei per la fiducia!",
		"Grazie, è stato un piacere assisterla!",
	}
	var pairs []DialogPair
	for i, inp := range inputs {
		pairs = append(pairs, DialogPair{Input: inp, Output: outputs[i%len(outputs)], Lang: "it"})
	}
	return pairs
}

func formalEN(_ *rand.Rand) []DialogPair {
	inputs := []string{
		"Good morning, how may I assist you?", "Thank you for your response.",
		"Could you please repeat that?", "I apologize for the interruption.",
		"How may I be of service?", "Please hold for a moment.",
	}
	outputs := []string{
		"Good morning! I'm here to assist you with anything you need.",
		"You're welcome, it was a pleasure!", "Of course, with pleasure!",
		"No need to apologize, I'm here to help!", "I'm at your complete disposal.",
		"Of course, I'll take note immediately.",
	}
	var pairs []DialogPair
	for i, inp := range inputs {
		pairs = append(pairs, DialogPair{Input: inp, Output: outputs[i%len(outputs)], Lang: "en"})
	}
	return pairs
}

func ironicIT(_ *rand.Rand) []DialogPair {
	inputs := []string{
		"Sei un genio!", "Che idea originale!", "Come non averci pensato prima!",
		"Ma dai, davvero?!", "Certo, come no!", "Ovvio!", "Impossibile!",
		"Stai scherzando vero?", "Non ci credo!", "Sembra troppo bello per essere vero!",
	}
	outputs := []string{
		"Sì sì, un genio... mi sono quasi dimenticato dove ho messo le chiavi stamattina!",
		"Lo so, sono irresistibilmente creativo, cosa vuoi farci!",
		"Beh, a volte le cose ovvie sfuggono!",
		"Ma davvero davvero! Non ci sembra possibile neanche a noi!",
		"Ma certo! Come potrebbe essere altrimenti?!",
		"Ovvio, ovviamente! Cristallino!",
		"Impossibile? Eppure eccoci qui!",
		"No no, parlavo serissimamente!",
		"Eppure è tutto vero!",
		"Lo sembra, sì... procedi con cautela!",
	}
	var pairs []DialogPair
	for i, inp := range inputs {
		pairs = append(pairs, DialogPair{Input: inp, Output: outputs[i%len(outputs)], Lang: "it"})
	}
	return pairs
}

func ironicEN(_ *rand.Rand) []DialogPair {
	inputs := []string{
		"You're a genius!", "What an original idea!", "Why didn't I think of that!",
		"Really?!", "Sure, why not!", "Obviously!", "Impossible!",
		"You're kidding right?", "I can't believe it!", "Sounds too good to be true!",
	}
	outputs := []string{
		"Yes yes, a genius... I almost forgot where I put my keys this morning!",
		"I know, I'm irresistibly creative, what can I say!",
		"Well, obvious things sometimes slip by!",
		"Really really! We can't believe it either!",
		"Of course! How could it be otherwise?!",
		"Obviously, obviously! Crystal clear!",
		"Impossible? Yet here we are!",
		"No no, I was dead serious!",
		"And yet it's all true!",
		"It does seem so... proceed with caution!",
	}
	var pairs []DialogPair
	for i, inp := range inputs {
		pairs = append(pairs, DialogPair{Input: inp, Output: outputs[i%len(outputs)], Lang: "en"})
	}
	return pairs
}

func numbersIT(_ *rand.Rand) []DialogPair {
	var pairs []DialogPair
	for i := 0; i <= 20; i++ {
		pairs = append(pairs, DialogPair{
			Input:  fmt.Sprintf("Quanto fa %d + %d?", i, i),
			Output: fmt.Sprintf("Fa %d!", i+i),
			Lang:   "it",
		})
		if i > 0 {
			pairs = append(pairs, DialogPair{
				Input:  fmt.Sprintf("Quanto fa %d x %d?", i, 2),
				Output: fmt.Sprintf("Fa %d!", i*2),
				Lang:   "it",
			})
		}
	}
	return pairs
}

func colorsIT(_ *rand.Rand) []DialogPair {
	colors := []string{"rosso", "blu", "verde", "giallo", "arancione", "viola", "rosa", "nero", "bianco", "grigio"}
	var pairs []DialogPair
	for _, c := range colors {
		pairs = append(pairs, DialogPair{
			Input:  "Qual è il tuo colore preferito?",
			Output: fmt.Sprintf("Mi piace molto il %s!", c),
			Lang:   "it",
		})
		pairs = append(pairs, DialogPair{
			Input:  fmt.Sprintf("Ti piace il %s?", c),
			Output: fmt.Sprintf("Sì, il %s è un bel colore!", c),
			Lang:   "it",
		})
	}
	return pairs
}

func timeIT(_ *rand.Rand) []DialogPair {
	var pairs []DialogPair
	hours := []string{"8:00", "9:00", "10:00", "12:00", "13:00", "15:00", "18:00", "20:00", "22:00"}
	for _, h := range hours {
		pairs = append(pairs, DialogPair{
			Input:  "Che ore sono?",
			Output: fmt.Sprintf("Sono le %s!", h),
			Lang:   "it",
		})
	}
	days := []string{"lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica"}
	for _, d := range days {
		pairs = append(pairs, DialogPair{
			Input:  "Che giorno è oggi?",
			Output: fmt.Sprintf("Oggi è %s!", d),
			Lang:   "it",
		})
	}
	return pairs
}

func musicIT(_ *rand.Rand) []DialogPair {
	genres := []string{"rock", "pop", "jazz", "classica", "hip-hop", "metal", "elettronica", "indie", "reggae", "blues"}
	var pairs []DialogPair
	for _, g := range genres {
		pairs = append(pairs, DialogPair{
			Input:  fmt.Sprintf("Ti piace la musica %s?", g),
			Output: fmt.Sprintf("Sì, la musica %s è fantastica!", g),
			Lang:   "it",
		})
		pairs = append(pairs, DialogPair{
			Input:  "Che musica ascolti?",
			Output: fmt.Sprintf("In questo momento sto ascoltando molto %s!", g),
			Lang:   "it",
		})
	}
	return pairs
}

func gamesIT(_ *rand.Rand) []DialogPair {
	games := []string{"Minecraft", "Fortnite", "League of Legends", "Valorant", "Among Us", "Elden Ring", "FIFA", "GTA", "Cyberpunk", "Zelda"}
	var pairs []DialogPair
	for _, g := range games {
		pairs = append(pairs, DialogPair{
			Input:  fmt.Sprintf("Hai mai giocato a %s?", g),
			Output: fmt.Sprintf("Sì! %s è un gioco fantastico, ci ho passato tantissime ore!", g),
			Lang:   "it",
		})
		pairs = append(pairs, DialogPair{
			Input:  fmt.Sprintf("Ti piace %s?", g),
			Output: fmt.Sprintf("Sì, %s è uno dei miei preferiti!", g),
			Lang:   "it",
		})
	}
	return pairs
}

func techIT(_ *rand.Rand) []DialogPair {
	pairs := []DialogPair{
		{"Usi Linux o Windows?", "Uso Linux per sviluppo e Windows per gaming!", "it"},
		{"Qual è il tuo linguaggio di programmazione preferito?", "Amo Python e Go, ma ogni linguaggio ha i suoi punti di forza!", "it"},
		{"Hai un Mac?", "Sì, uso Mac per lavoro, è molto comodo!", "it"},
		{"Cosa pensi di ChatGPT?", "È impressionante! L'AI sta facendo passi da gigante.", "it"},
		{"Sai programmare?", "Sì, un po'! Sto ancora imparando ma mi piace molto.", "it"},
		{"Che phone usi?", "Ho uno smartphone Android, mi trovo benissimo!", "it"},
		{"Usi VS Code?", "Sì, VS Code è il mio editor preferito in assoluto!", "it"},
		{"Git o SVN?", "Git senza dubbio! È lo standard de facto.", "it"},
		{"Cloud o on-premise?", "Dipende dal caso d'uso, ma il cloud offre molta flessibilità!", "it"},
		{"Docker è utile?", "Assolutamente sì! Docker ha rivoluzionato il deployment.", "it"},
	}
	return pairs
}

func travelIT(_ *rand.Rand) []DialogPair {
	cities := []string{"Roma", "Milano", "Napoli", "Venezia", "Firenze", "Tokyo", "Parigi", "New York", "Londra", "Barcellona"}
	var pairs []DialogPair
	for _, city := range cities {
		pairs = append(pairs, DialogPair{
			Input:  fmt.Sprintf("Sei mai stato/a a %s?", city),
			Output: fmt.Sprintf("Sì! %s è una città meravigliosa, ti consiglio di visitarla!", city),
			Lang:   "it",
		})
		pairs = append(pairs, DialogPair{
			Input:  fmt.Sprintf("Vorresti andare a %s?", city),
			Output: fmt.Sprintf("%s è nella mia lista dei posti da visitare assolutamente!", city),
			Lang:   "it",
		})
	}
	return pairs
}

func schoolIT(_ *rand.Rand) []DialogPair {
	subjects := []string{"matematica", "storia", "inglese", "fisica", "chimica", "informatica", "arte", "musica", "filosofia", "biologia"}
	var pairs []DialogPair
	for _, s := range subjects {
		pairs = append(pairs, DialogPair{
			Input:  fmt.Sprintf("Ti piaceva la %s a scuola?", s),
			Output: fmt.Sprintf("La %s era interessante! Anche se a volte difficile.", s),
			Lang:   "it",
		})
	}
	pairs = append(pairs, DialogPair{"Stai studiando?", "Sì, ho un esame presto! Devo darci dentro!", "it"})
	pairs = append(pairs, DialogPair{"Quante ore studi al giorno?", "Dipende, di solito 3-4 ore. Non di più o il cervello va in pappa!", "it"})
	pairs = append(pairs, DialogPair{"Hai la laurea?", "Sì! È stato un percorso lungo ma ne valeva la pena.", "it"})
	return pairs
}

func sportsIT(_ *rand.Rand) []DialogPair {
	sports := []string{"calcio", "basket", "tennis", "nuoto", "ciclismo", "pallavolo", "boxe", "atletica", "golf", "rugby"}
	var pairs []DialogPair
	for _, s := range sports {
		pairs = append(pairs, DialogPair{
			Input:  fmt.Sprintf("Ti piace il %s?", s),
			Output: fmt.Sprintf("Sì, il %s è uno sport fantastico! Lo seguo spesso.", s),
			Lang:   "it",
		})
		pairs = append(pairs, DialogPair{
			Input:  fmt.Sprintf("Pratichi il %s?", s),
			Output: fmt.Sprintf("Un po' sì, anche se non sono un professionista del %s!", s),
			Lang:   "it",
		})
	}
	pairs = append(pairs, DialogPair{"Hai visto la partita ieri?", "Sì! Che partita incredibile, non ci speravo più!", "it"})
	pairs = append(pairs, DialogPair{"Chi vincerà il campionato?", "Difficile dirlo, quest'anno è tutto aperto!", "it"})
	return pairs
}

func moviesIT(_ *rand.Rand) []DialogPair {
	movies := []string{
		"Interstellar", "Il Padrino", "Inception", "Matrix", "Avengers",
		"Il Signore degli Anelli", "Star Wars", "Titanic", "La La Land", "Joker",
	}
	var pairs []DialogPair
	for _, m := range movies {
		pairs = append(pairs, DialogPair{
			Input:  fmt.Sprintf("Hai visto %s?", m),
			Output: fmt.Sprintf("Sì! %s è un capolavoro, l'ho adorato!", m),
			Lang:   "it",
		})
	}
	pairs = append(pairs, DialogPair{"Che film mi consigli?", "Ti consiglio Interstellar, è un film che ti lascia senza parole!", "it"})
	pairs = append(pairs, DialogPair{"Preferisci film o serie TV?", "Dipende dall'umore! Le serie TV per le serate lunghe, i film per quando ho poco tempo.", "it"})
	return pairs
}

func complimentsIT(_ *rand.Rand) []DialogPair {
	inputs := []string{
		"Sei molto bravo!", "Sei intelligente!", "Mi piace come pensi.",
		"Sei creativo!", "Hai un grande senso dell'umorismo!", "Sei molto gentile!",
		"Sei simpaticissimo!", "Parli molto bene!", "Sei molto preparato!",
		"Sei una brava persona!", "Sei fantastico/a!", "Sei incredibile!",
	}
	outputs := []string{
		"Grazie mille, sei troppo gentile!", "Oh! Grazie, mi fa molto piacere!",
		"Grazie! Cerco sempre di fare del mio meglio.", "Davvero? Grazie, mi fa piacere!",
		"Haha, grazie! Mi ci vuole ogni tanto!", "Grazie, cerco di esserlo!",
		"Anche tu sei simpaticissimo/a!", "Grazie mille, ci lavoro su!",
		"Oh, grazie! Ho ancora molto da imparare!", "Grazie, fa piacere sentirlo!",
		"Sei tu quello/a fantastico/a!", "Grazie! Sei molto gentile!",
	}
	var pairs []DialogPair
	for i, inp := range inputs {
		pairs = append(pairs, DialogPair{Input: inp, Output: outputs[i%len(outputs)], Lang: "it"})
	}
	return pairs
}

// ────────────────────────────────────────────────────────────────────────────
// Extra generators for reaching 5000+ pairs
// ────────────────────────────────────────────────────────────────────────────

func combinatorialIT(_ *rand.Rand) []DialogPair {
	subjects := []string{"tu", "lui", "lei", "noi", "voi", "loro", "un amico", "mia sorella"}
	verbs := []string{
		"mangi", "dormi", "corri", "studi", "lavori", "giochi", "guardi", "leggi",
		"ascolti", "parli", "cucini", "esci", "viaggi", "nuoti", "balli",
	}
	places := []string{
		"a casa", "al lavoro", "in palestra", "al parco", "in ufficio",
		"al bar", "a scuola", "in biblioteca", "in piscina", "al cinema",
	}
	responses := []string{
		"Sì, di solito!", "A volte sì!", "Non sempre.", "Spesso!", "Raramente.",
		"Dipende dal giorno.", "Quasi sempre!", "Solo quando posso.", "Ogni tanto!",
		"Non proprio, perché?",
	}
	var pairs []DialogPair
	for _, s := range subjects {
		for _, v := range verbs {
			for _, p := range places {
				input := fmt.Sprintf("Di solito %s %s %s?", s, v, p)
				output := responses[len(pairs)%len(responses)]
				pairs = append(pairs, DialogPair{Input: input, Output: output, Lang: "it"})
			}
		}
	}
	return pairs
}

func combinatorialEN(_ *rand.Rand) []DialogPair {
	subjects := []string{"you", "he", "she", "we", "they", "a friend", "your sister", "people"}
	verbs := []string{
		"eat", "sleep", "run", "study", "work", "play", "watch", "read",
		"listen", "talk", "cook", "go out", "travel", "swim", "dance",
	}
	places := []string{
		"at home", "at work", "at the gym", "in the park", "at the office",
		"at the bar", "at school", "in the library", "at the pool", "at the cinema",
	}
	responses := []string{
		"Yes, usually!", "Sometimes yes!", "Not always.", "Often!", "Rarely.",
		"Depends on the day.", "Almost always!", "Only when I can.", "Occasionally!",
		"Not really, why?",
	}
	var pairs []DialogPair
	for _, s := range subjects {
		for _, v := range verbs {
			for _, p := range places {
				input := fmt.Sprintf("Does %s %s %s?", s, v, p)
				output := responses[len(pairs)%len(responses)]
				pairs = append(pairs, DialogPair{Input: input, Output: output, Lang: "en"})
			}
		}
	}
	return pairs
}

func conversationIT(_ *rand.Rand) []DialogPair {
	starters := []string{
		"Cosa hai fatto oggi?", "Come è andata la giornata?", "Hai novità?",
		"Sei occupato?", "Cosa stai facendo?", "Hai tempo di parlare?",
		"Ti racconto una cosa!", "Hai sentito la notizia?", "Cosa pensi di questo?",
		"Ho bisogno di un consiglio!", "Puoi aiutarmi?", "Posso chiederti una cosa?",
		"Sei disponibile?", "Stai bene?", "Come ti senti oggi?",
		"Hai dormito bene?", "Hai mangiato?", "Dove sei stato oggi?",
		"Cosa hai in programma per questo weekend?", "Hai visto qualcosa di bello ultimamente?",
	}
	replies := []string{
		"Ho avuto una giornata abbastanza tranquilla!", "È andata bene, grazie per aver chiesto!",
		"Sì, dimmi! Sono tutto orecchi.", "No, sono libero. Parlami!", "Sto rilassandomi un po'.",
		"Certo, dimmi pure!", "Forza, racconta!", "No, cosa è successo?", "È interessante...",
		"Certo, sono qui per aiutarti!", "Dimmi pure, ci provo!", "Sì, chiedimi!",
		"Sì, sono qua!", "Sì, grazie per aver chiesto!", "Un po' stanco ma ok!",
		"Non malissimo, grazie!", "Sì ho mangiato poco fa!", "Ho girato un po' per la città.",
		"Non ho ancora deciso, e tu?", "Sì! Ho visto un film fantastico!",
	}
	var pairs []DialogPair
	for i, s := range starters {
		pairs = append(pairs, DialogPair{Input: s, Output: replies[i%len(replies)], Lang: "it"})
	}
	// Full cross-product expansion
	for _, s := range starters {
		for _, r := range replies {
			pairs = append(pairs, DialogPair{Input: s, Output: r, Lang: "it"})
		}
	}
	return pairs
}

func conversationEN(_ *rand.Rand) []DialogPair {
	starters := []string{
		"What did you do today?", "How was your day?", "Any news?",
		"Are you busy?", "What are you up to?", "Got a minute to chat?",
		"Let me tell you something!", "Did you hear the news?", "What do you think about this?",
		"I need some advice!", "Can you help me?", "Can I ask you something?",
		"Are you available?", "Are you okay?", "How are you feeling today?",
		"Did you sleep well?", "Have you eaten?", "Where did you go today?",
		"What are your plans for this weekend?", "Seen anything good lately?",
	}
	replies := []string{
		"Had a pretty quiet day!", "It went well, thanks for asking!",
		"Yes, tell me! I'm all ears.", "No, I'm free. Talk to me!", "Just chilling a bit.",
		"Sure, go ahead!", "Go on, tell me!", "No, what happened?", "That's interesting...",
		"Sure, I'm here to help!", "Tell me, I'll try!", "Yes, ask me!",
		"Yes, I'm here!", "Yes, thanks for asking!", "A bit tired but okay!",
		"Not too bad, thanks!", "Yes, just ate!", "Walked around the city.",
		"Haven't decided yet, you?", "Yes! Watched an amazing movie!",
	}
	var pairs []DialogPair
	for i, s := range starters {
		pairs = append(pairs, DialogPair{Input: s, Output: replies[i%len(replies)], Lang: "en"})
	}
	for _, s := range starters {
		for _, r := range replies {
			pairs = append(pairs, DialogPair{Input: s, Output: r, Lang: "en"})
		}
	}
	return pairs
}

func emojiChatIT(_ *rand.Rand) []DialogPair {
	emojiInputs := []string{
		"😊", "😂", "😢", "😎", "🤔", "❤️", "🔥", "👍", "🎉", "😴",
		"😊😊", "haha 😂", "aww 😢", "wow 🔥", "love it ❤️",
		"bro 😎", "thinking 🤔", "yes 👍", "lets go 🎉", "sleepy 😴",
	}
	responses := []string{
		"Anche io sono felice! 😊", "Haha sì! 😂", "Dai non essere triste! 😢",
		"Sei troppo cool! 😎", "Ci sto pensando anch'io... 🤔", "Grazie, anche io! ❤️",
		"Assolutamente fire! 🔥", "Grazie! 👍", "Grande festa! 🎉", "Anche io ho sonno! 😴",
		"Doppia felicità! 😊", "Troppo divertente! 😂", "Ci sono, dimmi tutto.",
		"Incredibile davvero!", "Concordo pienamente!", "Stile unico!",
		"Domanda legittima!", "Approvato!", "Pronti partenza via!", "Buonanotte!",
	}
	var pairs []DialogPair
	for i, inp := range emojiInputs {
		pairs = append(pairs, DialogPair{Input: inp, Output: responses[i%len(responses)], Lang: "it"})
	}
	return pairs
}

func askingAboutIT(_ *rand.Rand) []DialogPair {
	things := []string{
		"il tuo hobby preferito", "la tua serie TV preferita", "il tuo libro preferito",
		"il tuo animale preferito", "la tua vacanza preferita", "il tuo sport preferito",
		"il tuo personaggio famoso preferito", "il tuo supereroe preferito",
		"la tua stagione preferita", "il tuo genere musicale preferito",
		"il tuo piatto preferito", "la tua città preferita", "il tuo linguaggio di programmazione",
		"il tuo film preferito di tutti i tempi", "il tuo artista preferito",
	}
	questions := []string{
		"Qual è %s?", "Puoi dirmi %s?", "Mi parli di %s?",
		"Cosa pensi di %s?", "Hai un preferito per %s?",
	}
	answers := []string{
		"Bella domanda! Per me è difficile sceglierne uno solo.",
		"Ce ne sono troppi buoni da scegliere!",
		"Ho tante passioni, ma se devo sceglierne una...",
		"Dipende dall'umore! Ma di solito preferisco...",
		"Ne ho diversi, ma il primo che mi viene in mente è...",
	}
	var pairs []DialogPair
	for _, thing := range things {
		for _, q := range questions {
			for _, a := range answers {
				pairs = append(pairs, DialogPair{
					Input:  fmt.Sprintf(q, thing),
					Output: a,
					Lang:   "it",
				})
			}
		}
	}
	return pairs
}

func askingAboutEN(_ *rand.Rand) []DialogPair {
	things := []string{
		"your favorite hobby", "your favorite TV show", "your favorite book",
		"your favorite animal", "your favorite vacation spot", "your favorite sport",
		"your favorite celebrity", "your favorite superhero",
		"your favorite season", "your favorite music genre",
		"your favorite dish", "your favorite city", "your favorite programming language",
		"your all-time favorite movie", "your favorite artist",
	}
	questions := []string{
		"What is %s?", "Can you tell me %s?", "Tell me about %s?",
		"What do you think of %s?", "Do you have a favorite for %s?",
	}
	answers := []string{
		"Great question! It's hard for me to pick just one.",
		"There are too many good ones to choose from!",
		"I have so many passions, but if I had to pick one...",
		"Depends on my mood! But usually I prefer...",
		"I have several, but the first one that comes to mind is...",
	}
	var pairs []DialogPair
	for _, thing := range things {
		for _, q := range questions {
			for _, a := range answers {
				pairs = append(pairs, DialogPair{
					Input:  fmt.Sprintf(q, thing),
					Output: a,
					Lang:   "en",
				})
			}
		}
	}
	return pairs
}

// pick is used to avoid unused import of strings
var _ = strings.TrimSpace
