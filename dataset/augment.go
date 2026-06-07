package dataset

import (
	"math/rand"
	"strings"
)

// synonymsIT maps Italian words to their synonyms for swap augmentation.
var synonymsIT = map[string][]string{
	"bello":      {"fantastico", "stupendo", "magnifico", "meraviglioso"},
	"brutto":     {"orribile", "pessimo", "terribile", "sgradevole"},
	"grande":     {"enorme", "vasto", "immenso", "gigantesco"},
	"piccolo":    {"minuscolo", "ridotto", "esiguo"},
	"buono":      {"ottimo", "eccellente", "fantastico", "splendido"},
	"cattivo":    {"pessimo", "terribile", "orrendo"},
	"veloce":     {"rapido", "celere", "svelto"},
	"lento":      {"pigro", "tardivo", "lentissimo"},
	"felice":     {"contento", "allegro", "gioioso", "beato"},
	"triste":     {"malinconico", "depresso", "abbattuto"},
	"ciao":       {"salve", "hey", "ehi"},
	"grazie":     {"ti ringrazio", "mille grazie", "grazie mille"},
	"sì":         {"certamente", "assolutamente", "certo"},
	"no":         {"assolutamente no", "per niente", "nemmeno"},
	"mangiare":   {"pranzare", "cenare", "consumare"},
	"parlare":    {"chiacchierare", "conversare", "discutere"},
	"andare":     {"spostarsi", "recarsi", "dirigersi"},
	"vedere":     {"osservare", "guardare", "notare"},
	"sapere":     {"conoscere", "essere a conoscenza"},
	"volere":     {"desiderare", "preferire", "scegliere"},
	"pensare":    {"ritenere", "credere", "supporre"},
	"capire":     {"comprendere", "intendere", "afferrare"},
	"aiutare":    {"assistere", "supportare", "soccorrere"},
	"trovare":    {"scoprire", "localizzare", "individuare"},
	"iniziare":   {"cominciare", "avviare", "intraprendere"},
	"finire":     {"concludere", "terminare", "completare"},
	"amico":      {"amicizia", "compagno", "collega"},
	"casa":       {"abitazione", "dimora", "appartamento"},
	"lavoro":     {"occupazione", "impiego", "professione"},
}

// synonymsEN maps English words to their synonyms.
var synonymsEN = map[string][]string{
	"good":      {"great", "excellent", "wonderful", "fantastic"},
	"bad":       {"terrible", "awful", "horrible", "dreadful"},
	"big":       {"large", "huge", "enormous", "gigantic"},
	"small":     {"tiny", "little", "miniature", "compact"},
	"happy":     {"joyful", "pleased", "delighted", "content"},
	"sad":       {"unhappy", "miserable", "sorrowful", "melancholic"},
	"fast":      {"quick", "rapid", "swift", "speedy"},
	"slow":      {"sluggish", "leisurely", "gradual"},
	"hello":     {"hi", "hey", "greetings", "howdy"},
	"thanks":    {"thank you", "many thanks", "cheers"},
	"yes":       {"certainly", "absolutely", "indeed", "sure"},
	"no":        {"not at all", "absolutely not", "nope"},
	"eat":       {"consume", "dine", "have", "munch"},
	"talk":      {"speak", "chat", "discuss", "converse"},
	"go":        {"travel", "move", "head", "proceed"},
	"see":       {"view", "observe", "notice", "spot"},
	"know":      {"understand", "be aware", "realize"},
	"want":      {"desire", "wish", "prefer", "need"},
	"think":     {"believe", "consider", "suppose", "reckon"},
	"understand": {"comprehend", "grasp", "follow", "get"},
	"help":      {"assist", "support", "aid"},
	"find":      {"discover", "locate", "identify"},
	"start":     {"begin", "commence", "initiate"},
	"finish":    {"complete", "end", "conclude"},
	"nice":      {"pleasant", "lovely", "charming"},
	"friend":    {"buddy", "pal", "companion", "mate"},
	"home":      {"house", "residence", "place"},
	"work":      {"job", "occupation", "profession", "task"},
}

// TokenDropout randomly replaces tokens with PAD at the given rate.
func TokenDropout(tokens []int, rate float64, rng *rand.Rand) []int {
	out := make([]int, len(tokens))
	copy(out, tokens)
	for i := range out {
		// Don't drop special tokens (IDs 0-3)
		if out[i] > 3 && rng.Float64() < rate {
			out[i] = 0 // PAD
		}
	}
	return out
}

// SynonymSwap replaces words in text with synonyms from the tables above at the given rate.
func SynonymSwap(text string, rate float64, rng *rand.Rand) string {
	words := strings.Fields(text)
	for i, w := range words {
		lower := strings.ToLower(w)
		// Try Italian then English synonym tables
		if syns, ok := synonymsIT[lower]; ok && rng.Float64() < rate {
			replacement := syns[rng.Intn(len(syns))]
			// Preserve capitalisation
			if len(w) > 0 && w[0] >= 'A' && w[0] <= 'Z' {
				replacement = strings.ToUpper(replacement[:1]) + replacement[1:]
			}
			words[i] = replacement
		} else if syns, ok := synonymsEN[lower]; ok && rng.Float64() < rate {
			replacement := syns[rng.Intn(len(syns))]
			if len(w) > 0 && w[0] >= 'A' && w[0] <= 'Z' {
				replacement = strings.ToUpper(replacement[:1]) + replacement[1:]
			}
			words[i] = replacement
		}
	}
	return strings.Join(words, " ")
}

// BackTranslationSim swaps input and output of a dialog pair to simulate back-translation.
// The model learns bidirectional mapping.
func BackTranslationSim(p DialogPair, rng *rand.Rand) DialogPair {
	if rng.Float64() < 0.3 {
		return DialogPair{Input: p.Output, Output: p.Input, Lang: p.Lang}
	}
	return p
}

// AugmentDataset applies random augmentations to a dataset and returns an augmented copy.
func AugmentDataset(d *Dataset, rng *rand.Rand) *Dataset {
	augmented := make([]DialogPair, 0, len(d.Pairs)*2)
	augmented = append(augmented, d.Pairs...)

	for _, p := range d.Pairs {
		// Synonym swap augmentation
		if rng.Float64() < 0.5 {
			aug := DialogPair{
				Input:  SynonymSwap(p.Input, 0.2, rng),
				Output: SynonymSwap(p.Output, 0.2, rng),
				Lang:   p.Lang,
			}
			augmented = append(augmented, aug)
		}

		// Back-translation simulation
		if rng.Float64() < 0.3 {
			augmented = append(augmented, BackTranslationSim(p, rng))
		}
	}

	return &Dataset{Pairs: augmented, BPE: d.BPE}
}
