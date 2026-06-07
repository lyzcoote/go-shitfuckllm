package model

import (
	"math/rand"
	"sync"
)

var (
	globalRandMu sync.Mutex
	globalRNG    = rand.New(rand.NewSource(42))
)

// globalRand returns a uniform random float64 in [0, 1).
func globalRand() float64 {
	globalRandMu.Lock()
	defer globalRandMu.Unlock()
	return globalRNG.Float64()
}

// SetSeed reseeds the global random number generator.
func SetSeed(seed int64) {
	globalRandMu.Lock()
	defer globalRandMu.Unlock()
	globalRNG = rand.New(rand.NewSource(seed))
}

// randNorm returns a normally distributed float64 (Box-Muller).
func randNorm() float64 {
	globalRandMu.Lock()
	defer globalRandMu.Unlock()
	return globalRNG.NormFloat64()
}
