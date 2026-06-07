package train

import (
	"sync"

	"awesomeProject/model"
	"awesomeProject/tokenizer"
)

// parallelAccumulate runs forward + backward for a set of (input, target) token sequences
// across `workers` goroutines, each using its own model so their gradients never race:
// worker 0 uses `master`, worker w>0 uses `replicas[w-1]`. Replica weights are first synced
// from the master, then the per-replica gradients are summed back into the master. The
// summed loss and the number of processed examples are returned; the optimiser step is left
// to the caller.
//
// This is numerically identical to sequential gradient accumulation (the per-example
// gradients are the same terms, just summed in a different order). Intra-op matmul
// parallelism is disabled for the duration so W workers don't each spawn matmul goroutines
// (oversubscription). The caller must hold the shared model lock.
func parallelAccumulate(master *model.TransformerModel, replicas []*model.TransformerModel, workers int, inputs, targets [][]int) (sumLoss float64, counted int) {
	prevMM := model.MatMulWorkers()
	model.SetMatMulWorkers(1)
	defer model.SetMatMulWorkers(prevMM)

	masterParams := master.Parameters()

	// Sync replica weights from the master and zero every gradient buffer.
	master.ZeroGrad()
	for _, rep := range replicas {
		rp := rep.Parameters()
		for p := range masterParams {
			copy(rp[p].Data, masterParams[p].Data)
			zeroSlice(rp[p].Grad)
		}
	}

	n := len(inputs)
	type result struct {
		loss float64
		n    int
	}
	results := make([]result, workers)

	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		lo := w * n / workers
		hi := (w + 1) * n / workers
		if lo >= hi {
			continue
		}
		m := master
		if w > 0 {
			m = replicas[w-1]
		}
		wg.Add(1)
		go func(w, lo, hi int, m *model.TransformerModel) {
			defer wg.Done()
			var ls float64
			var c int
			for i := lo; i < hi; i++ {
				logits := m.Forward(inputs[i])
				loss := model.CrossEntropyLoss(logits, targets[i], tokenizer.PAD)
				loss.Backward()
				ls += loss.Data[0]
				c++
			}
			results[w] = result{ls, c}
		}(w, lo, hi, m)
	}
	wg.Wait()

	// Reduce replica gradients into the master's gradient buffers.
	for _, rep := range replicas {
		rp := rep.Parameters()
		for p := range masterParams {
			mg := masterParams[p].Grad
			rg := rp[p].Grad
			for j := range mg {
				mg[j] += rg[j]
			}
		}
	}

	for _, r := range results {
		sumLoss += r.loss
		counted += r.n
	}
	return sumLoss, counted
}

// makeReplicas builds workers-1 model replicas (worker 0 reuses the master). The replicas'
// initial weights don't matter — they are synced from the master before every step.
func makeReplicas(master *model.TransformerModel, workers int) []*model.TransformerModel {
	var replicas []*model.TransformerModel
	for w := 1; w < workers; w++ {
		replicas = append(replicas, model.NewTransformerModel(master.Config))
	}
	return replicas
}
