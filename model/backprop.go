package model

import "math"

// ────────────────────────────────────────────────────────────────────────────
// Autograd operations — each returns a new Tensor with backwardFn set.
// ────────────────────────────────────────────────────────────────────────────

// Add returns a + b element-wise. Supports (m×n)+(m×n) and (m×n)+(1×n) broadcast.
func Add(a, b *Tensor) *Tensor {
	broadcast := (b.Rows == 1 && a.Rows > 1)
	out := NewTensor(a.Rows, a.Cols, a.RequiresGrad || b.RequiresGrad)
	for r := 0; r < a.Rows; r++ {
		br := 0
		if !broadcast {
			br = r
		}
		for c := 0; c < a.Cols; c++ {
			out.Data[r*a.Cols+c] = a.Data[r*a.Cols+c] + b.Data[br*b.Cols+c]
		}
	}
	out.children = []*Tensor{a, b}
	out.backwardFn = func() {
		if a.RequiresGrad {
			rawAdd(a.Grad, out.Grad)
		}
		if b.RequiresGrad {
			if broadcast {
				for r := 0; r < out.Rows; r++ {
					for c := 0; c < out.Cols; c++ {
						b.Grad[c] += out.Grad[r*out.Cols+c]
					}
				}
			} else {
				rawAdd(b.Grad, out.Grad)
			}
		}
	}
	return out
}

// MatMul returns a @ b. a is [m×k], b is [k×n], out is [m×n].
func MatMul(a, b *Tensor) *Tensor {
	out := NewTensor(a.Rows, b.Cols, a.RequiresGrad || b.RequiresGrad)
	result := rawMatMul(a.Data, b.Data, a.Rows, a.Cols, b.Cols)
	copy(out.Data, result)
	out.children = []*Tensor{a, b}
	out.backwardFn = func() {
		if a.RequiresGrad {
			// dA += dOut @ B^T
			bT := rawTranspose(b.Data, b.Rows, b.Cols)
			dA := rawMatMul(out.Grad, bT, out.Rows, out.Cols, b.Rows)
			rawAdd(a.Grad, dA)
		}
		if b.RequiresGrad {
			// dB += A^T @ dOut
			aT := rawTranspose(a.Data, a.Rows, a.Cols)
			dB := rawMatMul(aT, out.Grad, a.Cols, a.Rows, out.Cols)
			rawAdd(b.Grad, dB)
		}
	}
	return out
}

// Scale returns a * s element-wise (s is a constant scalar).
func Scale(a *Tensor, s float64) *Tensor {
	out := NewTensor(a.Rows, a.Cols, a.RequiresGrad)
	for i, v := range a.Data {
		out.Data[i] = v * s
	}
	out.children = []*Tensor{a}
	out.backwardFn = func() {
		if a.RequiresGrad {
			for i, g := range out.Grad {
				a.Grad[i] += g * s
			}
		}
	}
	return out
}

// SoftmaxRows applies row-wise softmax to a and returns the result.
func SoftmaxRows(a *Tensor) *Tensor {
	out := NewTensor(a.Rows, a.Cols, a.RequiresGrad)
	copy(out.Data, a.Data)
	rawSoftmaxRows(out.Data, out.Rows, out.Cols)
	out.children = []*Tensor{a}
	out.backwardFn = func() {
		if !a.RequiresGrad {
			return
		}
		// dX[i] = Σ_j dY[j] * Y[i,j] * (δ_ij - Y[i,j])
		cols := out.Cols
		for r := 0; r < out.Rows; r++ {
			y := out.Data[r*cols : r*cols+cols]
			dy := out.Grad[r*cols : r*cols+cols]
			dx := a.Grad[r*cols : r*cols+cols]
			dotYdY := 0.0
			for j := range y {
				dotYdY += y[j] * dy[j]
			}
			for i := range y {
				dx[i] += y[i] * (dy[i] - dotYdY)
			}
		}
	}
	return out
}

// GELU applies the GELU activation element-wise.
func GELU(a *Tensor) *Tensor {
	out := NewTensor(a.Rows, a.Cols, a.RequiresGrad)
	for i, v := range a.Data {
		out.Data[i] = geluVal(v)
	}
	out.children = []*Tensor{a}
	out.backwardFn = func() {
		if a.RequiresGrad {
			for i, v := range a.Data {
				a.Grad[i] += out.Grad[i] * geluGrad(v)
			}
		}
	}
	return out
}

// MaskFill sets positions where mask is true to val (used for causal masking).
// mask has shape [rows×cols] matching a.
func MaskFill(a *Tensor, mask []bool, val float64) *Tensor {
	out := NewTensor(a.Rows, a.Cols, a.RequiresGrad)
	copy(out.Data, a.Data)
	for i, m := range mask {
		if m {
			out.Data[i] = val
		}
	}
	out.children = []*Tensor{a}
	out.backwardFn = func() {
		if a.RequiresGrad {
			for i, m := range mask {
				if !m {
					a.Grad[i] += out.Grad[i]
				}
				// masked positions contribute zero gradient
			}
		}
	}
	return out
}

// Reshape returns a view of a with new dimensions (no data copy; same backing slice).
// The total element count must match.
func Reshape(a *Tensor, rows, cols int) *Tensor {
	out := &Tensor{
		Data:         a.Data,
		Grad:         a.Grad,
		Rows:         rows,
		Cols:         cols,
		RequiresGrad: a.RequiresGrad,
		children:     []*Tensor{a},
	}
	out.backwardFn = func() {} // grad already shared via pointer
	return out
}

// SliceCols returns a new Tensor containing columns [start, end) of a.
func SliceCols(a *Tensor, start, end int) *Tensor {
	width := end - start
	out := NewTensor(a.Rows, width, a.RequiresGrad)
	for r := 0; r < a.Rows; r++ {
		copy(out.Data[r*width:r*width+width], a.Data[r*a.Cols+start:r*a.Cols+end])
	}
	out.children = []*Tensor{a}
	out.backwardFn = func() {
		if a.RequiresGrad {
			for r := 0; r < a.Rows; r++ {
				for c := 0; c < width; c++ {
					a.Grad[r*a.Cols+start+c] += out.Grad[r*width+c]
				}
			}
		}
	}
	return out
}

// ConcatCols concatenates tensors along the column axis. All must have same Rows.
func ConcatCols(tensors ...*Tensor) *Tensor {
	rows := tensors[0].Rows
	totalCols := 0
	for _, t := range tensors {
		totalCols += t.Cols
	}
	requiresGrad := false
	for _, t := range tensors {
		if t.RequiresGrad {
			requiresGrad = true
		}
	}
	out := NewTensor(rows, totalCols, requiresGrad)
	colOffset := 0
	for _, t := range tensors {
		for r := 0; r < rows; r++ {
			copy(out.Data[r*totalCols+colOffset:r*totalCols+colOffset+t.Cols], t.Data[r*t.Cols:r*t.Cols+t.Cols])
		}
		colOffset += t.Cols
	}
	out.children = tensors
	out.backwardFn = func() {
		colOffset := 0
		for _, t := range tensors {
			if t.RequiresGrad {
				for r := 0; r < rows; r++ {
					for c := 0; c < t.Cols; c++ {
						t.Grad[r*t.Cols+c] += out.Grad[r*totalCols+colOffset+c]
					}
				}
			}
			colOffset += t.Cols
		}
	}
	return out
}

// LayerNormForward normalizes x along the last dimension with learnable gamma, beta.
// Returns normed output; also returns mean and rstd slices for use in backward.
func LayerNormForward(x, gamma, beta *Tensor, eps float64) *Tensor {
	rows, cols := x.Rows, x.Cols
	out := NewTensor(rows, cols, x.RequiresGrad || gamma.RequiresGrad || beta.RequiresGrad)

	means := make([]float64, rows)
	rstds := make([]float64, rows)
	xHat := make([]float64, rows*cols)

	for r := 0; r < rows; r++ {
		row := x.Data[r*cols : r*cols+cols]
		mean := 0.0
		for _, v := range row {
			mean += v
		}
		mean /= float64(cols)
		means[r] = mean

		variance := 0.0
		for _, v := range row {
			d := v - mean
			variance += d * d
		}
		variance /= float64(cols)
		rstd := 1.0 / math.Sqrt(variance+eps)
		rstds[r] = rstd

		for c, v := range row {
			xh := (v - mean) * rstd
			xHat[r*cols+c] = xh
			out.Data[r*cols+c] = xh*gamma.Data[c] + beta.Data[c]
		}
	}

	out.children = []*Tensor{x, gamma, beta}
	out.backwardFn = func() {
		// Standard LayerNorm backward
		for r := 0; r < rows; r++ {
			rstd := rstds[r]
			dout := out.Grad[r*cols : r*cols+cols]
			xh := xHat[r*cols : r*cols+cols]

			// dgamma, dbeta accumulate across rows
			if gamma.RequiresGrad {
				for c := 0; c < cols; c++ {
					gamma.Grad[c] += dout[c] * xh[c]
				}
			}
			if beta.RequiresGrad {
				for c := 0; c < cols; c++ {
					beta.Grad[c] += dout[c]
				}
			}

			if x.RequiresGrad {
				// dX = (1/N)*(γ/σ)*(N*dŷ - Σdŷ - x̂*Σ(dŷ*x̂))
				sumDout := 0.0
				sumDoutXhat := 0.0
				for c := 0; c < cols; c++ {
					sumDout += dout[c] * gamma.Data[c]
					sumDoutXhat += dout[c] * gamma.Data[c] * xh[c]
				}
				for c := 0; c < cols; c++ {
					dx := rstd / float64(cols) * (float64(cols)*dout[c]*gamma.Data[c] - sumDout - xh[c]*sumDoutXhat)
					x.Grad[r*cols+c] += dx
				}
			}
		}
	}
	return out
}

// CrossEntropyLoss computes mean cross-entropy loss over all (non-pad) positions.
// logits is [seqLen × vocabSize], targets is [seqLen], padID positions are ignored.
func CrossEntropyLoss(logits *Tensor, targets []int, padID int) *Tensor {
	seqLen, vocabSize := logits.Rows, logits.Cols
	loss := NewTensor(1, 1, logits.RequiresGrad)

	// Compute softmax probabilities (stored for backward)
	probs := make([]float64, seqLen*vocabSize)
	copy(probs, logits.Data)
	rawSoftmaxRows(probs, seqLen, vocabSize)

	totalLoss := 0.0
	count := 0
	for t := 0; t < seqLen; t++ {
		if targets[t] == padID {
			continue
		}
		p := math.Max(probs[t*vocabSize+targets[t]], 1e-30)
		totalLoss -= math.Log(p)
		count++
	}
	if count > 0 {
		loss.Data[0] = totalLoss / float64(count)
	}

	loss.children = []*Tensor{logits}
	loss.backwardFn = func() {
		if !logits.RequiresGrad {
			return
		}
		scale := loss.Grad[0] / float64(max(count, 1))
		for t := 0; t < seqLen; t++ {
			if targets[t] == padID {
				continue
			}
			for j := 0; j < vocabSize; j++ {
				grad := probs[t*vocabSize+j]
				if j == targets[t] {
					grad -= 1.0
				}
				logits.Grad[t*vocabSize+j] += grad * scale
			}
		}
	}
	return loss
}

// SumAll reduces all elements to a [1×1] scalar Tensor.
func SumAll(a *Tensor) *Tensor {
	out := NewTensor(1, 1, a.RequiresGrad)
	for _, v := range a.Data {
		out.Data[0] += v
	}
	out.children = []*Tensor{a}
	out.backwardFn = func() {
		if a.RequiresGrad {
			for i := range a.Grad {
				a.Grad[i] += out.Grad[0]
			}
		}
	}
	return out
}

// ────────────────────────────────────────────────────────────────────────────
// Backward pass — topological sort + reverse traversal
// ────────────────────────────────────────────────────────────────────────────

// Backward seeds this tensor's gradient with 1.0 and propagates backward
// through the computation graph via topological order.
func (t *Tensor) Backward() {
	topo := make([]*Tensor, 0, 64)
	visited := make(map[*Tensor]bool)
	buildTopo(t, visited, &topo)

	// Seed gradient
	if t.Grad == nil {
		t.Grad = make([]float64, len(t.Data))
	}
	t.Grad[0] = 1.0

	for i := len(topo) - 1; i >= 0; i-- {
		node := topo[i]
		if node.backwardFn != nil {
			node.backwardFn()
		}
	}
}

func buildTopo(t *Tensor, visited map[*Tensor]bool, topo *[]*Tensor) {
	if visited[t] {
		return
	}
	visited[t] = true
	for _, child := range t.children {
		buildTopo(child, visited, topo)
	}
	*topo = append(*topo, t)
}

