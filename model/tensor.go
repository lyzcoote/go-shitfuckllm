// Package model implements a transformer decoder-only LLM with manual autograd.
package model

import (
	"math"
	"runtime"
	"sync"
)

// matMulWorkers is the number of goroutines rawMatMul may use for large products.
// Defaults to the number of CPU cores; override with SetMatMulWorkers (e.g. =1 to disable).
var matMulWorkers = runtime.NumCPU()

// SetMatMulWorkers sets how many cores rawMatMul parallelises across. Values < 1 are
// clamped to 1 (single-threaded). Call once at startup before training/generation.
func SetMatMulWorkers(n int) {
	if n < 1 {
		n = 1
	}
	matMulWorkers = n
}

// MatMulWorkers returns the current matrix-multiply parallelism degree.
func MatMulWorkers() int { return matMulWorkers }

// Tensor is a 2-D float64 array (flat, row-major) with an attached gradient buffer
// and a backward function for automatic differentiation.
type Tensor struct {
	Data         []float64
	Grad         []float64
	Rows         int
	Cols         int
	RequiresGrad bool
	backwardFn   func()
	children     []*Tensor
}

// NewTensor allocates a zero-filled Tensor of shape [rows, cols].
func NewTensor(rows, cols int, requiresGrad bool) *Tensor {
	n := rows * cols
	t := &Tensor{
		Data:         make([]float64, n),
		Rows:         rows,
		Cols:         cols,
		RequiresGrad: requiresGrad,
	}
	if requiresGrad {
		t.Grad = make([]float64, n)
	}
	return t
}

// NewTensorFrom wraps existing data (not copied) into a Tensor.
func NewTensorFrom(data []float64, rows, cols int, requiresGrad bool) *Tensor {
	t := &Tensor{
		Data:         data,
		Rows:         rows,
		Cols:         cols,
		RequiresGrad: requiresGrad,
	}
	if requiresGrad {
		t.Grad = make([]float64, len(data))
	}
	return t
}

// At returns element at row r, column c.
func (t *Tensor) At(r, c int) float64 { return t.Data[r*t.Cols+c] }

// Set assigns value v to element at row r, column c.
func (t *Tensor) Set(r, c int, v float64) { t.Data[r*t.Cols+c] = v }

// AddGrad adds delta to the gradient at position (r, c).
func (t *Tensor) AddGrad(r, c int, delta float64) {
	if t.RequiresGrad {
		t.Grad[r*t.Cols+c] += delta
	}
}

// ZeroGrad zeros all gradient values.
func (t *Tensor) ZeroGrad() {
	for i := range t.Grad {
		t.Grad[i] = 0
	}
}

// Clone returns a deep copy of the tensor (Data only, no grad, no graph).
func (t *Tensor) Clone() *Tensor {
	out := NewTensor(t.Rows, t.Cols, false)
	copy(out.Data, t.Data)
	return out
}

// ────────────────────────────────────────────────────────────────────────────
// Raw (graph-free) matrix helpers — used inside backward closures.
// ────────────────────────────────────────────────────────────────────────────

// matMulParallelThreshold is the minimum number of multiply-add operations (m*k*n)
// above which rawMatMul splits the work across CPU cores. Below it, the goroutine
// scheduling overhead outweighs the gain and we run single-threaded.
const matMulParallelThreshold = 1 << 15 // 32768

// rawMatMul computes C = A * B where A is [m×k], B is [k×n], result is [m×n].
// For large products it parallelises across CPU cores by partitioning the rows of A;
// each worker writes a disjoint row range of C, so there is no data race.
func rawMatMul(A []float64, B []float64, m, k, n int) []float64 {
	C := make([]float64, m*n)

	workers := matMulWorkers
	if workers <= 1 || m < 2 || int64(m)*int64(k)*int64(n) < matMulParallelThreshold {
		matMulRows(C, A, B, 0, m, k, n)
		return C
	}
	if workers > m {
		workers = m
	}

	var wg sync.WaitGroup
	chunk := (m + workers - 1) / workers
	for start := 0; start < m; start += chunk {
		end := start + chunk
		if end > m {
			end = m
		}
		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			matMulRows(C, A, B, s, e, k, n)
		}(start, end)
	}
	wg.Wait()
	return C
}

// matMulRows computes rows [rowStart, rowEnd) of C = A * B.
func matMulRows(C, A, B []float64, rowStart, rowEnd, k, n int) {
	for i := rowStart; i < rowEnd; i++ {
		ci := C[i*n : i*n+n]
		for p := 0; p < k; p++ {
			aip := A[i*k+p]
			if aip == 0 {
				continue
			}
			bp := B[p*n : p*n+n]
			for j := 0; j < n; j++ {
				ci[j] += aip * bp[j]
			}
		}
	}
}

// rawTranspose returns A^T where A is [rows×cols], result is [cols×rows].
func rawTranspose(A []float64, rows, cols int) []float64 {
	T := make([]float64, rows*cols)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			T[c*rows+r] = A[r*cols+c]
		}
	}
	return T
}

// rawAdd adds B into A element-wise (in place). Both must have the same length.
func rawAdd(A, B []float64) {
	for i := range A {
		A[i] += B[i]
	}
}

// rawScale multiplies every element of A by s (in place).
func rawScale(A []float64, s float64) {
	for i := range A {
		A[i] *= s
	}
}

// rawSoftmaxRows applies softmax row-wise to a flat [rows×cols] array (in place).
func rawSoftmaxRows(A []float64, rows, cols int) {
	for r := 0; r < rows; r++ {
		row := A[r*cols : r*cols+cols]
		maxV := row[0]
		for _, v := range row {
			if v > maxV {
				maxV = v
			}
		}
		sum := 0.0
		for i, v := range row {
			row[i] = math.Exp(v - maxV)
			sum += row[i]
		}
		for i := range row {
			row[i] /= sum
		}
	}
}

// geluVal computes GELU(x) = 0.5*x*(1 + tanh(sqrt(2/π)*(x + 0.044715*x^3))).
func geluVal(x float64) float64 {
	c := math.Sqrt(2.0 / math.Pi)
	inner := c * (x + 0.044715*x*x*x)
	return 0.5 * x * (1.0 + math.Tanh(inner))
}

// geluGrad computes the derivative of GELU at x.
func geluGrad(x float64) float64 {
	c := math.Sqrt(2.0 / math.Pi)
	inner := c * (x + 0.044715*x*x*x)
	tanh := math.Tanh(inner)
	dtanh := 1.0 - tanh*tanh
	dInner := c * (1.0 + 3*0.044715*x*x)
	return 0.5*(1.0+tanh) + 0.5*x*dtanh*dInner
}
