package autoresearch

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"strconv"
	"time"
)

const (
	DefaultTimeBudget = 5 * time.Minute

	defaultContextLen = 8
	defaultEmbedDim   = 24
	defaultHiddenDim  = 128
	defaultBatchSize  = 64
	defaultLearning   = 3e-3
	defaultBeta1      = 0.9
	defaultBeta2      = 0.99
	defaultDecay      = 1e-5
	defaultSeed       = 1337
)

type TrainOptions struct {
	CacheDir   string
	TimeBudget time.Duration
	EvalBytes  int
}

type ExperimentConfig struct {
	ContextLen   int
	EmbedDim     int
	HiddenDim    int
	BatchSize    int
	LearningRate float64
	Beta1        float64
	Beta2        float64
	WeightDecay  float64
	Seed         int64
}

type RunSummary struct {
	ValBPB          float64
	TrainingSeconds float64
	TotalSeconds    float64
	PeakHeapMB      float64
	TotalTokensK    float64
	NumSteps        int
	NumParamsK      float64
	ContextLen      int
	HiddenDim       int
	EmbedDim        int
}

type parameter struct {
	Data        []float64
	Grad        []float64
	M           []float64
	V           []float64
	WeightDecay float64
}

type mlpModel struct {
	cfg        ExperimentConfig
	tokenizer  Tokenizer
	inputDim   int
	step       int
	embeddings parameter
	w1         parameter
	b1         parameter
	w2         parameter
	b2         parameter
}

func DefaultExperimentConfig() ExperimentConfig {
	return ExperimentConfig{
		ContextLen:   defaultContextLen,
		EmbedDim:     defaultEmbedDim,
		HiddenDim:    defaultHiddenDim,
		BatchSize:    defaultBatchSize,
		LearningRate: defaultLearning,
		Beta1:        defaultBeta1,
		Beta2:        defaultBeta2,
		WeightDecay:  defaultDecay,
		Seed:         defaultSeed,
	}
}

func RunExperiment(opts TrainOptions) (RunSummary, error) {
	totalStart := time.Now()

	cacheDir, err := ResolveCacheDir(opts.CacheDir)
	if err != nil {
		return RunSummary{}, err
	}
	timeBudget, err := resolveTimeBudget(opts.TimeBudget)
	if err != nil {
		return RunSummary{}, err
	}
	evalBytes, err := resolveEvalBytes(opts.EvalBytes)
	if err != nil {
		return RunSummary{}, err
	}

	cfg := DefaultExperimentConfig()
	tokenizer := NewTokenizer()
	dataset, err := LoadDataset(cacheDir)
	if err != nil {
		return RunSummary{}, err
	}
	if len(dataset.Train) < cfg.ContextLen+1 {
		return RunSummary{}, fmt.Errorf("training corpus is too small for context_len=%d", cfg.ContextLen)
	}
	if len(dataset.Val) == 0 {
		return RunSummary{}, fmt.Errorf("validation corpus is empty")
	}

	rng := rand.New(rand.NewSource(cfg.Seed))
	model := newMLPModel(cfg, tokenizer, rng)
	contexts := make([]int, cfg.BatchSize*cfg.ContextLen)
	targets := make([]int, cfg.BatchSize)

	peakHeap := currentHeapBytes()
	trainStart := time.Now()
	steps := 0
	totalTokens := 0

	for steps == 0 || time.Since(trainStart) < timeBudget {
		contexts, targets = SampleBatch(contexts, targets, dataset.Train, cfg.BatchSize, cfg.ContextLen, tokenizer.BOSID(), rng)
		model.TrainBatch(contexts, targets)
		steps++
		totalTokens += len(targets)
		if current := currentHeapBytes(); current > peakHeap {
			peakHeap = current
		}
	}

	trainingSeconds := time.Since(trainStart).Seconds()
	valBPB, err := EvaluateBPB(model, dataset.Val, cfg.ContextLen, tokenizer.BOSID(), evalBytes)
	if err != nil {
		return RunSummary{}, err
	}
	if current := currentHeapBytes(); current > peakHeap {
		peakHeap = current
	}

	return RunSummary{
		ValBPB:          valBPB,
		TrainingSeconds: trainingSeconds,
		TotalSeconds:    time.Since(totalStart).Seconds(),
		PeakHeapMB:      float64(peakHeap) / (1024 * 1024),
		TotalTokensK:    float64(totalTokens) / 1000.0,
		NumSteps:        steps,
		NumParamsK:      float64(model.NumParams()) / 1000.0,
		ContextLen:      cfg.ContextLen,
		HiddenDim:       cfg.HiddenDim,
		EmbedDim:        cfg.EmbedDim,
	}, nil
}

func (s RunSummary) Format() string {
	return fmt.Sprintf(
		"---\n"+
			"val_bpb:          %.6f\n"+
			"training_seconds: %.1f\n"+
			"total_seconds:    %.1f\n"+
			"peak_heap_mb:     %.1f\n"+
			"total_tokens_k:   %.1f\n"+
			"num_steps:        %d\n"+
			"num_params_k:     %.1f\n"+
			"context_len:      %d\n"+
			"hidden_dim:       %d\n"+
			"embed_dim:        %d\n",
		s.ValBPB,
		s.TrainingSeconds,
		s.TotalSeconds,
		s.PeakHeapMB,
		s.TotalTokensK,
		s.NumSteps,
		s.NumParamsK,
		s.ContextLen,
		s.HiddenDim,
		s.EmbedDim,
	)
}

func newMLPModel(cfg ExperimentConfig, tokenizer Tokenizer, rng *rand.Rand) *mlpModel {
	inputDim := cfg.ContextLen * cfg.EmbedDim
	model := &mlpModel{
		cfg:       cfg,
		tokenizer: tokenizer,
		inputDim:  inputDim,
		embeddings: newParameter(
			tokenizer.VocabSize()*cfg.EmbedDim,
			0,
		),
		w1: newParameter(cfg.HiddenDim*inputDim, cfg.WeightDecay),
		b1: newParameter(cfg.HiddenDim, 0),
		w2: newParameter(256*cfg.HiddenDim, cfg.WeightDecay),
		b2: newParameter(256, 0),
	}

	initUniform(model.embeddings.Data, rng, math.Sqrt(1.0/float64(cfg.EmbedDim)))
	initUniform(model.w1.Data, rng, math.Sqrt(6.0/float64(inputDim+cfg.HiddenDim)))
	initUniform(model.w2.Data, rng, math.Sqrt(6.0/float64(cfg.HiddenDim+256)))

	return model
}

func (m *mlpModel) NumParams() int {
	return len(m.embeddings.Data) + len(m.w1.Data) + len(m.b1.Data) + len(m.w2.Data) + len(m.b2.Data)
}

func (m *mlpModel) Predict(context []int, logits []float64) {
	hidden := make([]float64, m.cfg.HiddenDim)
	x := make([]float64, m.inputDim)
	m.buildInput(context, x)
	m.forwardHidden(x, hidden)
	m.forwardOutput(hidden, logits)
}

func (m *mlpModel) TrainBatch(contexts []int, targets []int) float64 {
	m.zeroGrad()

	x := make([]float64, m.inputDim)
	hidden := make([]float64, m.cfg.HiddenDim)
	logits := make([]float64, 256)
	probs := make([]float64, 256)
	dHidden := make([]float64, m.cfg.HiddenDim)
	dInput := make([]float64, m.inputDim)

	totalLoss := 0.0
	for batchIndex, target := range targets {
		context := contexts[batchIndex*m.cfg.ContextLen : (batchIndex+1)*m.cfg.ContextLen]
		m.buildInput(context, x)
		m.forwardHidden(x, hidden)
		m.forwardOutput(hidden, logits)

		loss := softmaxCrossEntropy(logits, target, probs)
		totalLoss += loss
		probs[target] -= 1

		clear(dHidden)
		for out := 0; out < 256; out++ {
			grad := probs[out]
			m.b2.Grad[out] += grad
			gradRow := m.w2.Grad[out*m.cfg.HiddenDim : (out+1)*m.cfg.HiddenDim]
			dataRow := m.w2.Data[out*m.cfg.HiddenDim : (out+1)*m.cfg.HiddenDim]
			for h := 0; h < m.cfg.HiddenDim; h++ {
				gradRow[h] += grad * hidden[h]
				dHidden[h] += dataRow[h] * grad
			}
		}

		clear(dInput)
		for h := 0; h < m.cfg.HiddenDim; h++ {
			gradPreAct := dHidden[h] * (1 - hidden[h]*hidden[h])
			m.b1.Grad[h] += gradPreAct
			gradRow := m.w1.Grad[h*m.inputDim : (h+1)*m.inputDim]
			dataRow := m.w1.Data[h*m.inputDim : (h+1)*m.inputDim]
			for i := 0; i < m.inputDim; i++ {
				gradRow[i] += gradPreAct * x[i]
				dInput[i] += dataRow[i] * gradPreAct
			}
		}

		for pos, tokenID := range context {
			gradRow := m.embeddings.Grad[tokenID*m.cfg.EmbedDim : (tokenID+1)*m.cfg.EmbedDim]
			inputGrad := dInput[pos*m.cfg.EmbedDim : (pos+1)*m.cfg.EmbedDim]
			for dim := 0; dim < m.cfg.EmbedDim; dim++ {
				gradRow[dim] += inputGrad[dim]
			}
		}
	}

	scale := 1.0 / float64(len(targets))
	m.step++
	for _, param := range []*parameter{&m.embeddings, &m.w1, &m.b1, &m.w2, &m.b2} {
		param.scaleGrad(scale)
		param.adamStep(m.cfg.LearningRate, m.cfg.Beta1, m.cfg.Beta2, m.step)
	}

	return totalLoss * scale
}

func (m *mlpModel) buildInput(context []int, dst []float64) {
	for pos, tokenID := range context {
		src := m.embeddings.Data[tokenID*m.cfg.EmbedDim : (tokenID+1)*m.cfg.EmbedDim]
		copy(dst[pos*m.cfg.EmbedDim:(pos+1)*m.cfg.EmbedDim], src)
	}
}

func (m *mlpModel) forwardHidden(x []float64, hidden []float64) {
	for h := 0; h < m.cfg.HiddenDim; h++ {
		sum := m.b1.Data[h]
		row := m.w1.Data[h*m.inputDim : (h+1)*m.inputDim]
		for i := 0; i < m.inputDim; i++ {
			sum += row[i] * x[i]
		}
		hidden[h] = math.Tanh(sum)
	}
}

func (m *mlpModel) forwardOutput(hidden []float64, logits []float64) {
	for out := 0; out < 256; out++ {
		sum := m.b2.Data[out]
		row := m.w2.Data[out*m.cfg.HiddenDim : (out+1)*m.cfg.HiddenDim]
		for h := 0; h < m.cfg.HiddenDim; h++ {
			sum += row[h] * hidden[h]
		}
		logits[out] = sum
	}
}

func (m *mlpModel) zeroGrad() {
	for _, param := range []*parameter{&m.embeddings, &m.w1, &m.b1, &m.w2, &m.b2} {
		clear(param.Grad)
	}
}

func softmaxCrossEntropy(logits []float64, target int, probs []float64) float64 {
	maxLogit := logits[0]
	for _, value := range logits[1:] {
		if value > maxLogit {
			maxLogit = value
		}
	}

	sumExp := 0.0
	for i, value := range logits {
		expValue := math.Exp(value - maxLogit)
		probs[i] = expValue
		sumExp += expValue
	}
	inv := 1.0 / sumExp
	for i := range probs {
		probs[i] *= inv
	}
	return -(logits[target] - maxLogit - math.Log(sumExp))
}

func newParameter(size int, decay float64) parameter {
	return parameter{
		Data:        make([]float64, size),
		Grad:        make([]float64, size),
		M:           make([]float64, size),
		V:           make([]float64, size),
		WeightDecay: decay,
	}
}

func (p *parameter) scaleGrad(scale float64) {
	for i := range p.Grad {
		p.Grad[i] *= scale
	}
}

func (p *parameter) adamStep(lr, beta1, beta2 float64, step int) {
	b1Correction := 1 - math.Pow(beta1, float64(step))
	b2Correction := 1 - math.Pow(beta2, float64(step))
	const eps = 1e-8

	for i, grad := range p.Grad {
		if p.WeightDecay != 0 {
			grad += p.WeightDecay * p.Data[i]
		}
		p.M[i] = beta1*p.M[i] + (1-beta1)*grad
		p.V[i] = beta2*p.V[i] + (1-beta2)*grad*grad

		mHat := p.M[i] / b1Correction
		vHat := p.V[i] / b2Correction
		p.Data[i] -= lr * mHat / (math.Sqrt(vHat) + eps)
	}
}

func initUniform(values []float64, rng *rand.Rand, scale float64) {
	for i := range values {
		values[i] = (rng.Float64()*2 - 1) * scale
	}
}

func currentHeapBytes() uint64 {
	var stats runtime.MemStats
	runtime.ReadMemStats(&stats)
	return stats.HeapAlloc
}

func resolveTimeBudget(override time.Duration) (time.Duration, error) {
	if override > 0 {
		return override, nil
	}
	if raw := getenv("AUTORESEARCH_TIME_BUDGET"); raw != "" {
		value, err := time.ParseDuration(raw)
		if err != nil {
			return 0, fmt.Errorf("parse AUTORESEARCH_TIME_BUDGET: %w", err)
		}
		return value, nil
	}
	return DefaultTimeBudget, nil
}

func resolveEvalBytes(override int) (int, error) {
	if override > 0 {
		return override, nil
	}
	if raw := getenv("AUTORESEARCH_EVAL_BYTES"); raw != "" {
		value, err := strconv.Atoi(raw)
		if err != nil {
			return 0, fmt.Errorf("parse AUTORESEARCH_EVAL_BYTES: %w", err)
		}
		return value, nil
	}
	return DefaultEvalBytes, nil
}

func getenv(key string) string {
	return os.Getenv(key)
}
