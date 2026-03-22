package autoresearch

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const (
	DefaultDatasetURL  = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
	DefaultCacheFolder = "autoresearch-go"
	DefaultValFraction = 0.10
	DefaultMinValBytes = 8 * 1024
	DefaultEvalBytes   = 16 * 1024

	trainFileName = "train.bin"
	valFileName   = "val.bin"
	metaFileName  = "meta.json"
)

type PreparationConfig struct {
	CacheDir    string
	DatasetURL  string
	Force       bool
	ValFraction float64
	MinValBytes int
}

type PreparationSummary struct {
	CacheDir    string
	Source      string
	CorpusBytes int
	TrainBytes  int
	ValBytes    int
}

type Dataset struct {
	CacheDir string
	Source   string
	Train    []byte
	Val      []byte
}

type Tokenizer struct{}

type Predictor interface {
	Predict(context []int, logits []float64)
}

type datasetMeta struct {
	Source      string    `json:"source"`
	CorpusBytes int       `json:"corpus_bytes"`
	TrainBytes  int       `json:"train_bytes"`
	ValBytes    int       `json:"val_bytes"`
	PreparedAt  time.Time `json:"prepared_at"`
}

func NewTokenizer() Tokenizer {
	return Tokenizer{}
}

func (Tokenizer) VocabSize() int {
	return 257
}

func (Tokenizer) BOSID() int {
	return 256
}

func (Tokenizer) Encode(text string) []int {
	out := make([]int, 0, len(text))
	for _, b := range []byte(text) {
		out = append(out, int(b))
	}
	return out
}

func (Tokenizer) Decode(ids []int) string {
	buf := make([]byte, 0, len(ids))
	for _, id := range ids {
		if id < 0 || id > 255 {
			continue
		}
		buf = append(buf, byte(id))
	}
	return string(buf)
}

func ResolveCacheDir(override string) (string, error) {
	if override != "" {
		return override, nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("resolve home dir: %w", err)
	}
	return filepath.Join(home, ".cache", DefaultCacheFolder), nil
}

func Prepare(ctx context.Context, cfg PreparationConfig) (PreparationSummary, error) {
	cacheDir, err := ResolveCacheDir(cfg.CacheDir)
	if err != nil {
		return PreparationSummary{}, err
	}

	if !cfg.Force {
		existing, err := LoadDataset(cacheDir)
		if err == nil {
			return PreparationSummary{
				CacheDir:    cacheDir,
				Source:      existing.Source,
				CorpusBytes: len(existing.Train) + len(existing.Val),
				TrainBytes:  len(existing.Train),
				ValBytes:    len(existing.Val),
			}, nil
		}
	}

	url := cfg.DatasetURL
	if url == "" {
		url = DefaultDatasetURL
	}
	valFraction := cfg.ValFraction
	if valFraction <= 0 || valFraction >= 1 {
		valFraction = DefaultValFraction
	}
	minValBytes := cfg.MinValBytes
	if minValBytes <= 0 {
		minValBytes = DefaultMinValBytes
	}

	corpus, err := downloadCorpus(ctx, url)
	if err != nil {
		return PreparationSummary{}, err
	}

	return writePreparedDataset(cacheDir, corpus, url, valFraction, minValBytes)
}

func LoadDataset(cacheDir string) (Dataset, error) {
	cacheDir, err := ResolveCacheDir(cacheDir)
	if err != nil {
		return Dataset{}, err
	}

	trainPath := filepath.Join(cacheDir, trainFileName)
	valPath := filepath.Join(cacheDir, valFileName)

	trainData, err := os.ReadFile(trainPath)
	if err != nil {
		return Dataset{}, fmt.Errorf("read %s: %w", trainPath, err)
	}
	valData, err := os.ReadFile(valPath)
	if err != nil {
		return Dataset{}, fmt.Errorf("read %s: %w", valPath, err)
	}

	source := "unknown"
	metaPath := filepath.Join(cacheDir, metaFileName)
	if rawMeta, err := os.ReadFile(metaPath); err == nil {
		var meta datasetMeta
		if jsonErr := json.Unmarshal(rawMeta, &meta); jsonErr == nil && meta.Source != "" {
			source = meta.Source
		}
	}

	if len(trainData) == 0 || len(valData) == 0 {
		return Dataset{}, errors.New("prepared dataset is empty")
	}

	return Dataset{
		CacheDir: cacheDir,
		Source:   source,
		Train:    trainData,
		Val:      valData,
	}, nil
}

func SampleBatch(contexts []int, targets []int, data []byte, batchSize, contextLen, bosID int, rng *rand.Rand) ([]int, []int) {
	requiredContexts := batchSize * contextLen
	if cap(contexts) < requiredContexts {
		contexts = make([]int, requiredContexts)
	}
	if cap(targets) < batchSize {
		targets = make([]int, batchSize)
	}
	contexts = contexts[:requiredContexts]
	targets = targets[:batchSize]

	for batchIndex := 0; batchIndex < batchSize; batchIndex++ {
		targetPos := rng.Intn(len(data))
		context := contexts[batchIndex*contextLen : (batchIndex+1)*contextLen]
		FillContext(context, data, targetPos, contextLen, bosID)
		targets[batchIndex] = int(data[targetPos])
	}

	return contexts, targets
}

func FillContext(dst []int, data []byte, targetPos, contextLen, bosID int) {
	start := targetPos - contextLen
	for i := 0; i < contextLen; i++ {
		src := start + i
		if src < 0 {
			dst[i] = bosID
			continue
		}
		dst[i] = int(data[src])
	}
}

func EvaluateBPB(model Predictor, data []byte, contextLen, bosID, maxTokens int) (float64, error) {
	if len(data) == 0 {
		return 0, errors.New("evaluation data is empty")
	}
	if contextLen <= 0 {
		return 0, errors.New("context length must be positive")
	}
	if maxTokens <= 0 || maxTokens > len(data) {
		maxTokens = len(data)
	}

	context := make([]int, contextLen)
	logits := make([]float64, 256)
	totalBits := 0.0

	for pos := 0; pos < maxTokens; pos++ {
		FillContext(context, data, pos, contextLen, bosID)
		model.Predict(context, logits)
		totalBits += negLog2Softmax(logits, int(data[pos]))
	}

	return totalBits / float64(maxTokens), nil
}

func negLog2Softmax(logits []float64, target int) float64 {
	maxLogit := logits[0]
	for _, value := range logits[1:] {
		if value > maxLogit {
			maxLogit = value
		}
	}

	sumExp := 0.0
	for _, value := range logits {
		sumExp += math.Exp(value - maxLogit)
	}
	logProb := logits[target] - maxLogit - math.Log(sumExp)
	return -logProb / math.Ln2
}

func downloadCorpus(ctx context.Context, url string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("User-Agent", "autoresearch-go/0.1")

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("download corpus: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("download corpus: unexpected status %s", resp.Status)
	}

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read corpus: %w", err)
	}
	corpus := normalizeCorpus(raw)
	if len(corpus) < 1024 {
		return nil, fmt.Errorf("download corpus: too small (%d bytes)", len(corpus))
	}
	return corpus, nil
}

func normalizeCorpus(raw []byte) []byte {
	text := strings.ReplaceAll(string(raw), "\r\n", "\n")
	text = strings.TrimSpace(text)
	text += "\n"
	return []byte(text)
}

func writePreparedDataset(cacheDir string, corpus []byte, source string, valFraction float64, minValBytes int) (PreparationSummary, error) {
	trainData, valData, err := splitCorpus(corpus, valFraction, minValBytes)
	if err != nil {
		return PreparationSummary{}, err
	}

	if err := os.MkdirAll(cacheDir, 0o755); err != nil {
		return PreparationSummary{}, fmt.Errorf("create cache dir: %w", err)
	}

	trainPath := filepath.Join(cacheDir, trainFileName)
	valPath := filepath.Join(cacheDir, valFileName)
	metaPath := filepath.Join(cacheDir, metaFileName)

	if err := os.WriteFile(trainPath, trainData, 0o644); err != nil {
		return PreparationSummary{}, fmt.Errorf("write %s: %w", trainPath, err)
	}
	if err := os.WriteFile(valPath, valData, 0o644); err != nil {
		return PreparationSummary{}, fmt.Errorf("write %s: %w", valPath, err)
	}

	meta := datasetMeta{
		Source:      source,
		CorpusBytes: len(corpus),
		TrainBytes:  len(trainData),
		ValBytes:    len(valData),
		PreparedAt:  time.Now().UTC(),
	}
	metaBytes, err := json.MarshalIndent(meta, "", "  ")
	if err != nil {
		return PreparationSummary{}, fmt.Errorf("marshal meta: %w", err)
	}
	if err := os.WriteFile(metaPath, metaBytes, 0o644); err != nil {
		return PreparationSummary{}, fmt.Errorf("write %s: %w", metaPath, err)
	}

	return PreparationSummary{
		CacheDir:    cacheDir,
		Source:      source,
		CorpusBytes: len(corpus),
		TrainBytes:  len(trainData),
		ValBytes:    len(valData),
	}, nil
}

func splitCorpus(corpus []byte, valFraction float64, minValBytes int) ([]byte, []byte, error) {
	if len(corpus) < 2*minValBytes {
		return nil, nil, fmt.Errorf("corpus is too small: %d bytes", len(corpus))
	}
	if valFraction <= 0 || valFraction >= 1 {
		valFraction = DefaultValFraction
	}
	if minValBytes <= 0 {
		minValBytes = DefaultMinValBytes
	}

	split := int(float64(len(corpus)) * (1 - valFraction))
	if split > len(corpus)-minValBytes {
		split = len(corpus) - minValBytes
	}
	if split < minValBytes {
		split = minValBytes
	}

	for offset := 0; offset < 1024 && split+offset < len(corpus)-minValBytes; offset++ {
		if corpus[split+offset] == '\n' {
			split += offset + 1
			break
		}
	}

	trainData := append([]byte(nil), corpus[:split]...)
	valData := append([]byte(nil), corpus[split:]...)
	if len(trainData) == 0 || len(valData) == 0 {
		return nil, nil, errors.New("train or validation split ended up empty")
	}
	return trainData, valData, nil
}
