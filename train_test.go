package autoresearch

import (
	"strings"
	"testing"
	"time"
)

func TestRunExperimentSmoke(t *testing.T) {
	cacheDir := t.TempDir()
	corpus := []byte(strings.Repeat("All the world's a stage,\nAnd all the men and women merely players.\n", 512))
	if _, err := writePreparedDataset(cacheDir, corpus, "test://corpus", 0.1, 256); err != nil {
		t.Fatalf("writePreparedDataset() error = %v", err)
	}

	summary, err := RunExperiment(TrainOptions{
		CacheDir:   cacheDir,
		TimeBudget: 25 * time.Millisecond,
		EvalBytes:  512,
	})
	if err != nil {
		t.Fatalf("RunExperiment() error = %v", err)
	}
	if summary.NumSteps == 0 {
		t.Fatalf("expected at least one training step")
	}
	if summary.ValBPB <= 0 {
		t.Fatalf("expected positive val_bpb, got %f", summary.ValBPB)
	}
}
