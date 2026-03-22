package autoresearch

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestPrepareAndLoadDataset(t *testing.T) {
	corpus := strings.Repeat("To be, or not to be, that is the question.\n", 256)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(corpus))
	}))
	defer server.Close()

	cacheDir := t.TempDir()
	summary, err := Prepare(context.Background(), PreparationConfig{
		CacheDir:    cacheDir,
		DatasetURL:  server.URL,
		Force:       true,
		MinValBytes: 256,
	})
	if err != nil {
		t.Fatalf("Prepare() error = %v", err)
	}
	if summary.TrainBytes == 0 || summary.ValBytes == 0 {
		t.Fatalf("unexpected empty split: %+v", summary)
	}

	dataset, err := LoadDataset(cacheDir)
	if err != nil {
		t.Fatalf("LoadDataset() error = %v", err)
	}
	if len(dataset.Train) != summary.TrainBytes || len(dataset.Val) != summary.ValBytes {
		t.Fatalf("loaded sizes do not match summary: %+v, got train=%d val=%d", summary, len(dataset.Train), len(dataset.Val))
	}
}
