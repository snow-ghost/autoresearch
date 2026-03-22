package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	"github.com/snow-ghost/autoresearch"
)

func main() {
	cacheDir := flag.String("cache-dir", "", "override cache directory")
	datasetURL := flag.String("dataset-url", "", "override dataset URL")
	force := flag.Bool("force", false, "redownload and rewrite prepared files")
	flag.Parse()

	summary, err := autoresearch.Prepare(context.Background(), autoresearch.PreparationConfig{
		CacheDir:   *cacheDir,
		DatasetURL: *datasetURL,
		Force:      *force,
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "prepare failed: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("cache_dir:    %s\n", summary.CacheDir)
	fmt.Printf("source:       %s\n", summary.Source)
	fmt.Printf("corpus_bytes: %d\n", summary.CorpusBytes)
	fmt.Printf("train_bytes:  %d\n", summary.TrainBytes)
	fmt.Printf("val_bytes:    %d\n", summary.ValBytes)
}
