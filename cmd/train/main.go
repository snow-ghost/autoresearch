package main

import (
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/snow-ghost/autoresearch"
)

func main() {
	cacheDir := flag.String("cache-dir", "", "override cache directory")
	evalBytes := flag.Int("eval-bytes", 0, "override number of validation bytes")
	timeBudgetRaw := flag.String("time-budget", "", "override time budget, e.g. 30s or 5m")
	flag.Parse()

	var timeBudget time.Duration
	if *timeBudgetRaw != "" {
		parsed, err := time.ParseDuration(*timeBudgetRaw)
		if err != nil {
			fmt.Fprintf(os.Stderr, "invalid -time-budget: %v\n", err)
			os.Exit(1)
		}
		timeBudget = parsed
	}

	summary, err := autoresearch.RunExperiment(autoresearch.TrainOptions{
		CacheDir:   *cacheDir,
		TimeBudget: timeBudget,
		EvalBytes:  *evalBytes,
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "train failed: %v\n", err)
		os.Exit(1)
	}

	fmt.Print(summary.Format())
}
