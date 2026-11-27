package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"OE/solver"
	"OE/task6"
	"OE/task8"
)

func main() {
	instances := flag.String("instances", "TSPA.csv,TSPB.csv",
		"comma-separated paths to instance CSV files")
	samples := flag.Int("samples", 1000, "random-start local optima per instance")
	referenceRuns := flag.Int("reference-runs", 20, "number of ILS runs for the reference solution")
	mslsStarts := flag.Int("msls-starts", 400, "local searches per MSLS run used to set the ILS time budget")
	outputPath := flag.String("json", "report/task8_data.json", "Task 8 analysis data JSON output path")
	flag.Parse()

	if *samples < 2 || *referenceRuns < 1 || *mslsStarts < 1 {
		fatalUsage("samples must be at least 2; reference-runs and msls-starts must be positive")
	}
	paths := splitPaths(*instances)
	if len(paths) == 0 {
		fatalUsage("at least one instance path is required")
	}

	datasets := make([]task8.Dataset, 0, len(paths))
	for _, path := range paths {
		dataset, err := analyzeInstance(path, *samples, *referenceRuns, *mslsStarts)
		if err != nil {
			fmt.Fprintln(os.Stderr, "error:", err)
			os.Exit(1)
		}
		datasets = append(datasets, dataset)
	}

	if err := writeJSON(*outputPath, datasets); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
	fmt.Println("Task 8 data written to", *outputPath)
}

func fatalUsage(msg string) {
	fmt.Fprintln(os.Stderr, "error:", msg)
	os.Exit(2)
}

func splitPaths(list string) []string {
	var paths []string
	for _, path := range strings.Split(list, ",") {
		if path = strings.TrimSpace(path); path != "" {
			paths = append(paths, path)
		}
	}
	return paths
}

// analyzeInstance builds the reference solution (best of several ILS runs with
// the MSLS time budget) and runs the local-optima similarity analysis.
func analyzeInstance(path string, samples, referenceRuns, mslsStarts int) (task8.Dataset, error) {
	p, err := solver.LoadProblem(path)
	if err != nil {
		return task8.Dataset{}, err
	}

	reference, err := bestILS(path, referenceRuns, mslsStarts)
	if err != nil {
		return task8.Dataset{}, err
	}

	dataset, err := task8.Analyze(p, samples, reference.Tour)
	if err != nil {
		return task8.Dataset{}, err
	}

	bestSampled := dataset.LocalOptima[dataset.BestLocalOptimum].Objective
	fmt.Printf("%s: %d local optima, best sampled objective %d, reference (ILS) objective %d\n",
		path, samples, bestSampled, dataset.ReferenceObjective)
	if dataset.ReferenceObjective > bestSampled {
		fmt.Fprintf(os.Stderr,
			"warning: %s reference is worse than the best sampled optimum; "+
				"it is not a 'very good' solution\n", path)
	}
	return dataset, nil
}

// bestILS returns the best result of `runs` ILS runs whose time limit equals
// the mean running time of MSLS.
func bestILS(path string, runs, mslsStarts int) (solver.Result, error) {
	mslsRuns, err := solver.Benchmark(
		[]solver.Strategy{task6.NewMSLS(mslsStarts)}, []string{path}, runs)
	if err != nil {
		return solver.Result{}, err
	}
	ils, err := solver.Benchmark(
		[]solver.Strategy{task6.NewILS(meanRunTime(mslsRuns))}, []string{path}, runs)
	if err != nil {
		return solver.Result{}, err
	}
	best := ils[0]
	for _, result := range ils[1:] {
		if result.Objective < best.Objective {
			best = result
		}
	}
	return best, nil
}

func writeJSON(path string, v any) error {
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	if dir := filepath.Dir(path); dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	return os.WriteFile(path, data, 0o644)
}

func meanRunTime(results []solver.Result) time.Duration {
	var total time.Duration
	for _, result := range results {
		total += result.Elapsed
	}
	return total / time.Duration(len(results))
}
