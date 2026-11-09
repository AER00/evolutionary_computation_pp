package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
	"text/tabwriter"
	"time"

	"OE/solver"
	"OE/task1"
	"OE/task3"
	"OE/task4"
	"OE/task5"
	"OE/task6"
)

// group aggregates the runs of one strategy on one instance.
type group struct {
	strategy string
	instance string
	runs     int
	min      int
	max      int
	avg      float64
	selected int
	elapsed  time.Duration
	best     solver.Result
}

func main() {
	instances := flag.String("instances", "TSPA.csv,TSPB.csv",
		"comma-separated paths to instance CSV files")
	repeat := flag.Int("repeat", 20,
		"runs per strategy and instance")
	jsonOut := flag.String("json", "",
		"also write every run's result to this JSON file (for plotting)")
	candidateSizes := flag.String("candidates", "10,13",
		"comma-separated candidate-list sizes for the candidate steepest search")
	mslsStarts := flag.Int("msls-starts", 400, "local search runs inside one MSLS run")

	flag.Parse()

	if *repeat < 1 {
		fmt.Fprintln(os.Stderr, "error: -repeat must be at least 1")
		os.Exit(2)
	}

	// RandomSolution is the baseline. The full-neighbourhood search is the
	// local-search reference, and the candidate lists are configured with
	// -candidates; the default is ten (the list size used by the task) and
	// thirteen (a wider list for the quality/time comparison).
	strategies := []solver.Strategy{
		&task1.RandomSolution{},
		&task3.RandomStartSteepestEdges{},
	}
	for _, raw := range strings.Split(*candidateSizes, ",") {
		raw = strings.TrimSpace(raw)
		if raw == "" {
			continue
		}
		k, err := strconv.Atoi(raw)
		if err != nil || k < 1 {
			fmt.Fprintf(os.Stderr, "error: invalid -candidates value %q\n", raw)
			os.Exit(2)
		}
		strategies = append(strategies, task4.NewCandidateSteepest(k))
	}
	strategies = append(strategies, &task5.SteepestLM{})

	var paths []string
	for _, name := range strings.Split(*instances, ",") {
		if name = strings.TrimSpace(name); name != "" {
			paths = append(paths, name)
		}
	}

	results, err := solver.Benchmark(strategies, paths, *repeat)
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}

	task6Results, ilsRuns, err := runMSLSAndILS(paths, *repeat, *mslsStarts)
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
	results = append(results, task6Results...)

	if *jsonOut != "" {
		if err := writeJSON(*jsonOut, results); err != nil {
			fmt.Fprintln(os.Stderr, "error:", err)
			os.Exit(1)
		}
	}

	groups := groupResults(results)
	for _, g := range groups {
		printSolution(g.best)
	}
	printTable(groups)
	printILSRuns(ilsRuns)
}

// groupResults buckets results by strategy and instance, keeping the order in
// which each pair first appeared.
func groupResults(results []solver.Result) []*group {
	type key struct{ strategy, instance string }

	index := make(map[key]*group)
	var groups []*group

	for _, r := range results {
		k := key{r.Strategy, r.Instance}
		g, ok := index[k]
		if !ok {
			g = &group{
				strategy: r.Strategy,
				instance: r.Instance,
				min:      r.Objective,
				max:      r.Objective,
				selected: len(r.Tour),
				best:     r,
			}
			index[k] = g
			groups = append(groups, g)
		}

		g.runs++
		if r.Objective < g.min {
			g.min = r.Objective
		}
		if r.Objective > g.max {
			g.max = r.Objective
		}
		if r.Objective < g.best.Objective {
			g.best = r
		}
		g.avg += float64(r.Objective)
		g.elapsed += r.Elapsed
	}

	for _, g := range groups {
		g.avg /= float64(g.runs)
	}
	return groups
}

// printSolution reports the best run of one strategy on one instance.
func printSolution(res solver.Result) {
	fmt.Printf("\nStrategy: %s\n", res.Strategy)
	fmt.Printf("Instance: %s\n", res.Instance)
	fmt.Printf("Objective value: %d\n", res.Objective)
	fmt.Printf("Selected nodes: %d\n", len(res.Tour))
	fmt.Println("Tour:")
	fmt.Println(res.Tour)
}

// runJSON is one run as written to the -json file. Mirrors solver.Result but flattens duration to ms
type runJSON struct {
	Strategy  string  `json:"strategy"`
	Instance  string  `json:"instance"`
	Iteration int     `json:"iteration"`
	Objective int     `json:"objective"`
	TourLen   int     `json:"tour_len"`
	NodeCost  int     `json:"node_cost"`
	Selected  int     `json:"selected"`
	ElapsedMs float64 `json:"elapsed_ms"`
	Tour      []int   `json:"tour"`
}

func writeJSON(path string, results []solver.Result) error {
	runs := make([]runJSON, len(results))
	for i, r := range results {
		runs[i] = runJSON{
			Strategy:  r.Strategy,
			Instance:  r.Instance,
			Iteration: r.Iteration,
			Objective: r.Objective,
			TourLen:   r.TourLen,
			NodeCost:  r.NodeCost,
			Selected:  len(r.Tour),
			ElapsedMs: float64(r.Elapsed.Microseconds()) / 1000,
			Tour:      r.Tour,
		}
	}

	data, err := json.Marshal(runs)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

// printTable reports the objective across runs for every strategy/instance pair.
func printTable(groups []*group) {
	if len(groups) == 0 {
		return
	}

	fmt.Println()
	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(w, "Strategy\tInstance\tRuns\tMin\tMax\tAvg\tSelected\tTime")
	for _, g := range groups {
		fmt.Fprintf(w, "%s\t%s\t%d\t%d\t%d\t%.1f\t%d\t%s\n",
			g.strategy, g.instance, g.runs, g.min, g.max, g.avg, g.selected,
			g.elapsed.Round(time.Millisecond))
	}
	w.Flush()
}

// ilsCounts holds the number of local search runs of every ILS run on one instance.
type ilsCounts struct {
	instance string
	counts   []int
}

// runMSLSAndILS runs MSLS and then ILS on every instance. The ILS time limit
// on an instance is the average running time of MSLS on that instance.
func runMSLSAndILS(paths []string, repeat, starts int) ([]solver.Result, []ilsCounts, error) {
	var results []solver.Result
	var counts []ilsCounts

	for _, path := range paths {
		mslsResults, err := solver.Benchmark(
			[]solver.Strategy{task6.NewMSLS(starts)}, []string{path}, repeat)
		if err != nil {
			return nil, nil, err
		}

		ils := task6.NewILS(averageTime(mslsResults))
		ilsResults, err := solver.Benchmark([]solver.Strategy{ils}, []string{path}, repeat)
		if err != nil {
			return nil, nil, err
		}

		results = append(results, mslsResults...)
		results = append(results, ilsResults...)
		counts = append(counts, ilsCounts{path, ils.LSRuns()})
	}
	return results, counts, nil
}

// averageTime is the mean elapsed time of the given results.
func averageTime(results []solver.Result) time.Duration {
	var total time.Duration
	for _, r := range results {
		total += r.Elapsed
	}
	return total / time.Duration(len(results))
}

// printILSRuns reports how many local searches ILS performed per run.
func printILSRuns(all []ilsCounts) {
	fmt.Println("\nILS: number of local search runs")
	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(w, "Instance\tMin\tMax\tAvg")
	for _, e := range all {
		if len(e.counts) == 0 {
			continue
		}
		lo, hi, sum := e.counts[0], e.counts[0], 0
		for _, c := range e.counts {
			lo, hi, sum = min(lo, c), max(hi, c), sum+c
		}
		fmt.Fprintf(w, "%s\t%d\t%d\t%.1f\n", e.instance, lo, hi, float64(sum)/float64(len(e.counts)))
	}
	w.Flush()
}
