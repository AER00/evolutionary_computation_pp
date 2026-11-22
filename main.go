package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"
	"text/tabwriter"
	"time"

	"OE/solver"
	"OE/task1"
	"OE/task2"
	"OE/task3"
	"OE/task5"
	"OE/task6"
	"OE/task7"
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
	// candidateSizes := flag.String("candidates", "10,13",
	// 	"comma-separated candidate-list sizes for the candidate steepest search")
	mslsStarts := flag.Int("msls-starts", 400, "local search runs inside one MSLS run")
	regretWeight := flag.Float64("regret-weight", 0.5,
		"regret weight of the LNS repair operator (best value from the greedy assignment)")
	construction := flag.Bool("construction", false,
		"also benchmark the construction heuristics (used as the Task 6/7 reference)")

	flag.Parse()

	if *repeat < 1 {
		fmt.Fprintln(os.Stderr, "error: -repeat must be at least 1")
		os.Exit(2)
	}
	if *regretWeight < 0 || *regretWeight > 1 {
		fmt.Fprintln(os.Stderr, "error: -regret-weight must be in [0, 1]")
		os.Exit(2)
	}

	// RandomSolution is the baseline. The full-neighbourhood search is the
	// local-search reference, and the candidate lists are configured with
	// -candidates; the default is ten (the list size used by the task) and
	// thirteen (a wider list for the quality/time comparison).
	strategies := []solver.Strategy{
		&task1.RandomSolution{},
		&task3.RandomStartSteepestEdges{},
		&task5.SteepestLM{},
	}
	if *construction {
		strategies = append(strategies,
			&task1.ExtendEnd{},
			&task1.ExtendAny{},
			&task1.GreedyCycle{},
			&task2.RegretOnlyExtendAny{},
			&task2.RegretOnlyGreedyCycle{},
			&task2.RegretWeightedExtendAny{},
			&task2.RegretWeightedGreedyCycle{},
		)
	} else {
		strategies = append(strategies, &task2.RegretWeightedGreedyCycle{})
	}

	var paths []string
	for name := range strings.SplitSeq(*instances, ",") {
		if name = strings.TrimSpace(name); name != "" {
			paths = append(paths, name)
		}
	}

	results, err := solver.Benchmark(strategies, paths, *repeat)
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}

	metaResults, iterCounts, err := runMetaheuristics(paths, *repeat, *mslsStarts, *regretWeight)
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
	results = append(results, metaResults...)

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
	printIterations(iterCounts)
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

// runJSON is one run as written to the -json file. Mirrors solver.Result but flattens duration to ms.
// LSRuns holds the number of main-loop iterations for ILS (local search runs) and LNS.
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
	LSRuns    int     `json:"ls_runs,omitempty"`
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
			LSRuns:    r.LSRuns,
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

// iterCounts holds the number of main-loop iterations of every run of one
// time-limited strategy on one instance (local search runs for ILS,
// destroy-repair iterations for LNS).
type iterCounts struct {
	strategy string
	instance string
	counts   []int
}

// timeLimited is a metaheuristic that stops after a time budget and reports
// how many main-loop iterations each run performed.
type timeLimited struct {
	strategy solver.Strategy
	counts   func() []int
}

// runMetaheuristics runs MSLS first and then ILS and the six LNS variants on
// every instance. The time limit of ILS and LNS on an instance is the average
// running time of MSLS on that instance.
func runMetaheuristics(paths []string, repeat, starts int, regretWeight float64) ([]solver.Result, []iterCounts, error) {
	var results []solver.Result
	var counts []iterCounts

	destroyOps := []struct {
		name string
		op   task7.DestroyOp
	}{
		{"RandomNodes", task7.RemoveRandomNodes},
		{"SingleSubpath", task7.RemoveSubpath},
		{"MultiSubpath", task7.RemoveSubpaths},
	}

	for _, path := range paths {
		mslsResults, err := solver.Benchmark(
			[]solver.Strategy{task6.NewMSLS(starts)}, []string{path}, repeat)
		if err != nil {
			return nil, nil, err
		}
		results = append(results, mslsResults...)
		budget := averageTime(mslsResults)

		// Fresh strategy objects per instance so the counters are not mixed.
		ils := task6.NewILS(budget)
		methods := []timeLimited{{ils, ils.LSRuns}}
		for _, d := range destroyOps {
			for _, useLS := range []bool{false, true} {
				l := task7.NewLNS(d.name, d.op, useLS, budget, regretWeight)
				methods = append(methods, timeLimited{l, l.Iterations})
			}
		}

		for _, m := range methods {
			res, err := solver.Benchmark([]solver.Strategy{m.strategy}, []string{path}, repeat)
			if err != nil {
				return nil, nil, err
			}
			// Benchmark runs the iterations in order, so the two slices line up.
			c := m.counts()
			for k := range res {
				if k < len(c) {
					res[k].LSRuns = c[k]
				}
			}
			results = append(results, res...)
			counts = append(counts, iterCounts{m.strategy.Name(), path, c})
		}
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

// printIterations reports the number of main-loop iterations per run
// (ILS: local search runs; LNS: destroy-repair iterations).
func printIterations(all []iterCounts) {
	fmt.Println("\nNumber of main-loop iterations")
	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(w, "Strategy\tInstance\tMin\tMax\tAvg")
	for _, e := range all {
		if len(e.counts) == 0 {
			continue
		}
		lo, hi, sum := e.counts[0], e.counts[0], 0
		for _, c := range e.counts {
			lo, hi, sum = min(lo, c), max(hi, c), sum+c
		}
		fmt.Fprintf(w, "%s\t%s\t%d\t%d\t%.1f\n",
			e.strategy, e.instance, lo, hi, float64(sum)/float64(len(e.counts)))
	}
	w.Flush()
}
