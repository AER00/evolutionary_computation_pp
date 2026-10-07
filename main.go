package main

import (
	"flag"
	"fmt"
	"os"
	"strings"
	"text/tabwriter"
	"time"

	"OE/solver"
	"OE/task1"
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
	repeat := flag.Int("repeat", 200,
		"runs per strategy and instance")

	flag.Parse()

	if *repeat < 1 {
		fmt.Fprintln(os.Stderr, "error: -repeat must be at least 1")
		os.Exit(2)
	}

	strategies := []solver.Strategy{
		&task1.RandomSolution{},
		&task1.ExtendEnd{},
		&task1.ExtendAny{},
		&task1.GreedyCycle{},
	}

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

	groups := groupResults(results)
	for _, g := range groups {
		printSolution(g.best)
	}
	printTable(groups)
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
