package solver

import (
	"time"
)

// Strategy is one optimization method. Every task supplies its own.
//
// A strategy only ever returns a tour — it never reports a score, and it
// decides for itself when to stop. How long the run took is measured by the
// runner, not asked of the strategy.
type Strategy interface {
	// Name identifies the method, e.g. "sa".
	Name() string
	// Solve searches for a tour of exactly p.M distinct nodes and returns it.
	Solve(p *Problem, iteration int) []int
}

// Result is the outcome of one run.
type Result struct {
	Strategy  string
	Instance  string
	Objective int // TourLen + NodeCost
	TourLen   int
	NodeCost  int
	Tour      []int
	Elapsed   time.Duration
}

// SolveProblem runs strategy on an already-loaded problem and measures how
// long the run took.
func SolveProblem(strategy Strategy, p *Problem, iteration int) Result {
	started := time.Now()
	tour := strategy.Solve(p, iteration)
	elapsed := time.Since(started)

	tourLen := TourLength(p, tour)
	nodeCost := NodesCost(p, tour)

	return Result{
		Strategy:  strategy.Name(),
		Instance:  p.Name,
		Objective: tourLen + nodeCost,
		TourLen:   tourLen,
		NodeCost:  nodeCost,
		Tour:      tour,
		Elapsed:   elapsed,
	}
}

// SolveInstance loads an instance into memory, runs strategy on it and scores
// the result. Loading is not part of Elapsed — only the run is timed.
func SolveInstance(strategy Strategy, path string, iteration int) (Result, error) {
	p, err := LoadProblem(path)
	if err != nil {
		return Result{}, err
	}
	return SolveProblem(strategy, p, iteration), nil
}

// Benchmark runs every strategy on every instance, repeat times each, and
// returns the results in strategy-major order. Instances are loaded once.
//
// Each run gets its own RNG seeded from seed plus the run index, so every
// strategy faces the same random stream on a given run and the comparison
// does not depend on what ran before it.
func Benchmark(strategies []Strategy, instances []string, repeat int) ([]Result, error) {
	problems := make([]*Problem, len(instances))
	for i, path := range instances {
		p, err := LoadProblem(path)
		if err != nil {
			return nil, err
		}
		problems[i] = p
	}

	results := make([]Result, 0, len(strategies)*len(problems)*repeat)
	for _, p := range problems {
		for _, s := range strategies {
			for r := 0; r < repeat; r++ {
				results = append(results, SolveProblem(s, p, r))
			}
		}
	}
	return results, nil
}
