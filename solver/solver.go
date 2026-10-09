package solver

import (
	"time"
)

type Strategy interface {
	Name() string
	Solve(p *Problem, iteration int) []int
}

type Result struct {
	Strategy  string
	Instance  string
	Objective int // TourLen + NodeCost
	TourLen   int
	NodeCost  int
	Tour      []int
	Iteration int // the run index; constructive methods start from this node
	Elapsed   time.Duration
}

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
		Iteration: iteration,
		Elapsed:   elapsed,
	}
}

func SolveInstance(strategy Strategy, path string, iteration int) (Result, error) {
	p, err := LoadProblem(path)
	if err != nil {
		return Result{}, err
	}
	return SolveProblem(strategy, p, iteration), nil
}

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
