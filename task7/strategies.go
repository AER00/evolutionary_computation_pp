package task7

import (
	"fmt"
	"math/rand"
	"time"

	"OE/solver"
	"OE/task1"
	"OE/task2"
	"OE/task5"
)

// LNS is large neighborhood search with a configurable destroy operator and
// optional local search after repair. It stops after Budget (the average MSLS
// time). The number of main-loop iterations of every run is available through
// Iterations.
type LNS struct {
	DestroyName  string
	DestroyOp    DestroyOp
	UseLS        bool
	Budget       time.Duration
	RegretWeight float64 // best weight found in the greedy-heuristics assignment
	iters        []int
}

func NewLNS(destroyName string, destroyOp DestroyOp, useLS bool, budget time.Duration, regretWeight float64) *LNS {
	return &LNS{
		DestroyName:  destroyName,
		DestroyOp:    destroyOp,
		UseLS:        useLS,
		Budget:       budget,
		RegretWeight: regretWeight,
	}
}

var _ solver.Strategy = (*LNS)(nil)

func (s *LNS) Name() string {
	if s.UseLS {
		return fmt.Sprintf("LNS-%s+LS", s.DestroyName)
	}
	return fmt.Sprintf("LNS-%s", s.DestroyName)
}

func (s *LNS) Solve(p *solver.Problem, iteration int) []int {
	path, n := lns(p, s.Budget, iteration, s.RegretWeight, s.UseLS, s.DestroyOp)
	s.iters = append(s.iters, n)
	return path
}

// Iterations returns the number of main-loop iterations of every run.
func (s *LNS) Iterations() []int { return s.iters }

// lns runs large neighborhood search until budget elapses and returns the final
// solution and the number of main-loop iterations.
//
// Start: random solution improved by steepest local search (always).
// Loop:  destroy (remove 30% of nodes) -> regret repair -> [local search if
// useLS] -> accept if not worse (minimisation, equal accepted for plateau moves).
func lns(
	p *solver.Problem,
	budget time.Duration,
	iteration int,
	regretWeight float64,
	useLS bool,
	destroyOp DestroyOp,
) ([]int, int) {
	start := time.Now()
	rng := rand.New(rand.NewSource(int64(iteration)*104729 + 3))

	cur := task5.ImprovePathLM(p, (&task1.RandomSolution{}).Solve(p, iteration*1_000_003))
	curObj := objective(p, cur)
	k := destroyCount(len(cur))

	iters := 0
	for time.Since(start) < budget {
		y := destroyOp(cur, k, rng)
		y = task2.ExtendAtBestRegretWeightedPosition(p, y, true, regretWeight)
		if useLS {
			y = task5.ImprovePathLM(p, y)
		}
		iters++
		if obj := objective(p, y); obj <= curObj {
			cur, curObj = y, obj
		}
	}
	return cur, iters
}
