package task5

import (
	"OE/solver"
	"OE/task1"
)

// SteepestLM is steepest local search (inter-route node exchange + 2-opt)
// from a random start, accelerated with the list of improving moves.
type SteepestLM struct{}

var _ solver.Strategy = (*SteepestLM)(nil)

func (*SteepestLM) Name() string { return "RandomStartSteepestEdgesLM" }

func (*SteepestLM) Solve(p *solver.Problem, iteration int) []int {
	path := (&task1.RandomSolution{}).Solve(p, iteration)
	return ImprovePathLM(p, path)
}
