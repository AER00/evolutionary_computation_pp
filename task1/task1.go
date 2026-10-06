// Package task1 holds task 1's strategy: simulated annealing, starting from a
// nearest-neighbour tour over the cheapest nodes and finishing with a 2-opt
// polish.
package task1

import (
	"math/rand"

	"OE/solver"
)

type RandomSolution struct {
	solver.Strategy
}

func (s *RandomSolution) Name() string {
	return "RandomSolution"
}

func (s *RandomSolution) Solve(p *solver.Problem, iteration int) []int {
	n, _, _ := p.N, p.Dist, p.Costs
	generator := rand.New(rand.NewSource(int64(iteration)))

	// Pick M distinct nodes at random.
	nodes := make([]int, n)
	for i := 0; i < n; i++ {
		nodes[i] = i
	}

	generator.Shuffle(
		n, func(i, j int) {
			nodes[i], nodes[j] = nodes[j], nodes[i]
		},
	)

	return nodes[:p.M]
}
