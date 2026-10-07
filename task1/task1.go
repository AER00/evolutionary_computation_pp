package task1

import (
	"math/rand"

	"OE/solver"
)

// Enforce at compile time that *RandomSolution implements solver.Strategy
var _ solver.Strategy = (*RandomSolution)(nil)
var _ solver.Strategy = (*ExtendEnd)(nil)

type RandomSolution struct{}
type ExtendEnd struct{}

func (s *RandomSolution) Name() string {
	return "RandomSolution"
}

func (s *RandomSolution) Solve(p *solver.Problem, iteration int) []int {
	n := p.N
	generator := rand.New(rand.NewSource(int64(iteration)))

	nodes := make([]int, n)
	for i := 0; i < n; i++ {
		nodes[i] = i
	}

	generator.Shuffle(n, func(i, j int) {
		nodes[i], nodes[j] = nodes[j], nodes[i]
	})

	return nodes[:p.M]
}

func (s *ExtendEnd) Name() string {
	return "ExtendEnd"
}

func (s *ExtendEnd) Solve(p *solver.Problem, iteration int) []int {
	path := make([]int, 0, p.M)
	used := make([]bool, p.N)

	// Initialize path
	path = append(path, iteration)
	used[iteration] = true

	// Find a node with minimum cost + distance to the last node in path
	for len(path) < p.M {
		lastNode := path[len(path)-1]
		minCost := int(^uint(0) >> 1) // Max int
		var nextNode int

		for i := 0; i < p.N; i++ {
			if used[i] {
				continue
			}
			cost := p.Costs[i] + p.Dist[lastNode][i]
			if cost < minCost {
				minCost = cost
				nextNode = i
			}
		}

		path = append(path, nextNode)
		used[nextNode] = true
	}

	return path
}
