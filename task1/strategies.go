package task1

import (
	"OE/solver"
	"math"
	"math/rand"
)

type RandomSolution struct{}

func (*RandomSolution) Name() string { return "RandomSolution" }

func (*RandomSolution) Solve(p *solver.Problem, iteration int) []int {
	generator := rand.New(rand.NewSource(int64(iteration)))

	nodes := make([]int, p.N)
	for i := range nodes {
		nodes[i] = i
	}
	generator.Shuffle(p.N, func(i, j int) {
		nodes[i], nodes[j] = nodes[j], nodes[i]
	})

	return nodes[:p.M]
}

type ExtendEnd struct{}

func (*ExtendEnd) Name() string { return "ExtendEnd" }

func (*ExtendEnd) Solve(p *solver.Problem, iteration int) []int {
	path := []int{iteration}
	used := make([]bool, p.N)
	used[iteration] = true

	for len(path) < p.M {
		lastNode := path[len(path)-1]
		best := math.MaxInt
		next := -1

		for v := range p.N {
			if used[v] {
				continue
			}
			// Appending pays for the new node and for the edge that reaches it.
			delta := p.Costs[v] + p.Dist[lastNode][v]
			if delta < best {
				best, next = delta, v
			}
		}

		path = append(path, next)
		used[next] = true
	}

	return path
}

type ExtendAny struct{}

func (*ExtendAny) Name() string { return "ExtendAny" }

func (*ExtendAny) Solve(p *solver.Problem, iteration int) []int {
	return extendAtBestPosition(p, iteration, false)
}

type GreedyCycle struct{}

func (*GreedyCycle) Name() string { return "GreedyCycle" }

func (*GreedyCycle) Solve(p *solver.Problem, iteration int) []int {
	return extendAtBestPosition(p, iteration, true)
}
