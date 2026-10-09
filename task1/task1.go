package task1

import (
	"math"
	"slices"

	"OE/solver"
)

// Enforce at compile time that *RandomSolution implements solver.Strategy
var (
	_ solver.Strategy = (*RandomSolution)(nil)
	_ solver.Strategy = (*ExtendEnd)(nil)
	_ solver.Strategy = (*ExtendAny)(nil)
	_ solver.Strategy = (*GreedyCycle)(nil)
)

func extendAtBestPosition(p *solver.Problem, start int, cyclic bool) []int {
	path := []int{start}
	used := make([]bool, p.N)
	used[start] = true

	for len(path) < p.M {
		best, next, at := math.MaxInt, -1, 0

		for v := range p.N {
			if used[v] {
				continue
			}
			for j := range len(path) + 1 {
				if delta := objectiveDelta(p, path, v, j, cyclic); delta < best {
					best, next, at = delta, v, j
				}
			}
		}

		path = slices.Insert(path, at, next)
		used[next] = true
	}

	return path
}

func objectiveDelta(p *solver.Problem, path []int, v, at int, cyclic bool) int {
	delta := p.Costs[v] // the new node's own cost is always paid

	if at == 0 {
		next := path[0]
		delta += p.Dist[v][next]
		if cyclic {
			last := path[len(path)-1]
			delta += p.Dist[last][v] - p.Dist[last][next]
		}
	} else if at == len(path) {
		last := path[len(path)-1]
		delta += p.Dist[last][v]
		if cyclic {
			next := path[0]
			delta += p.Dist[v][next] - p.Dist[last][next]
		}
	} else {
		prev, next := path[at-1], path[at]
		delta += p.Dist[prev][v] + p.Dist[v][next] - p.Dist[prev][next]
	}

	return delta
}
