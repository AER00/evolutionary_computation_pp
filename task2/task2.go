package task2

import (
	"math"
	"slices"

	"OE/solver"
	"OE/task1"
)

var (
	_ solver.Strategy = (*RegretOnlyExtendAny)(nil)
	_ solver.Strategy = (*RegretOnlyGreedyCycle)(nil)
	_ solver.Strategy = (*RegretWeightedExtendAny)(nil)
	_ solver.Strategy = (*RegretWeightedGreedyCycle)(nil)
)

func extendAtBestRegretWeightedPosition(p *solver.Problem, start int, cyclic bool, regretWeight float64) []int {
	path := []int{start}
	used := make([]bool, p.N)
	used[start] = true

	for len(path) < p.M {
		bestScore, next, at := math.Inf(-1), -1, 0

		for v := range p.N {
			if used[v] {
				continue
			}
			score, insertionPlace := regretWeightedScore(p, path, v, cyclic, regretWeight)
			if score > bestScore {
				bestScore, next, at = score, v, insertionPlace
			}
		}

		path = slices.Insert(path, at, next)
		used[next] = true
	}

	return path
}

// Returns (score, insertionPlace) where score is the regret-weighted score of inserting v into path at insertionPlace,
// and insertionPlace is the index in path where v should be inserted to minimize the score.
// If cyclic is true, the path is considered cyclic for distance calculations.
// The regretWeight parameter controls how much weight to give to the regret term in the score calculation.
func regretWeightedScore(p *solver.Problem, path []int, v int, cyclic bool, regretWeight float64) (float64, int) {
	if regretWeight < 0 {
		panic("regretWeight must be non-negative")
	} else if regretWeight > 1 {
		panic("regretWeight must be less than or equal to 1")
	}

	bestDelta := math.MaxInt
	secondBestDelta := math.MaxInt
	bestInsertionPlace := -1

	for insertionPlace := 0; insertionPlace <= len(path); insertionPlace++ {
		delta := task1.ObjectiveDelta(p, path, v, insertionPlace, cyclic)

		if delta < bestDelta {
			secondBestDelta = bestDelta
			bestDelta = delta
			bestInsertionPlace = insertionPlace
		} else if delta < secondBestDelta {
			secondBestDelta = delta
		}
	}

	// Calculate regret as the difference between the best delta and the second-best delta
	regret := secondBestDelta - bestDelta

	// Calculate the regret-weighted score
	score := regretWeight*float64(regret) - (1-regretWeight)*float64(bestDelta)

	return score, bestInsertionPlace
}
