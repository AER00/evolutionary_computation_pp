package solver

import (
	"math/rand"
	"sort"
)

// NearestNeighborTour builds a tour over nodes by repeatedly stepping to the
// closest unvisited node, starting from a random one of them.
func NearestNeighborTour(p *Problem, nodes []int, rng *rand.Rand) []int {
	start := nodes[rng.Intn(len(nodes))]

	unvisited := make(map[int]bool, len(nodes))
	for _, v := range nodes {
		unvisited[v] = true
	}
	delete(unvisited, start)

	tour := []int{start}
	cur := start

	for len(unvisited) > 0 {
		best := -1
		bestD := 1 << 30
		for v := range unvisited {
			if d := p.Dist[cur][v]; d < bestD {
				bestD = d
				best = v
			}
		}
		tour = append(tour, best)
		delete(unvisited, best)
		cur = best
	}
	return tour
}

// TwoOpt improves tour in place, reversing segments until no improving 2-opt
// move is left.
func TwoOpt(p *Problem, tour []int) []int {
	m := len(tour)
	improved := true
	for improved {
		improved = false
		for i := 0; i < m-1; i++ {
			for j := i + 2; j < m; j++ {
				if i == 0 && j == m-1 {
					continue
				}
				a, b := tour[i], tour[i+1]
				c, d := tour[j], tour[(j+1)%m]
				delta := p.Dist[a][c] + p.Dist[b][d] - p.Dist[a][b] - p.Dist[c][d]
				if delta < 0 {
					for l, r := i+1, j; l < r; l, r = l+1, r-1 {
						tour[l], tour[r] = tour[r], tour[l]
					}
					improved = true
				}
			}
		}
	}
	return tour
}

// CheapestNodes returns the indices of the m cheapest-cost nodes, ordered by
// cost. It is the usual starting selection for this problem.
func CheapestNodes(p *Problem, m int) []int {
	order := make([]int, p.N)
	for i := range order {
		order[i] = i
	}
	// Strict total order — equal costs fall back to the node index — so the
	// result does not depend on sort stability.
	sort.Slice(order, func(a, b int) bool {
		ca, cb := p.Costs[order[a]], p.Costs[order[b]]
		if ca != cb {
			return ca < cb
		}
		return order[a] < order[b]
	})
	return order[:m]
}
