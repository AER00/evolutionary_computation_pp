package task4

import (
	"OE/solver"
	"sort"
)

// prepareCandidates builds, for every node, the n candidates ordered by the
// task's criterion: distance plus the cost of the target node. The list is
// fixed for the whole search; it is not updated when the tour changes.
func prepareCandidates(p *solver.Problem, nCandidates int) [][]int {
	wholeCost := make([][]int, p.N)
	for i := 0; i < p.N; i++ {
		wholeCost[i] = make([]int, p.N)
		for j := 0; j < p.N; j++ {
			wholeCost[i][j] = p.Dist[i][j] + p.Costs[j]
		}
	}

	indices := make([]int, p.N)
	candidates := make([][]int, p.N)
	for i := 0; i < p.N; i++ {
		for j := 0; j < p.N; j++ {
			indices[j] = j
		}
		sort.Slice(indices, func(a, b int) bool {
			return wholeCost[i][indices[a]] < wholeCost[i][indices[b]]
		})

		candidates[i] = make([]int, 0, nCandidates)
		for _, idx := range indices {
			if idx != i {
				candidates[i] = append(candidates[i], idx)
				if len(candidates[i]) >= nCandidates {
					break
				}
			}
		}
	}

	return candidates
}

// interRouteIteratorWithCandidates yields every inter-route node exchange that
// would introduce at least one candidate edge. To create the edge
// (oldNode, newNode) either the predecessor or the successor of oldNode is
// replaced by newNode, so both replacements are yielded.
//
// selected[v] tells whether v is in the tour. The iterator reads it instead of
// scanning the path, so the membership test is O(1).
func interRouteIteratorWithCandidates(p *solver.Problem, path []int, candidates [][]int, selected []bool, yieldMove func(pos int, newNode int, delta int)) {
	m := len(path)
	for i := 0; i < m; i++ {
		oldNode := path[i]
		prevNode := path[(i-1+m)%m]
		prevPrevNode := path[(i-2+m)%m]
		nextNode := path[(i+1)%m]
		nextNextNode := path[(i+2)%m]

		for _, newNode := range candidates[oldNode] {
			if selected[newNode] {
				continue
			}

			// Replace prevNode with newNode, creating (prevPrevNode, newNode, oldNode).
			deltaBefore := p.Costs[newNode] + p.Dist[prevPrevNode][newNode] + p.Dist[newNode][oldNode] -
				p.Costs[prevNode] - p.Dist[prevPrevNode][prevNode] - p.Dist[prevNode][oldNode]
			yieldMove((i-1+m)%m, newNode, deltaBefore)

			// Replace nextNode with newNode, creating (oldNode, newNode, nextNextNode).
			deltaAfter := p.Costs[newNode] + p.Dist[oldNode][newNode] + p.Dist[newNode][nextNextNode] -
				p.Costs[nextNode] - p.Dist[oldNode][nextNode] - p.Dist[nextNode][nextNextNode]
			yieldMove((i+1)%m, newNode, deltaAfter)
		}
	}
}

// intraRouteIteratorWithCandidates yields every 2-opt move that would create a
// candidate edge. For a candidate edge (oldNode, newNode) the move can remove
// the edges after both endpoints or the edges before both endpoints.
//
// pos[v] is the position of v in the path, or -1 when v is not selected. The
// iterator reads it instead of searching the path, so both the membership test
// and the position lookup are O(1).
func intraRouteIteratorWithCandidates(p *solver.Problem, path []int, candidates [][]int, pos []int, yieldMove func(edgeI int, edgeJ int, delta int)) {
	m := len(path)
	for i := 0; i < m; i++ {
		oldNode := path[i]
		for _, newNode := range candidates[oldNode] {
			j := pos[newNode]
			if j < 0 {
				continue
			}

			// Adjacent nodes already share an edge; there is nothing to add.
			if j == (i-1+m)%m || j == (i+1)%m {
				continue
			}

			yieldTwoOpt := func(firstEdge int, secondEdge int) {
				edgeI, edgeJ := firstEdge, secondEdge
				if edgeI > edgeJ {
					edgeI, edgeJ = edgeJ, edgeI
				}
				// Skip adjacent edges and the reversal of the whole tour.
				if edgeJ-edgeI < 2 || (edgeI == 0 && edgeJ == m-1) {
					return
				}

				edgeINext := (edgeI + 1) % m
				edgeJNext := (edgeJ + 1) % m
				delta := -p.Dist[path[edgeI]][path[edgeINext]] -
					p.Dist[path[edgeJ]][path[edgeJNext]] +
					p.Dist[path[edgeI]][path[edgeJ]] +
					p.Dist[path[edgeINext]][path[edgeJNext]]
				yieldMove(edgeI, edgeJ, delta)
			}

			// Remove the two edges after oldNode and newNode.
			yieldTwoOpt(i, j)
			// Remove the two edges before oldNode and newNode.
			yieldTwoOpt((i-1+m)%m, (j-1+m)%m)
		}
	}
}

// improvePathSteepestWithCandidates runs steepest descent over the candidate
// neighbourhood. It keeps two arrays in sync with the current path: selected
// tells whether a node is in the tour and pos gives its position (or -1).
// The candidate lists are fixed for the whole search.
//
// Moves can be generated from more than one candidate edge, so per-generation
// stamp arrays are used to evaluate each distinct move once.
func improvePathSteepestWithCandidates(p *solver.Problem, path []int, candidates [][]int) []int {
	bestPath := make([]int, len(path))
	copy(bestPath, path)
	m := len(bestPath)

	selected := make([]bool, p.N)
	pos := make([]int, p.N)
	for i := range pos {
		pos[i] = -1
	}
	for i, node := range bestPath {
		selected[node] = true
		pos[node] = i
	}

	interSeen := make([]int32, m*p.N)
	intraSeen := make([]int32, m*m)
	var generation int32

	for {
		generation++
		bestDelta := 0
		bestKind := 0 // 0 = none, 1 = inter-route, 2 = intra-route
		bestI, bestJ, bestNode := 0, 0, 0

		// Check inter-route moves.
		interRouteIteratorWithCandidates(p, bestPath, candidates, selected, func(i int, newNode int, delta int) {
			key := i*p.N + newNode
			if interSeen[key] == generation {
				return
			}
			interSeen[key] = generation
			if delta < bestDelta {
				bestDelta = delta
				bestKind = 1
				bestI = i
				bestNode = newNode
			}
		})

		// Check intra-route moves (edge exchange).
		intraRouteIteratorWithCandidates(p, bestPath, candidates, pos, func(i int, j int, delta int) {
			key := i*m + j
			if intraSeen[key] == generation {
				return
			}
			intraSeen[key] = generation
			if delta < bestDelta {
				bestDelta = delta
				bestKind = 2
				bestI = i
				bestJ = j
			}
		})

		if bestDelta >= 0 {
			// No improving move found, return the best path found.
			return bestPath
		}

		if bestKind == 1 {
			oldNode := bestPath[bestI]
			selected[oldNode] = false
			pos[oldNode] = -1
			bestPath[bestI] = bestNode
			selected[bestNode] = true
			pos[bestNode] = bestI
			continue
		}

		// Swap edges (i, i+1) and (j, j+1) by reversing the segment between i+1 and j.
		i, j := bestI, bestJ
		for k := 0; k < (j-i)/2; k++ {
			bestPath[i+1+k], bestPath[j-k] = bestPath[j-k], bestPath[i+1+k]
		}
		for t := i + 1; t <= j; t++ {
			pos[bestPath[t]] = t
		}
	}
}
