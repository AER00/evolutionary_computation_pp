package task4

import (
	"OE/solver"
	"slices"
	"sort"
)

// for every node we prepare n_candidates candidate edges, sorted by distance. This is used to speed up the local search.
func prepareCandidates(p *solver.Problem, n_candidates int) [][]int {
	// Add node costs to the distance matrix
	wholeCost := make([][]int, p.N)
	for i := 0; i < p.N; i++ {
		wholeCost[i] = make([]int, p.N)
		for j := 0; j < p.N; j++ {
			wholeCost[i][j] = p.Dist[i][j] + p.Costs[j]
		}
	}

	// wholeCost[i] contains distances from node i + target node cost
	candidates := make([][]int, p.N)
	for i := 0; i < p.N; i++ {
		// Create a slice of indices for sorting
		indices := make([]int, p.N)
		for j := 0; j < p.N; j++ {
			indices[j] = j
		}

		// Sort indices based on the wholeCost values
		sort.Slice(indices, func(a, b int) bool {
			return wholeCost[i][indices[a]] < wholeCost[i][indices[b]]
		})

		// Remove the node itself from the candidates
		candidates[i] = make([]int, 0, n_candidates)
		for _, idx := range indices {
			if idx != i {
				candidates[i] = append(candidates[i], idx)
				if len(candidates[i]) >= n_candidates {
					break
				}
			}
		}
	}

	return candidates
}

func interRouteIteratorWithCandidates(p *solver.Problem, path []int, candidates [][]int, yieldMove func(i int, newNode int, delta int)) {
	for i, oldNode := range path {
		for _, newNode := range candidates[oldNode] {
			if slices.Contains(path, newNode) {
				continue // Skip if newNode is already in the path
			}
			// we want to add edge (oldNode, newNode)
			// we can insert newNode either before or after oldNode
			prevNode := path[(i-1+p.M)%p.M]
			prevPrevNode := path[(i-2+p.M)%p.M]
			nextNode := path[(i+1)%p.M]
			nextNextNode := path[(i+2)%p.M]

			// Calculate the objective delta for swapping newNode with prevNode
			deltaBefore := p.Costs[newNode] + p.Dist[prevPrevNode][newNode] + p.Dist[newNode][oldNode] - p.Costs[prevNode] - p.Dist[prevPrevNode][prevNode] - p.Dist[prevNode][oldNode]
			yieldMove((i-1+p.M)%p.M, newNode, deltaBefore)

			// Calculate the objective delta for swapping newNode with nextNode
			deltaAfter := p.Costs[newNode] + p.Dist[oldNode][newNode] + p.Dist[newNode][nextNextNode] - p.Costs[nextNode] - p.Dist[oldNode][nextNode] - p.Dist[nextNode][nextNextNode]
			yieldMove((i+1)%p.M, newNode, deltaAfter)
		}
	}
}

func intraRouteIteratorWithCandidates(p *solver.Problem, path []int, candidates [][]int, yieldMove func(i int, j int, delta int)) {
	for i, oldNode := range path {
		for _, newNode := range candidates[oldNode] {
			// we want to act only on the nodes that are already in the path
			j := slices.Index(path, newNode)
			if j < 0 {
				continue
			}

			// we cannot add an edge that is already there
			if j == (i-1+p.M)%p.M || j == (i+1)%p.M {
				continue
			}

			// For the candidate edge (oldNode, newNode), either remove the
			// edges after both endpoints or the edges before both endpoints.
			yieldTwoOpt := func(firstEdge int, secondEdge int) {
				edgeI, edgeJ := firstEdge, secondEdge
				if edgeI > edgeJ {
					edgeI, edgeJ = edgeJ, edgeI
				}
				if edgeJ-edgeI < 2 || (edgeI == 0 && edgeJ == p.M-1) {
					return
				}

				edgeINext := (edgeI + 1) % p.M
				edgeJNext := (edgeJ + 1) % p.M
				delta := -p.Dist[path[edgeI]][path[edgeINext]] -
					p.Dist[path[edgeJ]][path[edgeJNext]] +
					p.Dist[path[edgeI]][path[edgeJ]] +
					p.Dist[path[edgeINext]][path[edgeJNext]]
				yieldMove(edgeI, edgeJ, delta)
			}

			// Remove the two edges after oldNode and newNode.
			yieldTwoOpt(i, j)
			// Remove the two edges before oldNode and newNode.
			yieldTwoOpt((i-1+p.M)%p.M, (j-1+p.M)%p.M)
		}
	}
}

// improvePathSteepest uses the candidate edges to speed up the local search.
// it uses edge exchange as the intra-route neighborhood.
func improvePathSteepestWithCandidates(p *solver.Problem, path []int, candidates [][]int) []int {
	bestPath := make([]int, len(path))
	copy(bestPath, path)

	for true {
		bestDelta := 0
		var bestMove func()

		// Check inter-route moves
		interRouteIteratorWithCandidates(p, bestPath, candidates, func(i int, newNode int, delta int) {
			if delta < bestDelta {
				bestDelta = delta
				bestMove = func() {
					bestPath[i] = newNode
				}
			}
		})

		// Check intra-route moves (edge exchange)
		intraRouteIteratorWithCandidates(p, bestPath, candidates, func(i int, j int, delta int) {
			if delta < bestDelta {
				bestDelta = delta
				bestMove = func() {
					// Swap edges (i, i+1) and (j, j+1) by reversing the segment between i+1 and j.
					for k := 0; k < (j-i)/2; k++ {
						bestPath[i+1+k], bestPath[j-k] = bestPath[j-k], bestPath[i+1+k]
					}
				}
			}
		})

		if bestDelta >= 0 {
			// No improving move found, return the best path found
			return bestPath
		}

		// Apply the best move found
		bestMove()
	}

	return nil // This line is unreachable, but required to satisfy the compiler
}
