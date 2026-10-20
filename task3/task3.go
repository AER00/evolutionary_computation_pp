package task3

import (
	"OE/solver"
	"math/rand"
)

// This iterator yields all possible neighborhoods for inter-route moves, with objective delta calculation.
func interRouteIterator(p *solver.Problem, path []int, yieldMove func(i int, newNode int, delta int) bool) {
	used := make([]bool, p.N)
	for _, node := range path {
		used[node] = true
	}

	// Create a list of unused nodes
	unusedNodes := []int{}
	for v := range p.N {
		if !used[v] {
			unusedNodes = append(unusedNodes, v)
		}
	}

	for i := 0; i < p.M; i++ {
		for j := 0; j < p.N-p.M; j++ {
			// yield the move of taking the j-th unused node and switching it with the i-th node in the path, along with the objective delta.

			oldNode := path[i]
			newNode := unusedNodes[j]

			prevNode := path[(i-1+p.M)%p.M]
			nextNode := path[(i+1)%p.M]

			// Calculate the objective delta for this move
			delta := p.Costs[newNode] + p.Dist[prevNode][newNode] + p.Dist[newNode][nextNode] - p.Costs[oldNode] - p.Dist[prevNode][oldNode] - p.Dist[oldNode][nextNode]

			// Yield the move and its delta (index i, newNode, delta)
			if yieldMove(i, newNode, delta) {
				return
			}
		}
	}
}

func intraRouteNodesExchangeIterator(p *solver.Problem, path []int, yieldMove func(i int, j int, delta int) bool) {
	for i := 0; i < p.M; i++ {
		for j := i + 1; j < p.M; j++ {
			// yield the move of switching the i-th and j-th nodes in the path, along with the objective delta.

			nodeI := path[i]
			nodeJ := path[j]

			prevI := path[(i-1+p.M)%p.M]
			nextI := path[(i+1)%p.M]
			prevJ := path[(j-1+p.M)%p.M]
			nextJ := path[(j+1)%p.M]

			var delta int
			if j == i+1 {
				// ... prevI, I, J, nextJ ...
				delta = p.Dist[prevI][nodeJ] + p.Dist[nodeI][nextJ] - p.Dist[prevI][nodeI] - p.Dist[nodeJ][nextJ]
			} else if i == 0 && j == p.M-1 {
				// ... prevJ, J, I, nextI ...
				delta = p.Dist[prevJ][nodeI] + p.Dist[nodeJ][nextI] - p.Dist[prevJ][nodeJ] - p.Dist[nodeI][nextI]
			} else {
				// ... prevI, I, nextI, ..., prevJ, J, nextJ ...
				delta = p.Dist[prevI][nodeJ] + p.Dist[nodeJ][nextI] + p.Dist[prevJ][nodeI] + p.Dist[nodeI][nextJ] - p.Dist[prevI][nodeI] - p.Dist[nodeI][nextI] - p.Dist[prevJ][nodeJ] - p.Dist[nodeJ][nextJ]
			}

			// Yield the move and its delta (indices i, j, delta)
			if yieldMove(i, j, delta) {
				return
			}
		}
	}
}

func intraRouteEdgesExchangeIterator(p *solver.Problem, path []int, yieldMove func(i int, j int, delta int) bool) {
	for i := 0; i < p.M; i++ {
		for j := i + 2; j < p.M; j++ {
			// yield the move of switching the edges (i, i+1) and (j, j+1) in the path, along with the objective delta.
			if i == 0 && j == p.M-1 {
				// skip the case where we would break the cycle
				continue
			}

			nodeI := path[i]
			nodeI1 := path[(i+1)%p.M]
			nodeJ := path[j]
			nodeJ1 := path[(j+1)%p.M]

			// Calculate the objective delta for this move
			delta := p.Dist[nodeI][nodeJ] + p.Dist[nodeI1][nodeJ1] - p.Dist[nodeI][nodeI1] - p.Dist[nodeJ][nodeJ1]

			// Yield the move and its delta (indices i, j, delta)
			if yieldMove(i, j, delta) {
				return
			}
		}
	}
}

func improvePathSteepest(p *solver.Problem, path []int) []int {
	bestPath := make([]int, len(path))
	copy(bestPath, path)

	for true {
		bestDelta := 0
		var bestMove func()

		// Check inter-route moves
		interRouteIterator(p, bestPath, func(i int, newNode int, delta int) bool {
			if delta < bestDelta {
				bestDelta = delta
				bestMove = func() {
					bestPath[i] = newNode
				}
			}
			return false // continue searching for better moves
		})

		// Check intra-route node exchanges
		intraRouteNodesExchangeIterator(p, bestPath, func(i int, j int, delta int) bool {
			if delta < bestDelta {
				bestDelta = delta
				bestMove = func() {
					bestPath[i], bestPath[j] = bestPath[j], bestPath[i]
				}
			}
			return false // continue searching for better moves
		})

		// Check intra-route edge exchanges
		intraRouteEdgesExchangeIterator(p, bestPath, func(i int, j int, delta int) bool {
			if delta < bestDelta {
				bestDelta = delta
				bestMove = func() {
					// Swap edges (i, i+1) and (j, j+1) by reversing the segment between i+1 and j
					for k := 0; k < (j-i)/2; k++ {
						bestPath[i+1+k], bestPath[j-k] = bestPath[j-k], bestPath[i+1+k]
					}
				}
			}
			return false // continue searching for better moves
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

func improvePathFirst(p *solver.Problem, path []int) []int {
	bestPath := make([]int, len(path))
	copy(bestPath, path)

	for true {
		improved := false

		for _, kind := range rand.Perm(3) {
			switch kind {
			case 0:
				interRouteIterator(p, bestPath, func(i int, newNode int, delta int) bool {
					if delta < 0 {
						bestPath[i] = newNode
						improved = true
						return true // stop searching for more moves
					}
					return false // continue searching for better moves
				})
			case 1:
				intraRouteNodesExchangeIterator(p, bestPath, func(i int, j int, delta int) bool {
					if delta < 0 {
						bestPath[i], bestPath[j] = bestPath[j], bestPath[i]
						improved = true
						return true // stop searching for more moves
					}
					return false // continue searching for better moves
				})
			case 2:
				intraRouteEdgesExchangeIterator(p, bestPath, func(i int, j int, delta int) bool {
					if delta < 0 {
						// Swap edges (i, i+1) and (j, j+1) by reversing the segment between i+1 and j
						for k := 0; k < (j-i)/2; k++ {
							bestPath[i+1+k], bestPath[j-k] = bestPath[j-k], bestPath[i+1+k]
						}
						improved = true
						return true // stop searching for more moves
					}
					return false // continue searching for better moves
				})
			}
			if improved {
				break // break out of the kind loop to start over
			}
		}

		if !improved {
			// No improving move found, return the best path found
			return bestPath
		}
	}

	return nil // This line is unreachable, but required to satisfy the compiler
}

func generateAndImprove(p *solver.Problem, iteration int, startingFunc func(p *solver.Problem, iteration int) []int, improveFunc func(p *solver.Problem, path []int) []int) []int {
	startingPath := startingFunc(p, iteration)
	improvedPath := improveFunc(p, startingPath)
	return improvedPath
}
