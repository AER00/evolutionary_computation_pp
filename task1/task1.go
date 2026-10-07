package task1

import (
	"math/rand"

	"OE/solver"
)

// Enforce at compile time that *RandomSolution implements solver.Strategy
var _ solver.Strategy = (*RandomSolution)(nil)
var _ solver.Strategy = (*ExtendEnd)(nil)
var _ solver.Strategy = (*ExtendAny)(nil)
var _ solver.Strategy = (*GreedyCycle)(nil)

type RandomSolution struct{}
type ExtendEnd struct{}
type ExtendAny struct{}
type GreedyCycle struct{}

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

func getObjectiveChange(p *solver.Problem, path []int, newNode int, insertIndex int, isCyclic bool) int {
	// Calculate the change in objective when inserting newNode at insertIndex in path
	// This function assumes that path is a valid tour and newNode is not already in path

	// When adding a new node, we have to add its cost and the distance to its neighbors
	// we have to remove the distance between the two neighbors
	change := p.Costs[newNode]

	// Handle the cyclic case
	if insertIndex == 0 {
		nextNode := path[0]
		change += p.Dist[newNode][nextNode]
		if isCyclic {
			prevNode := path[len(path)-1]
			change += p.Dist[prevNode][newNode] - p.Dist[prevNode][nextNode]
		}
	} else if insertIndex == len(path) {
		prevNode := path[len(path)-1]
		change += p.Dist[prevNode][newNode]
		if isCyclic {
			nextNode := path[0]
			change += p.Dist[newNode][nextNode] - p.Dist[prevNode][nextNode]
		}
		// Handle the case where the new node is inserted in the middle of the path
	} else {
		prevNode := path[insertIndex-1]
		nextNode := path[insertIndex]
		change += p.Dist[prevNode][newNode] + p.Dist[newNode][nextNode] - p.Dist[prevNode][nextNode]
	}

	return change
}

func (s *ExtendAny) Name() string {
	return "ExtendAny"
}

func (s *ExtendAny) Solve(p *solver.Problem, iteration int) []int {
	path := make([]int, 0, p.M)
	used := make([]bool, p.N)

	// Initialize path
	path = append(path, iteration)
	used[iteration] = true

	// Find a node with minimum cost + distance to any node in path
	for len(path) < p.M {
		minCost := int(^uint(0) >> 1) // Max int
		var nextNode int
		var insertIndex int

		for i := 0; i < p.N; i++ {
			if used[i] {
				continue
			}

			for j := 0; j <= len(path); j++ {
				change := getObjectiveChange(p, path, i, j, false)
				if change < minCost {
					minCost = change
					nextNode = i
					insertIndex = j
				}
			}
		}

		// Insert the next node at the best position found
		path = append(path[:insertIndex], append([]int{nextNode}, path[insertIndex:]...)...)
		used[nextNode] = true
	}

	return path
}

func (s *GreedyCycle) Name() string {
	return "GreedyCycle"
}

func (s *GreedyCycle) Solve(p *solver.Problem, iteration int) []int {
	path := make([]int, 0, p.M)
	used := make([]bool, p.N)

	// Initialize path
	path = append(path, iteration)
	used[iteration] = true

	// Find a node with minimum cost + distance to any node in path
	for len(path) < p.M {
		minCost := int(^uint(0) >> 1) // Max int
		var nextNode int
		var insertIndex int

		for i := 0; i < p.N; i++ {
			if used[i] {
				continue
			}

			for j := 0; j <= len(path); j++ {
				change := getObjectiveChange(p, path, i, j, true)
				if change < minCost {
					minCost = change
					nextNode = i
					insertIndex = j
				}
			}
		}

		// Insert the next node at the best position found
		path = append(path[:insertIndex], append([]int{nextNode}, path[insertIndex:]...)...)
		used[nextNode] = true
	}

	return path
}
