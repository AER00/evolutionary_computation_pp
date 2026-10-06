// Package task1 holds task 1's strategy: simulated annealing, starting from a
// nearest-neighbour tour over the cheapest nodes and finishing with a 2-opt
// polish.
package task1

import (
	"math"
	"math/rand"

	"OE/solver"
)

const (
	t0   = 1000.0 // starting temperature
	tMin = 0.01   // temperature reached on the last iteration
)

// SimulatedAnnealing is the optimization method implemented for task 1.
type SimulatedAnnealing struct {
	// Iterations is how many annealing steps a run performs.
	Iterations int
}

// New returns the task 1 strategy, configured to run iterations steps.
func New(iterations int) *SimulatedAnnealing {
	return &SimulatedAnnealing{Iterations: iterations}
}

// Name identifies the strategy.
func (s *SimulatedAnnealing) Name() string { return "sa" }

// Solve builds an initial solution and improves it with simulated annealing,
// alternating between 2-opt moves and swaps of a selected node for an
// unselected one. It returns the best tour seen.
func (s *SimulatedAnnealing) Solve(p *solver.Problem, rng *rand.Rand) []int {
	n, dist, costs := p.N, p.Dist, p.Costs

	// Initial selection: cheapest m nodes.
	selected := make([]bool, n)
	for _, v := range solver.CheapestNodes(p, p.M) {
		selected[v] = true
	}

	selectedNodes := make([]int, 0, p.M)
	for i := 0; i < n; i++ {
		if selected[i] {
			selectedNodes = append(selectedNodes, i)
		}
	}

	tour := solver.NearestNeighborTour(p, selectedNodes, rng)
	tour = solver.TwoOpt(p, tour)

	curCost := solver.Objective(p, tour)
	bestTour := append([]int(nil), tour...)
	bestCost := curCost

	// Stretch the cooling over the whole run, so the temperature falls from t0
	// to tMin across exactly Iterations steps instead of hitting the floor
	// after ~23k steps and hill-climbing for the rest of the run.
	alpha := math.Pow(tMin/t0, 1.0/float64(s.Iterations))
	T := t0

	for step := 0; step < s.Iterations; step++ {
		if rng.Float64() < 0.5 {
			// 2-opt move
			if len(tour) < 3 {
				continue
			}
			i := rng.Intn(len(tour))
			j := rng.Intn(len(tour))
			if i == j {
				continue
			}
			if i > j {
				i, j = j, i
			}
			if j-i < 2 {
				continue
			}
			if i == 0 && j == len(tour)-1 {
				continue
			}

			a, b := tour[i], tour[(i+1)%len(tour)]
			c, d := tour[j], tour[(j+1)%len(tour)]
			delta := dist[a][c] + dist[b][d] - dist[a][b] - dist[c][d]

			if delta < 0 || rng.Float64() < math.Exp(-float64(delta)/T) {
				for l, r := i+1, j; l < r; l, r = l+1, r-1 {
					tour[l], tour[r] = tour[r], tour[l]
				}
				curCost += delta
				if curCost < bestCost {
					bestCost = curCost
					bestTour = append([]int(nil), tour...)
				}
			}
		} else {
			// Swap one selected node for one unselected node.
			sel := make([]int, 0, n)
			unsel := make([]int, 0, n)
			for v := 0; v < n; v++ {
				if selected[v] {
					sel = append(sel, v)
				} else {
					unsel = append(unsel, v)
				}
			}
			if len(sel) == 0 || len(unsel) == 0 {
				continue
			}

			sv := sel[rng.Intn(len(sel))]
			uv := unsel[rng.Intn(len(unsel))]

			pos := -1
			for k, v := range tour {
				if v == sv {
					pos = k
					break
				}
			}
			if pos == -1 {
				continue
			}

			prev := tour[(pos-1+len(tour))%len(tour)]
			nxt := tour[(pos+1)%len(tour)]
			removeDelta := dist[prev][nxt] - dist[prev][sv] - dist[sv][nxt]

			newTour := make([]int, 0, len(tour)-1)
			newTour = append(newTour, tour[:pos]...)
			newTour = append(newTour, tour[pos+1:]...)

			bestInsDelta := 1 << 30
			bestPos := 0
			L := len(newTour)
			for k := 0; k < L; k++ {
				a := newTour[k]
				b := newTour[(k+1)%L]
				insDelta := dist[a][uv] + dist[uv][b] - dist[a][b]
				if insDelta < bestInsDelta {
					bestInsDelta = insDelta
					bestPos = k + 1
				}
			}

			delta := removeDelta + bestInsDelta - costs[sv] + costs[uv]

			if delta < 0 || rng.Float64() < math.Exp(-float64(delta)/T) {
				selected[sv] = false
				selected[uv] = true

				newTour = append(newTour, 0)
				copy(newTour[bestPos+1:], newTour[bestPos:])
				newTour[bestPos] = uv
				tour = newTour
				curCost += delta

				if curCost < bestCost {
					bestCost = curCost
					bestTour = append([]int(nil), tour...)
				}
			}
		}

		T *= alpha
	}

	return solver.TwoOpt(p, bestTour)
}
