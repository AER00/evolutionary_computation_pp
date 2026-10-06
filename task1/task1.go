// Package task1 implements the task 1 solver: simulated annealing over a
// quota-TSP / orienteering-style problem.
//
// Given n nodes with (x, y, cost), select exactly m = ceil(n/2) of them and
// form a Hamiltonian cycle through the selection, minimizing
//
//	tour length + sum of the selected nodes' costs
//
// with distances taken as Euclidean and rounded to the nearest integer.
//
// The primitives here (ReadCSV, TourLength, NearestNeighborTour, TwoOpt) are
// exported so later tasks can reuse them.
package task1

import (
	"bufio"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"
)

// ReadCSV loads an instance and returns the node count, the full n x n integer
// distance matrix, and the per-node costs.
func ReadCSV(filename string) (int, [][]int, []int, error) {
	f, err := os.Open(filename)
	if err != nil {
		return 0, nil, nil, err
	}
	defer f.Close()

	var xs, ys, costs []int
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		parts := strings.Split(line, ";")
		if len(parts) != 3 {
			continue
		}
		x, _ := strconv.Atoi(parts[0])
		y, _ := strconv.Atoi(parts[1])
		c, _ := strconv.Atoi(parts[2])
		xs = append(xs, x)
		ys = append(ys, y)
		costs = append(costs, c)
	}
	if err := scanner.Err(); err != nil {
		return 0, nil, nil, err
	}

	n := len(xs)
	dist := make([][]int, n)
	for i := range dist {
		dist[i] = make([]int, n)
	}
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			dx := xs[i] - xs[j]
			dy := ys[i] - ys[j]
			dist[i][j] = int(math.Sqrt(float64(dx*dx+dy*dy)) + 0.5)
		}
	}
	return n, dist, costs, nil
}

// TourLength returns the length of the closed tour visiting tour in order.
func TourLength(tour []int, dist [][]int) int {
	total := 0
	m := len(tour)
	for i := 0; i < m; i++ {
		total += dist[tour[i]][tour[(i+1)%m]]
	}
	return total
}

// NearestNeighborTour builds a tour over nodes by repeatedly stepping to the
// closest unvisited node, starting from a random node.
func NearestNeighborTour(nodes []int, dist [][]int, rng *rand.Rand) []int {
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
			d := dist[cur][v]
			if d < bestD {
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

// TwoOpt improves tour in place by repeatedly reversing segments until no
// improving 2-opt move remains.
func TwoOpt(tour []int, dist [][]int) []int {
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
				delta := dist[a][c] + dist[b][d] - dist[a][b] - dist[c][d]
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

// SolveInstance runs the full pipeline on one instance and prints the result.
//
// Each call draws its own starting tour from rng, so callers wanting
// independent runs should either reseed rng or accept a shared stream.
func SolveInstance(filename string, timeLimit time.Duration, rng *rand.Rand) {
	n, dist, costs, err := ReadCSV(filename)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	m := (n + 1) / 2

	// Initial selection: cheapest m nodes
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(i, j int) bool {
		return costs[order[i]] < costs[order[j]]
	})

	selected := make([]bool, n)
	for i := 0; i < m; i++ {
		selected[order[i]] = true
	}

	var selectedNodes []int
	for i := 0; i < n; i++ {
		if selected[i] {
			selectedNodes = append(selectedNodes, i)
		}
	}

	tour := NearestNeighborTour(selectedNodes, dist, rng)
	tour = TwoOpt(tour, dist)

	curCost := 0
	for _, v := range tour {
		curCost += costs[v]
	}
	curCost += TourLength(tour, dist)

	bestTour := append([]int(nil), tour...)
	bestCost := curCost

	startTime := time.Now()
	T := 1000.0
	alpha := 0.9995

	for time.Since(startTime) < timeLimit {
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
			// Swap one selected node with one unselected node
			var sel, unsel []int
			for i := 0; i < n; i++ {
				if selected[i] {
					sel = append(sel, i)
				} else {
					unsel = append(unsel, i)
				}
			}
			if len(sel) == 0 || len(unsel) == 0 {
				continue
			}

			s := sel[rng.Intn(len(sel))]
			u := unsel[rng.Intn(len(unsel))]

			pos := -1
			for k, v := range tour {
				if v == s {
					pos = k
					break
				}
			}
			if pos == -1 {
				continue
			}

			prev := tour[(pos-1+len(tour))%len(tour)]
			nxt := tour[(pos+1)%len(tour)]
			removeDelta := dist[prev][nxt] - dist[prev][s] - dist[s][nxt]

			newTour := make([]int, 0, len(tour)-1)
			newTour = append(newTour, tour[:pos]...)
			newTour = append(newTour, tour[pos+1:]...)

			bestInsDelta := 1 << 30
			bestPos := 0
			L := len(newTour)
			for k := 0; k < L; k++ {
				a := newTour[k]
				b := newTour[(k+1)%L]
				insDelta := dist[a][u] + dist[u][b] - dist[a][b]
				if insDelta < bestInsDelta {
					bestInsDelta = insDelta
					bestPos = k + 1
				}
			}

			delta := removeDelta + bestInsDelta - costs[s] + costs[u]

			if delta < 0 || rng.Float64() < math.Exp(-float64(delta)/T) {
				selected[s] = false
				selected[u] = true

				newTour = append(newTour, 0)
				copy(newTour[bestPos+1:], newTour[bestPos:])
				newTour[bestPos] = u
				tour = newTour
				curCost += delta

				if curCost < bestCost {
					bestCost = curCost
					bestTour = append([]int(nil), tour...)
				}
			}
		}

		T *= alpha
		if T < 0.01 {
			T = 0.01
		}
	}

	bestTour = TwoOpt(bestTour, dist)
	bestCost = 0
	for _, v := range bestTour {
		bestCost += costs[v]
	}
	bestCost += TourLength(bestTour, dist)

	fmt.Printf("\nInstance: %s\n", filename)
	fmt.Printf("Objective value: %d\n", bestCost)
	fmt.Printf("Selected nodes: %d\n", len(bestTour))
	fmt.Println("Tour:")
	fmt.Println(bestTour)
}
