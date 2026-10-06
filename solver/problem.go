// Package solver holds everything shared by every task: the problem
// definition, the objective function, the strategy seam, the runner that
// loads an instance / runs a strategy / measures the time, and the
// construction heuristics that more than one strategy builds on.
package solver

import (
	"bufio"
	"math"
	"os"
	"strconv"
	"strings"
)

// Problem is everything an optimization method may look at: the node costs and
// the precomputed distance matrix. Node coordinates are deliberately not kept
// — they are only needed while the matrix is built, so a method never depends
// on them and an instance given as a bare distance matrix would work the same.
type Problem struct {
	Name  string
	N     int
	M     int // how many nodes to select: ceil(N/2)
	Dist  [][]int
	Costs []int
}

// LoadProblem reads an x;y;cost CSV and builds the integer distance matrix.
// Lines that do not hold three parsable fields are skipped.
func LoadProblem(filename string) (*Problem, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
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
		return nil, err
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

	return &Problem{
		Name:  filename,
		N:     n,
		M:     (n + 1) / 2,
		Dist:  dist,
		Costs: costs,
	}, nil
}

// TourLength returns the length of the closed tour visiting tour in order.
func TourLength(p *Problem, tour []int) int {
	total := 0
	m := len(tour)
	for i := 0; i < m; i++ {
		total += p.Dist[tour[i]][tour[(i+1)%m]]
	}
	return total
}

// NodesCost returns the summed cost of the nodes in tour.
func NodesCost(p *Problem, tour []int) int {
	total := 0
	for _, v := range tour {
		total += p.Costs[v]
	}
	return total
}

// Objective is the quantity being minimized: the tour length plus the cost of
// the selected nodes.
func Objective(p *Problem, tour []int) int {
	return TourLength(p, tour) + NodesCost(p, tour)
}
