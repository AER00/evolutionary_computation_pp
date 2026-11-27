package task8

import (
	"fmt"

	"OE/solver"
	"OE/task1"
	"OE/task5"
)

// Dataset is the full Task 8 analysis result for one instance.
type Dataset struct {
	Instance           string         `json:"instance"`
	SelectedNodes      int            `json:"selected_nodes"`
	BestLocalOptimum   int            `json:"best_local_optimum"`
	ReferenceObjective int            `json:"reference_objective"`
	ReferenceTour      []int          `json:"reference_tour"`
	LocalOptima        []LocalOptimum `json:"local_optima"`
}

// LocalOptimum holds the objective and similarity counts of one sampled optimum.
type LocalOptimum struct {
	Objective            int     `json:"objective"`
	AvgCommonEdges       float64 `json:"avg_common_edges"`
	AvgCommonNodes       float64 `json:"avg_common_nodes"`
	BestCommonEdges      int     `json:"best_common_edges"`
	BestCommonNodes      int     `json:"best_common_nodes"`
	ReferenceCommonEdges int     `json:"reference_common_edges"`
	ReferenceCommonNodes int     `json:"reference_common_nodes"`
	// IdenticalToReference marks an optimum equal to the reference cycle; such
	// a point would have 100% self-similarity and is omitted from charts.
	IdenticalToReference bool `json:"identical_to_reference"`
}

// solution is a compact cycle representation for O(M) similarity counting.
type solution struct {
	tour []int
	in   []bool     // in[node]: node is selected
	adj  [][2]int32 // adj[node]: its two cycle neighbours, -1 if not selected
}

func newSolution(n int, tour []int) solution {
	s := solution{
		tour: tour,
		in:   make([]bool, n),
		adj:  make([][2]int32, n),
	}
	for i := range s.adj {
		s.adj[i] = [2]int32{-1, -1}
	}
	m := len(tour)
	for i, node := range tour {
		s.in[node] = true
		s.adj[node] = [2]int32{
			int32(tour[(i+m-1)%m]),
			int32(tour[(i+1)%m]),
		}
	}
	return s
}

func (s solution) hasEdge(u, v int) bool {
	a := s.adj[u]
	return a[0] == int32(v) || a[1] == int32(v)
}

// commonCounts returns the number of shared edges and shared selected nodes.
func commonCounts(a, b solution) (edges, nodes int) {
	m := len(a.tour)
	for i, u := range a.tour {
		if b.in[u] {
			nodes++
		}
		if b.hasEdge(u, a.tour[(i+1)%m]) {
			edges++
		}
	}
	return edges, nodes
}

// Analyze descends from `count` random solutions and records, for each local
// optimum, its similarity to all other sampled optima, to the best sampled
// optimum, and to the reference tour.
func Analyze(p *solver.Problem, count int, reference []int) (Dataset, error) {
	if p == nil {
		return Dataset{}, fmt.Errorf("problem must not be nil")
	}
	if count < 2 {
		return Dataset{}, fmt.Errorf("at least two local optima are required")
	}
	if p.N <= 0 || p.M < 3 || p.M > p.N {
		return Dataset{}, fmt.Errorf("problem has invalid node or selection counts")
	}
	if !validTour(p, reference) {
		return Dataset{}, fmt.Errorf("reference tour is not a valid solution")
	}

	solutions, optima, bestIndex, err := sampleOptima(p, count)
	if err != nil {
		return Dataset{}, err
	}
	fillSimilarities(p, solutions, optima, bestIndex, newSolution(p.N, reference))

	return Dataset{
		Instance:           p.Name,
		SelectedNodes:      p.M,
		BestLocalOptimum:   bestIndex,
		ReferenceObjective: solver.Objective(p, reference),
		ReferenceTour:      append([]int(nil), reference...),
		LocalOptima:        optima,
	}, nil
}

// sampleOptima runs greedy local search from `count` random starts.
func sampleOptima(p *solver.Problem, count int) ([]solution, []LocalOptimum, int, error) {
	random := &task1.RandomSolution{}
	solutions := make([]solution, count)
	optima := make([]LocalOptimum, count)
	bestIndex := 0
	for i := 0; i < count; i++ {
		tour := task5.ImprovePathLM(p, random.Solve(p, i))
		if !validTour(p, tour) {
			return nil, nil, 0, fmt.Errorf("local search returned an invalid tour (sample %d)", i)
		}
		solutions[i] = newSolution(p.N, tour)
		optima[i].Objective = solver.Objective(p, tour)
		if optima[i].Objective < optima[bestIndex].Objective {
			bestIndex = i
		}
	}
	return solutions, optima, bestIndex, nil
}

// fillSimilarities computes all three similarity versions for every optimum.
func fillSimilarities(
	p *solver.Problem,
	solutions []solution,
	optima []LocalOptimum,
	bestIndex int,
	reference solution,
) {
	count := len(solutions)
	edgeTotals := make([]int, count)
	nodeTotals := make([]int, count)

	// Pairwise similarity is symmetric: compute each pair once.
	for i := 0; i < count; i++ {
		for j := i + 1; j < count; j++ {
			edges, nodes := commonCounts(solutions[i], solutions[j])
			edgeTotals[i] += edges
			edgeTotals[j] += edges
			nodeTotals[i] += nodes
			nodeTotals[j] += nodes
		}
	}

	for i := range optima {
		o := &optima[i]
		o.AvgCommonEdges = float64(edgeTotals[i]) / float64(count-1)
		o.AvgCommonNodes = float64(nodeTotals[i]) / float64(count-1)
		o.BestCommonEdges, o.BestCommonNodes = commonCounts(solutions[i], solutions[bestIndex])
		o.ReferenceCommonEdges, o.ReferenceCommonNodes = commonCounts(solutions[i], reference)
		o.IdenticalToReference = o.ReferenceCommonEdges == p.M
	}
}

func validTour(p *solver.Problem, tour []int) bool {
	if len(tour) != p.M {
		return false
	}
	seen := make([]bool, p.N)
	for _, node := range tour {
		if node < 0 || node >= p.N || seen[node] {
			return false
		}
		seen[node] = true
	}
	return true
}
