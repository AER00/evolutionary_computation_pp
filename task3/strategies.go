package task3

import (
	"OE/solver"
	"OE/task1"
	"OE/task2"
	"math/rand"
)

// localSearch is one of the eight combinations: steepest or greedy local
// search, nodes- or edges-exchange inside the tour, random or greedy start.
type localSearch struct {
	name        string
	steepest    bool
	intra       IntraMove
	greedyStart bool
}

var _ solver.Strategy = (*localSearch)(nil)

func (s *localSearch) Name() string { return s.name }

func (s *localSearch) Solve(p *solver.Problem, iteration int) []int {
	var path []int
	if s.greedyStart {
		path = (&task2.RegretWeightedGreedyCycle{}).Solve(p, iteration)
	} else {
		path = (&task1.RandomSolution{}).Solve(p, iteration)
	}

	if s.steepest {
		return improvePathSteepest(p, path, s.intra)
	}

	// Seed the random browse order with the run index so that the whole
	// experiment is reproducible.
	rng := rand.New(rand.NewSource(int64(iteration)))
	return improvePathFirst(p, path, s.intra, rng)
}

// All returns the eight required combinations in a fixed order.
func All() []solver.Strategy {
	searches := []struct {
		label    string
		steepest bool
	}{
		{"Steepest", true},
		{"Greedy", false},
	}
	intras := []struct {
		label string
		move  IntraMove
	}{
		{"Nodes", NodesExchange},
		{"Edges", EdgesExchange},
	}
	starts := []struct {
		label  string
		greedy bool
	}{
		{"RandomStart", false},
		{"GreedyStart", true},
	}

	strategies := make([]solver.Strategy, 0, len(searches)*len(intras)*len(starts))
	for _, search := range searches {
		for _, intra := range intras {
			for _, start := range starts {
				strategies = append(strategies, &localSearch{
					name:        start.label + search.label + intra.label,
					steepest:    search.steepest,
					intra:       intra.move,
					greedyStart: start.greedy,
				})
			}
		}
	}
	return strategies
}
