package task4

import (
	"fmt"

	"OE/solver"
	"OE/task1"
)

// CandidateSteepest starts from a random solution and runs steepest local
// search using only moves that introduce one of the K candidate edges of a
// node. The candidate lists are built once per problem in Prepare, outside the
// timed part of the benchmark, and are reused by every run.
type CandidateSteepest struct {
	K int

	preparedFor *solver.Problem
	candidates  [][]int
}

// NewCandidateSteepest returns a candidate-list steepest search with K
// candidates per node.
func NewCandidateSteepest(k int) *CandidateSteepest {
	return &CandidateSteepest{K: k}
}

func (s *CandidateSteepest) Name() string {
	return fmt.Sprintf("RandomStartSteepestEdges%dCandidates", s.K)
}

// Prepare builds the candidate lists for p. Benchmark calls it once before the
// timed runs, so the construction time is not charged to the first run.
func (s *CandidateSteepest) Prepare(p *solver.Problem) {
	if s.preparedFor == p {
		return
	}
	s.preparedFor = p
	s.candidates = prepareCandidates(p, s.K)
}

func (s *CandidateSteepest) Solve(p *solver.Problem, iteration int) []int {
	path := (&task1.RandomSolution{}).Solve(p, iteration)
	s.Prepare(p) // no-op when Benchmark has already prepared the lists
	return improvePathSteepestWithCandidates(p, path, s.candidates)
}
