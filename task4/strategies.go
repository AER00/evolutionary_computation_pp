package task4

import (
	"OE/solver"
	"OE/task1"
)

type RandomStartSteepestEdges13Candidates struct {
	candidatesFor *solver.Problem
	candidates    [][]int
}

func (*RandomStartSteepestEdges13Candidates) Name() string {
	return "RandomStartSteepestEdges13Candidates"
}

func (s *RandomStartSteepestEdges13Candidates) Solve(p *solver.Problem, iteration int) []int {
	path := (&task1.RandomSolution{}).Solve(p, iteration)
	if s.candidatesFor != p {
		s.candidatesFor = p
		s.candidates = prepareCandidates(p, 13)
	}
	return improvePathSteepestWithCandidates(p, path, s.candidates)
}

type RandomStartSteepestEdges10Candidates struct {
	candidatesFor *solver.Problem
	candidates    [][]int
}

func (*RandomStartSteepestEdges10Candidates) Name() string {
	return "RandomStartSteepestEdges10Candidates"
}

func (s *RandomStartSteepestEdges10Candidates) Solve(p *solver.Problem, iteration int) []int {
	path := (&task1.RandomSolution{}).Solve(p, iteration)
	if s.candidatesFor != p {
		s.candidatesFor = p
		s.candidates = prepareCandidates(p, 10)
	}
	return improvePathSteepestWithCandidates(p, path, s.candidates)
}
