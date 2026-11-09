package task6

import (
	"time"

	"OE/solver"
)

// MSLS is multiple start local search: the best of Starts steepest local
// searches (list-of-moves version) run from random solutions.
type MSLS struct{ Starts int }

func NewMSLS(starts int) *MSLS { return &MSLS{Starts: starts} }

var _ solver.Strategy = (*MSLS)(nil)

func (*MSLS) Name() string { return "MSLS" }

func (s *MSLS) Solve(p *solver.Problem, iteration int) []int {
	return msls(p, s.Starts, iteration)
}

// ILS is iterated local search that stops after Budget (the average MSLS
// time). The number of local search runs of every ILS run is available
// through LSRuns.
type ILS struct {
	Budget time.Duration
	runs   []int // local search runs of each ILS run, in order of execution
}

func NewILS(budget time.Duration) *ILS { return &ILS{Budget: budget} }

var _ solver.Strategy = (*ILS)(nil)

func (*ILS) Name() string { return "ILS" }

func (s *ILS) Solve(p *solver.Problem, iteration int) []int {
	path, runs := ils(p, s.Budget, iteration)
	s.runs = append(s.runs, runs)
	return path
}

// LSRuns returns the number of local search runs of every ILS run.
func (s *ILS) LSRuns() []int { return s.runs }
