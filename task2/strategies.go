package task2

import (
	"OE/solver"
)

type RegretOnlyExtendAny struct{}

func (*RegretOnlyExtendAny) Name() string { return "RegretOnlyExtendAny" }

func (*RegretOnlyExtendAny) Solve(p *solver.Problem, iteration int) []int {
	path := []int{iteration}
	return ExtendAtBestRegretWeightedPosition(p, path, false, 1.0)
}

type RegretOnlyGreedyCycle struct{}

func (*RegretOnlyGreedyCycle) Name() string { return "RegretOnlyGreedyCycle" }

func (*RegretOnlyGreedyCycle) Solve(p *solver.Problem, iteration int) []int {
	path := []int{iteration}
	return ExtendAtBestRegretWeightedPosition(p, path, true, 1.0)
}

type RegretWeightedExtendAny struct{}

func (*RegretWeightedExtendAny) Name() string { return "RegretWeightedExtendAny" }

func (*RegretWeightedExtendAny) Solve(p *solver.Problem, iteration int) []int {
	path := []int{iteration}
	return ExtendAtBestRegretWeightedPosition(p, path, false, 0.5)
}

type RegretWeightedGreedyCycle struct{}

func (*RegretWeightedGreedyCycle) Name() string { return "RegretWeightedGreedyCycle" }

func (*RegretWeightedGreedyCycle) Solve(p *solver.Problem, iteration int) []int {
	path := []int{iteration}
	return ExtendAtBestRegretWeightedPosition(p, path, true, 0.5)
}
