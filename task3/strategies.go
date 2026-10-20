package task3

import (
	"OE/solver"
	"OE/task1"
	"OE/task2"
)

type RandomStartFirstImprovement struct{}

func (*RandomStartFirstImprovement) Name() string { return "RandomStartFirstImprovement" }

func (*RandomStartFirstImprovement) Solve(p *solver.Problem, iteration int) []int {
	return generateAndImprove(p, iteration, (&task1.RandomSolution{}).Solve, improvePathFirst)
}

type GreedyStartFirstImprovement struct{}

func (*GreedyStartFirstImprovement) Name() string { return "GreedyStartFirstImprovement" }

func (*GreedyStartFirstImprovement) Solve(p *solver.Problem, iteration int) []int {
	return generateAndImprove(p, iteration, (&task2.RegretWeightedGreedyCycle{}).Solve, improvePathFirst)
}

type RandomStartBestImprovement struct{}

func (*RandomStartBestImprovement) Name() string { return "RandomStartBestImprovement" }

func (*RandomStartBestImprovement) Solve(p *solver.Problem, iteration int) []int {
	return generateAndImprove(p, iteration, (&task1.RandomSolution{}).Solve, improvePathSteepest)
}

type GreedyStartBestImprovement struct{}

func (*GreedyStartBestImprovement) Name() string { return "GreedyStartBestImprovement" }

func (*GreedyStartBestImprovement) Solve(p *solver.Problem, iteration int) []int {
	return generateAndImprove(p, iteration, (&task2.RegretWeightedGreedyCycle{}).Solve, improvePathSteepest)
}
