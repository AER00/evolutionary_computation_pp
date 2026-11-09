package task6

import (
	"math/rand"
	"time"

	"OE/solver"
	"OE/task1"
	"OE/task5"
)

// objective is the tour length plus the node costs of a closed path.
func objective(p *solver.Problem, path []int) int {
	total := 0
	for i, v := range path {
		total += p.Costs[v] + p.Dist[v][path[(i+1)%len(path)]]
	}
	return total
}

// localSearch runs steepest local search from a random solution.
func localSearch(p *solver.Problem, seed int) []int {
	return task5.ImprovePathLM(p, (&task1.RandomSolution{}).Solve(p, seed))
}

// msls returns the best of `starts` local searches run from random solutions.
func msls(p *solver.Problem, starts, iteration int) []int {
	var best []int
	bestObj := 0
	for i := 0; i < starts; i++ {
		path := localSearch(p, iteration*1_000_003+i)
		if obj := objective(p, path); best == nil || obj < bestObj {
			best, bestObj = path, obj
		}
	}
	return best
}

// ils runs iterated local search until budget elapses: perturb the current
// solution, run local search, accept if not worse. It returns the final
// solution and the number of local search runs.
func ils(p *solver.Problem, budget time.Duration, iteration int) ([]int, int) {
	start := time.Now()
	rng := rand.New(rand.NewSource(int64(iteration)*7919 + 1))

	cur := localSearch(p, iteration*1_000_003)
	curObj := objective(p, cur)
	runs := 1

	for time.Since(start) < budget {
		cand := task5.ImprovePathLM(p, perturb(p, cur, rng))
		runs++
		if obj := objective(p, cand); obj <= curObj { // accept equal: allows plateau moves
			cur, curObj = cand, obj
		}
	}
	return cur, runs
}

// perturb = local double-bridge (A B C D -> A C B D, |B|,|C| <= m/10)
// + 3 random node replacements (changes the selected set).
func perturb(p *solver.Problem, path []int, rng *rand.Rand) []int {
	m := len(path)
	L := m / 10

	// Rotate randomly, then swap two adjacent short segments B and C.
	r := rng.Intn(m)
	lenB, lenC := 1+rng.Intn(L), 1+rng.Intn(L)
	p1 := 1 + rng.Intn(m-lenB-lenC)
	p3 := p1 + lenB + lenC

	q := make([]int, 0, m)
	q = append(q, path[r:]...)
	q = append(q, path[:r]...)
	rotated := append([]int(nil), q...)

	q = append(q[:p1], rotated[p1+lenB:p3]...) // A + C
	q = append(q, rotated[p1:p1+lenB]...)      // + B
	q = append(q, rotated[p3:]...)             // + D

	// Replace 3 random selected nodes with distinct random unselected ones.
	if p.N > m {
		selected := make([]bool, p.N)
		for _, v := range q {
			selected[v] = true
		}
		for k := 0; k < 3; k++ {
			u := rng.Intn(p.N)
			for selected[u] {
				u = rng.Intn(p.N)
			}
			i := rng.Intn(m)
			selected[q[i]], selected[u] = false, true
			q[i] = u
		}
	}

	return q
}
