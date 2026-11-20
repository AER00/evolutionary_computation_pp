package task7

import (
	"math"
	"math/rand"
	"sort"

	"OE/solver"
)

// DestroyFraction is the share of nodes removed by every destroy operator.
const DestroyFraction = 0.3

// DestroyOp removes k nodes from a cyclic path and returns the remaining nodes
// in their original cyclic order.
type DestroyOp func(path []int, k int, rng *rand.Rand) []int

// destroyCount returns how many nodes to remove from a path of m nodes.
func destroyCount(m int) int {
	return int(math.Round(DestroyFraction * float64(m)))
}

// objective is the tour length plus the node costs of a closed path.
func objective(p *solver.Problem, path []int) int {
	total := 0
	for i, v := range path {
		total += p.Costs[v] + p.Dist[v][path[(i+1)%len(path)]]
	}
	return total
}

// RemoveRandomNodes removes k uniformly chosen nodes.
func RemoveRandomNodes(path []int, k int, rng *rand.Rand) []int {
	m := len(path)
	removed := make([]bool, m)
	for _, i := range rng.Perm(m)[:k] {
		removed[i] = true
	}
	kept := make([]int, 0, m-k)
	for i, v := range path {
		if !removed[i] {
			kept = append(kept, v)
		}
	}
	return kept
}

// RemoveSubpath removes k consecutive nodes of the cycle starting at a random
// position (the cycle is cut at a random place, so the subpath may wrap).
func RemoveSubpath(path []int, k int, rng *rand.Rand) []int {
	m := len(path)
	r := rng.Intn(m)
	kept := make([]int, 0, m-k)
	for i := 0; i < m-k; i++ {
		kept = append(kept, path[(r+k+i)%m])
	}
	return kept
}

// RemoveSubpaths removes k nodes in total as c in {3,4,5} disjoint subpaths.
// From a random start the cycle is split into c alternating (removed, kept)
// blocks whose lengths are random compositions of k and m-k into c positive
// parts, so the segments never overlap and exactly k nodes are removed.
func RemoveSubpaths(path []int, k int, rng *rand.Rand) []int {
	m := len(path)
	c := 3 + rng.Intn(3)
	c = min(c, k, m-k)
	if c < 1 {
		return RemoveSubpath(path, k, rng)
	}
	removedLens := composition(k, c, rng)
	keptLens := composition(m-k, c, rng)

	pos := rng.Intn(m)
	kept := make([]int, 0, m-k)
	for j := 0; j < c; j++ {
		pos += removedLens[j] // skip removed block
		for i := 0; i < keptLens[j]; i++ {
			kept = append(kept, path[pos%m])
			pos++
		}
	}
	return kept
}

// composition splits n into c random positive parts.
func composition(n, c int, rng *rand.Rand) []int {
	cuts := make([]int, 0, c-1)
	for _, x := range rng.Perm(n - 1)[:c-1] {
		cuts = append(cuts, x+1)
	}
	sort.Ints(cuts)
	parts := make([]int, 0, c)
	prev := 0
	for _, x := range cuts {
		parts = append(parts, x-prev)
		prev = x
	}
	return append(parts, n-prev)
}
