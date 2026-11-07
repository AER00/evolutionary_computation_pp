package task5

import "OE/solver"

// move is stored by node ids (not positions), so it stays meaningful after
// the tour changes.
//
//	inter (edge == false): prev-old-next becomes prev-new-next; a=prev, b=old, c=next, d=new
//	2-opt (edge == true):  edges (a,b) and (c,d) are removed, edges (a,c) and (b,d) are added.
//
// For 2-opt the move is applicable only when the two removed edges currently
// have the same relative direction (a->b with c->d, or b->a with d->c).
// For every pair of edges X: p->q and Y: r->s two moves are stored:
//
//	same direction:     {a:p, b:q, c:r, d:s}  adds (p,r), (q,s)
//	inverted direction: {a:p, b:q, c:s, d:r}  adds (p,s), (q,r)
//
// Exactly one of them is applicable at any given moment; the other one becomes
// applicable after a reversal flips the relative direction of the two edges.
type move struct {
	edge       bool
	delta      int
	a, b, c, d int
}

// moveHeap is a typed min-heap by delta: the best move is popped first.
// Hand-written to avoid the interface boxing (and allocation) of container/heap.
type moveHeap []move

func (h moveHeap) Len() int { return len(h) }

func (h *moveHeap) push(x move) {
	*h = append(*h, x)
	s := *h
	i := len(s) - 1
	for i > 0 {
		parent := (i - 1) / 2
		if s[parent].delta <= s[i].delta {
			break
		}
		s[parent], s[i] = s[i], s[parent]
		i = parent
	}
}

func (h *moveHeap) pop() move {
	s := *h
	top := s[0]
	last := len(s) - 1
	s[0] = s[last]
	s = s[:last]
	*h = s

	i := 0
	for {
		l, r, smallest := 2*i+1, 2*i+2, i
		if l < last && s[l].delta < s[smallest].delta {
			smallest = l
		}
		if r < last && s[r].delta < s[smallest].delta {
			smallest = r
		}
		if smallest == i {
			break
		}
		s[i], s[smallest] = s[smallest], s[i]
		i = smallest
	}
	return top
}

// applyResult is the outcome of checking a move from LM against the current tour.
type applyResult int

const (
	moveDrop    applyResult = iota // a removed edge no longer exists: remove from LM
	moveKeep                       // edges exist, relative direction differs: keep in LM, not applicable now
	moveApplied                    // move was applied: remove from LM
)

type state struct {
	p    *solver.Problem
	m    int
	path []int
	pos  []int // position in path, -1 if the node is not selected
	lm   moveHeap
	keep []move // scratch: moves skipped (not applicable now) while browsing LM
}

func newState(p *solver.Problem, path []int) *state {
	s := &state{p: p, m: len(path), path: append([]int(nil), path...), pos: make([]int, p.N)}
	for i := range s.pos {
		s.pos[i] = -1
	}
	for i, v := range s.path {
		s.pos[v] = i
	}
	return s
}

// addInter evaluates putting every unselected node at position i.
func (s *state) addInter(i int) {
	p, m := s.p, s.m
	old, prev, next := s.path[i], s.path[(i-1+m)%m], s.path[(i+1)%m]
	base := -p.Costs[old] - p.Dist[prev][old] - p.Dist[old][next]
	for u := 0; u < p.N; u++ {
		if s.pos[u] >= 0 {
			continue
		}
		if d := base + p.Costs[u] + p.Dist[prev][u] + p.Dist[u][next]; d < 0 {
			s.lm.push(move{delta: d, a: prev, b: old, c: next, d: u})
		}
	}
}

// addEdges evaluates both 2-opt reconnections for edge e (path[e]->path[e+1])
// and edge f (path[f]->path[f+1]). The reconnection that is valid in the
// current orientation and the one that becomes valid after a relative flip are
// both stored; improving ones go to LM.
func (s *state) addEdges(e, f int) {
	p, m := s.p, s.m
	if e > f {
		e, f = f, e
	}
	if f-e < 2 || (e == 0 && f == m-1) {
		return
	}
	a, b, c, d := s.path[e], s.path[e+1], s.path[f], s.path[(f+1)%m]
	removed := p.Dist[a][b] + p.Dist[c][d]
	// same direction: add (a,c), (b,d)
	if delta := p.Dist[a][c] + p.Dist[b][d] - removed; delta < 0 {
		s.lm.push(move{edge: true, delta: delta, a: a, b: b, c: c, d: d})
	}
	// relatively inverted: add (a,d), (b,c); stored as edges (a,b) and (d,c)
	if delta := p.Dist[a][d] + p.Dist[b][c] - removed; delta < 0 {
		s.lm.push(move{edge: true, delta: delta, a: a, b: b, c: d, d: c})
	}
}

// evaluateAll adds every improving move of the initial solution to LM,
// including the moves with relatively inverted edges.
func (s *state) evaluateAll() {
	for i := 0; i < s.m; i++ {
		s.addInter(i)
		for j := i + 2; j < s.m; j++ {
			s.addEdges(i, j)
		}
	}
}

// tryApply checks mv against the current tour:
//   - a removed edge no longer exists          -> moveDrop
//   - edges exist, relative direction differs  -> moveKeep (not applied)
//   - edges exist, same relative direction
//     (both forward or both reversed)          -> applied, moveApplied
func (s *state) tryApply(mv move) applyResult {
	m := s.m
	if !mv.edge {
		i := s.pos[mv.b]
		if i < 0 || s.pos[mv.d] >= 0 {
			return moveDrop
		}
		pa, na := s.path[(i-1+m)%m], s.path[(i+1)%m]
		if (pa != mv.a || na != mv.c) && (pa != mv.c || na != mv.a) {
			return moveDrop // a removed edge no longer exists
		}
		s.applyInter(i, mv.d)
		return moveApplied
	}

	pa, pb, pc, pd := s.pos[mv.a], s.pos[mv.b], s.pos[mv.c], s.pos[mv.d]
	if pa < 0 || pb < 0 || pc < 0 || pd < 0 {
		return moveDrop // a node was removed from the tour
	}
	forwardAB, reverseAB := pb == (pa+1)%m, pa == (pb+1)%m
	forwardCD, reverseCD := pd == (pc+1)%m, pc == (pd+1)%m
	if !(forwardAB || reverseAB) || !(forwardCD || reverseCD) {
		return moveDrop // an edge no longer exists
	}
	switch {
	case forwardAB && forwardCD: // same direction
		s.applyEdge(min(pa, pc), max(pa, pc))
	case reverseAB && reverseCD: // both reversed: forward edges are b->a and d->c
		s.applyEdge(min(pb, pd), max(pb, pd))
	default:
		return moveKeep // relative direction differs
	}
	return moveApplied
}

// applyInter puts newNode at position i and generates the affected moves.
func (s *state) applyInter(i, newNode int) {
	p, m := s.p, s.m
	old := s.path[i]
	s.path[i], s.pos[newNode], s.pos[old] = newNode, i, -1

	prev, next := (i-1+m)%m, (i+1)%m
	// Positions whose neighbours changed: all unselected nodes are candidates.
	s.addInter(prev)
	s.addInter(i)
	s.addInter(next)
	// The removed node can now be inserted at any other position.
	for q := 0; q < m; q++ {
		if q == prev || q == i || q == next {
			continue
		}
		o, pv, nx := s.path[q], s.path[(q-1+m)%m], s.path[(q+1)%m]
		if d := p.Costs[old] + p.Dist[pv][old] + p.Dist[old][nx] -
			p.Costs[o] - p.Dist[pv][o] - p.Dist[o][nx]; d < 0 {
			s.lm.push(move{delta: d, a: pv, b: o, c: nx, d: old})
		}
	}
	// The two new edges (prev->i and i->next) are paired with every other edge.
	for f := 0; f < m; f++ {
		s.addEdges(prev, f)
		s.addEdges(i, f)
	}
}

// applyEdge reverses path[i+1..j] (i<j) and generates the affected moves.
// Only the two new edges (positions i and j) need to be paired with the other
// edges: all other pairs keep their stored moves, since both reconnection
// variants (same / inverted direction) are already in LM.
func (s *state) applyEdge(i, j int) {
	m := s.m
	for l, r := i+1, j; l < r; l, r = l+1, r-1 {
		s.path[l], s.path[r] = s.path[r], s.path[l]
	}
	for t := i + 1; t <= j; t++ {
		s.pos[s.path[t]] = t
	}

	// Nodes with new neighbours: inter moves must be re-evaluated.
	for _, k := range [4]int{i, i + 1, j, (j + 1) % m} {
		s.addInter(k)
	}
	// The two new edges against all other edges.
	for f := 0; f < m; f++ {
		s.addEdges(i, f)
		if f != i {
			s.addEdges(j, f)
		}
	}
}

// improvePathLM is steepest local search driven by the list of improving moves.
// LM is browsed best-first. Moves with a missing edge are dropped, moves whose
// edges currently have a different relative direction are skipped but kept
// (they are pushed back after a move is applied), and the first applicable move
// is applied. The search ends when no applicable move remains in LM.
func improvePathLM(p *solver.Problem, path []int) []int {
	s := newState(p, path)
	s.evaluateAll()

	for {
		s.keep = s.keep[:0]
		applied := false
		for s.lm.Len() > 0 {
			mv := s.lm.pop()
			res := s.tryApply(mv)
			if res == moveKeep {
				s.keep = append(s.keep, mv)
				continue
			}
			if res == moveApplied {
				applied = true
				break
			}
			// moveDrop: discard
		}
		if !applied {
			break // nothing applicable left
		}
		for _, mv := range s.keep {
			s.lm.push(mv)
		}
	}
	return s.path
}
