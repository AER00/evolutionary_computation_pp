package task5

import (
	"testing"

	"OE/solver"
	"OE/task1"
)

type traceStep struct {
	delta int
	after []int
}

func sameTour(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// fullTrace runs the full-neighbourhood steepest search and records every
// accepted move's delta and the resulting tour. It mirrors
// task3.improvePathSteepest.
func fullTrace(p *solver.Problem, path []int) []traceStep {
	x := append([]int(nil), path...)
	m := len(x)
	var trace []traceStep

	for {
		bestDelta := 0
		var bestMove func()

		used := make([]bool, p.N)
		for _, v := range x {
			used[v] = true
		}
		unused := make([]int, 0, p.N-m)
		for v := 0; v < p.N; v++ {
			if !used[v] {
				unused = append(unused, v)
			}
		}

		for i := 0; i < m; i++ {
			old := x[i]
			prev := x[(i-1+m)%m]
			next := x[(i+1)%m]
			for _, u := range unused {
				d := p.Costs[u] + p.Dist[prev][u] + p.Dist[u][next] -
					p.Costs[old] - p.Dist[prev][old] - p.Dist[old][next]
				if d < bestDelta {
					bestDelta = d
					ii, uu := i, u
					bestMove = func() { x[ii] = uu }
				}
			}
		}

		for i := 0; i < m; i++ {
			for j := i + 2; j < m; j++ {
				if i == 0 && j == m-1 {
					continue
				}
				a, b := x[i], x[(i+1)%m]
				c, d := x[j], x[(j+1)%m]
				delta := p.Dist[a][c] + p.Dist[b][d] - p.Dist[a][b] - p.Dist[c][d]
				if delta < bestDelta {
					bestDelta = delta
					ii, jj := i, j
					bestMove = func() {
						for k := 0; k < (jj-ii)/2; k++ {
							x[ii+1+k], x[jj-k] = x[jj-k], x[ii+1+k]
						}
					}
				}
			}
		}

		if bestDelta >= 0 {
			return trace
		}
		bestMove()
		trace = append(trace, traceStep{delta: bestDelta, after: append([]int(nil), x...)})
	}
}

// lmTrace mirrors improvePathLM but records every applied move's delta and the
// resulting tour.
func lmTrace(p *solver.Problem, path []int) []traceStep {
	s := newState(p, path)
	s.evaluateAll()
	var trace []traceStep

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
				trace = append(trace,
					traceStep{delta: mv.delta, after: append([]int(nil), s.path...)})
				applied = true
				break
			}
		}
		if !applied {
			return trace
		}
		for _, mv := range s.keep {
			s.lm.push(mv)
		}
	}
}

// TestDeltaTraceMatchesFullSearch checks that, while the two searches are on
// the same tour, they always accept the same best delta. A divergence is only
// allowed at a tie: same delta, different move (and then different tours).
func TestDeltaTraceMatchesFullSearch(t *testing.T) {
	for _, name := range []string{"../TSPA.csv", "../TSPB.csv"} {
		p, err := solver.LoadProblem(name)
		if err != nil {
			t.Fatal(err)
		}
		tieDiverged := 0
		bugChecked := 0
		for seed := 0; seed < 200; seed++ {
			start := (&task1.RandomSolution{}).Solve(p, seed)
			full := fullTrace(p, start)
			lm := lmTrace(p, start)
			prevFull := start
			prevLM := start
			n := len(full)
			if len(lm) < n {
				n = len(lm)
			}
			divergedAtTie := false
			for i := 0; i < n; i++ {
				if !sameTour(prevFull, prevLM) {
					divergedAtTie = true // divergence already happened at an earlier tie
					break
				}
				if full[i].delta != lm[i].delta {
					t.Fatalf("%s seed %d step %d: same pre-tour but different best delta: full %d, lm %d",
						name, seed, i, full[i].delta, lm[i].delta)
				}
				bugChecked++
				if !sameTour(full[i].after, lm[i].after) {
					tieDiverged++ // same delta, different move: a tie
					divergedAtTie = true
					break
				}
				prevFull = full[i].after
				prevLM = lm[i].after
			}
			if !divergedAtTie && len(full) != len(lm) {
				t.Logf("%s seed %d: identical tours through both traces, full has %d moves, lm %d",
					name, seed, len(full), len(lm))
				t.Fatalf("%s seed %d: same tour but trace lengths differ: full %d, lm %d",
					name, seed, len(full), len(lm))
			}
		}
		t.Logf("%s: %d accepted moves compared before any divergence, %d seeds diverged at a tie",
			name, bugChecked, tieDiverged)
	}
}
