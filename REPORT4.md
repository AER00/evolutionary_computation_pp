# Task 4 — Steepest local search with candidate moves

- 160296: Paweł Laszczyk
- 160314: Ashley Jurga

**Solver revision:** `75ec58b` (branch `task4-fixes`)<br>
**Source code link**: [GitHub](https://github.com/AER00/evolutionary_computation_pp)<br>
**Instances:** `TSPA.csv`, `TSPB.csv` — 200 nodes each<br>
**Methods:** `RandomSolution` (baseline), full-neighbourhood steepest local search with a random start (reference), and the same search restricted to candidate lists with `k = 10` and `k = 13`<br>
**Previous methods:** the same methods as in Report 3, registered in the same extended benchmark run for consistent timings; the algorithms are unchanged.

---

## 1. The problem

The problem is the same as in Tasks 1–3. Each instance has `N` points with coordinates and a
**cost**. We select exactly

```
M = ceil(N / 2) = 100
```

points and put them in a closed tour. The objective is

```
objective = tour length + cost of the selected nodes
```

and lower is better. Distances are Euclidean and rounded to the nearest integer, so all objective
values are integers. A solution is stored as the list of selected nodes in visit order; the last
node connects back to the first one.

Task 4 changes the neighbourhood, not the objective. The local search from Task 3 checks every
inter-route exchange and every 2-opt move. Candidate moves check only a small part of that
neighbourhood, chosen by a heuristic. The goal is to keep most of the quality while evaluating
far fewer moves.

## 2. Methods

### 2.1 Candidate lists

For every node `i` we build a list `C[i]` of `k` other nodes. The nodes are sorted by

```
score[i][j] = dist[i][j] + cost[j]
```

and the `k` smallest are kept. The cost term matters: a cheap node is more attractive than an
expensive one at the same distance. The lists are built once per instance and do not change
during the search.

```
BuildCandidates(instance, k):
    for i ← 0 to N-1:
        for j ← 0 to N-1:
            score[i][j] ← dist[i][j] + cost[j]

        order ← [0, 1, ..., N-1]
        sort order by score[i][order[a]] ascending

        C[i] ← []
        for v in order:
            if v ≠ i:
                C[i].append(v)
                if len(C[i]) = k:
                    break

    return C
```

The lists are directed: `v` can be in `C[i]` even when `i` is not in `C[v]`. The iterators below
read the list of every selected node; section 2.6 explains what this misses for inter-route moves.

### 2.2 Inter-route moves through candidate edges

An inter-route move replaces one selected node with one unselected node. In the full
neighbourhood every pair `(position, unselected node)` is checked. In the candidate version we
only check moves that create a candidate edge.

Suppose the tour contains `old` and `new` is in `C[old]`. To create the edge `(old, new)`, we can
replace one of the two neighbours of `old`:

* replace the predecessor `prev` of `old`, creating `prevPrev → new → old`;
* replace the successor `next` of `old`, creating `old → new → nextNext`.

Both moves are checked. The delta includes the node costs, because one selected node leaves and
one unselected node enters.

```
InterRouteIterator(path, C, selected):
    m ← len(path)

    for i ← 0 to m-1:
        old      ← path[i]
        prev     ← path[(i-1+m) mod m]
        prevPrev ← path[(i-2+m) mod m]
        next     ← path[(i+1) mod m]
        nextNext ← path[(i+2) mod m]

        for new in C[old]:
            if selected[new]:
                continue

            # Replace prev with new.
            Δprev ← cost[new] + dist[prevPrev][new] + dist[new][old]
                    − cost[prev] − dist[prevPrev][prev] − dist[prev][old]
            yield (replace position (i-1+m) mod m with new, Δprev)

            # Replace next with new.
            Δnext ← cost[new] + dist[old][new] + dist[new][nextNext]
                    − cost[next] − dist[old][next] − dist[next][nextNext]
            yield (replace position (i+1) mod m with new, Δnext)
```

`selected[new]` is a boolean array, so the membership test is O(1). A replacement can be
generated twice, once from each neighbour of the replaced position. The two yields have the same
delta, so the search removes the duplicate (section 2.4).

### 2.3 Intra-route moves through candidate edges

The intra-route move is a 2-opt exchange. For edges `(a, aNext)` and `(b, bNext)`, it removes
them, adds `(a, b)` and `(aNext, bNext)`, and reverses the segment between the removed edges.
No node enters or leaves, so node costs cancel and the delta contains only distances.

For a candidate edge `(old, new)` there are two such moves. If `i` is the position of `old` and
`j` is the position of `new`, they remove the edges **after** both nodes or the edges **before**
both nodes.

```
IntraRouteIterator(path, C, pos):
    m ← len(path)

    for i ← 0 to m-1:
        old ← path[i]

        for new in C[old]:
            j ← pos[new]
            if j = -1:
                continue                         # new is not selected
            if j = (i-1+m) mod m or j = (i+1) mod m:
                continue                         # the nodes are adjacent

            YieldTwoOpt(i, j)
            YieldTwoOpt((i-1+m) mod m, (j-1+m) mod m)

YieldTwoOpt(e1, e2):
    if e1 > e2:
        swap e1, e2
    if e2 - e1 < 2 or (e1 = 0 and e2 = m-1):
        return                                   # adjacent edges or wrap-around pair

    a     ← path[e1]
    aNext ← path[(e1+1) mod m]
    b     ← path[e2]
    bNext ← path[(e2+1) mod m]

    Δ ← −dist[a][aNext] − dist[b][bNext]
        + dist[a][b] + dist[aNext][bNext]

    yield (reverse segment e1+1 .. e2, Δ)
```

`pos[new]` is the position of `new`, or `-1` when it is not selected, so both the membership
test and the position lookup are O(1). The first call removes the edges after `old` and `new`,
the second removes the edges before them. The `e2 - e1 < 2` check skips adjacent edges, and
`e1 = 0 and e2 = m-1` skips the wrap-around pair `(v0, v1)` and `(v_{m-1}, v0)`, which share
node `v0` and cannot be used for a 2-opt move.

### 2.4 Steepest search with candidate lists

This is normal steepest descent, but the neighbourhood is the candidate set. One iteration checks
all candidate inter-route moves and all candidate 2-opt moves, keeps the best delta, and applies
it if it is negative. If nothing improves, the search stops.

The tour is kept in a fixed-size array. Two arrays are maintained with it:

* `selected[v]` — whether `v` is in the tour;
* `pos[v]` — its position in the tour, or `-1` when it is not selected.

They are updated after every accepted move, so the iterators never scan the path. Duplicate moves
are removed with generation stamps.

```
SteepestCandidates(path, C):
    x ← copy(path)
    m ← len(x)

    for v ← 0 to N-1:
        selected[v] ← false
        pos[v]      ← -1
    for i ← 0 to m-1:
        selected[x[i]] ← true
        pos[x[i]]      ← i

    generation ← 0
    repeat:
        generation ← generation + 1
        bestDelta  ← 0
        bestMove   ← none

        for each (q, new, Δ) from InterRouteIterator(x, C, selected):
            if seenInter[q][new] = generation:
                continue
            seenInter[q][new] ← generation
            if Δ < bestDelta:
                bestDelta ← Δ
                bestMove  ← (replace x[q] with new)

        for each (e1, e2, Δ) from IntraRouteIterator(x, C, pos):
            if seenIntra[e1][e2] = generation:
                continue
            seenIntra[e1][e2] ← generation
            if Δ < bestDelta:
                bestDelta ← Δ
                bestMove  ← (reverse x[e1+1 .. e2])

        if bestDelta ≥ 0:
            return x

        apply bestMove
        update selected and pos
```

The comparison is strict, and `bestDelta` starts at zero, so only improving moves are accepted.
Ties go to the first move in the fixed iteration order. Candidate lists are not updated during
the search, and no delta is reused from a previous iteration.

### 2.5 The full-neighbourhood reference

The reference is the steepest edge-exchange search from Task 3 with a random start. It uses the
same deltas but checks every move.

```
SteepestFull(path):
    x ← copy(path)
    m ← len(x)

    repeat:
        bestDelta ← 0
        bestMove  ← none

        used[v] ← false for all v
        for v in x:
            used[v] ← true
        unused ← [v | used[v] = false]

        # All inter-route exchanges.
        for i ← 0 to m-1:
            old  ← x[i]
            prev ← x[(i-1+m) mod m]
            next ← x[(i+1) mod m]

            for new in unused:
                Δ ← cost[new] + dist[prev][new] + dist[new][next]
                    − cost[old] − dist[prev][old] − dist[old][next]
                if Δ < bestDelta:
                    bestDelta ← Δ
                    bestMove  ← (replace x[i] with new)

        # All 2-opt exchanges.
        for i ← 0 to m-1:
            for j ← i+2 to m-1:
                if i = 0 and j = m-1:
                    continue

                a     ← x[i]
                aNext ← x[(i+1) mod m]
                b     ← x[j]
                bNext ← x[(j+1) mod m]

                Δ ← dist[a][b] + dist[aNext][bNext]
                    − dist[a][aNext] − dist[b][bNext]
                if Δ < bestDelta:
                    bestDelta ← Δ
                    bestMove  ← (reverse x[i+1 .. j])

        if bestDelta ≥ 0:
            return x

        apply bestMove
```

One full iteration checks `M(N−M) = 10,000` inter-route moves and `M(M−3)/2 = 4,850` 2-opt
moves, about 14,850 in total. A candidate iteration yields at most `2Mk` moves of each type
(`4Mk` in total) before skips: for `k = 10` that is 2,000 per type and 4,000 in total, and for
`k = 13` 2,600 per type and 5,200 in total. The real number is lower because many candidates
are already selected and duplicates are removed.

### 2.6 What the restriction does, and what it does not do

Every candidate move also exists in the full neighbourhood, but not the other way round. A good
move is invisible to the candidate search if it creates no candidate edge. This is the expected
trade-off: fewer evaluations, but a risk of stopping in a worse local optimum.

The list is built from `dist + cost`. For an inter-route move that replaces `prev` by `new`, the
true node-cost change is `cost[new] − cost[prev]`, but the candidate score only sees `cost[new]`.
For a 2-opt move the node cost cancels completely. So the list is a heuristic ordering, not an
exact ranking of the moves by their delta. Section 4.4 shows what this does on the two instances.

The lists are also directed. Because `dist + cost` is not symmetric, a pair can be a candidate
edge from one side only. The intra-route iterator still sees both endpoints because both are
selected and both are visited. The inter-route iterator only reads the selected node's list, so a
swap whose candidate edge appears only in the unselected node's list is not generated. With a
distance-only list this would not happen; it is a consequence of the cost-aware criterion.

## 3. Experiment setup

* `N = 200`, `M = 100`, two instances.
* **200 runs per method per instance.** Run `r` starts from `RandomSolution` with seed `r`, the
  same starts as in the Task 3 local search.
* 4 methods × 2 instances × 200 runs = **1600 solutions**: `RandomSolution` (baseline),
  full neighbourhood, 10 candidates, 13 candidates.
* Candidate lists are built once per instance before the timed loop, so their construction is
  not charged to the first run.
* Times are measured around the strategy call only. They come from one run of the program and
  vary between runs; only comparisons inside the same run are meaningful.
* The previous methods are registered together with the Task 4 methods in one extended
  benchmark run, so the timings in section 4.2 are directly comparable. Their objectives,
  components and tours are identical to Report 3's 200-run experiment.
* The extended timing table was produced with a temporary harness that registers the 16 Report 3
  strategies (the four Task 1 constructions, the four Task 2 constructions and the eight from
  `task3.All()`) plus the two Task 4 candidate strategies in one `Benchmark` call. The
  repository's default `go run .` runs the Task 4 subset; the algorithms are unchanged.
* The sweep in section 4.5 uses the same 200 runs and adds `k = 5, 8, 20, 30, 50`.

To reproduce:

```sh
go run . -json results_task4.json
go run . -candidates 5,8,10,13,20,30,50 -json results_task4_sweep.json
python3 tools/check_with_excel.py results_task4.json \
    --out report/excel_check_task4.json
python3 tools/make_report_task4.py results_task4.json \
    --sweep results_task4_sweep.json \
    --excel-check report/excel_check_task4.json
```

`make_report_task4.py` produces the Task 4 tables and figures only. The extended timing table
in section 4.2 was produced by the temporary all-method harness described above; the repository's
default `go run .` runs the Task 4 subset. The algorithms and the solutions are unchanged.

## 4. Results

### 4.1 Objective values

| Method | TSPA.csv av (min–max) | TSPB.csv av (min–max) |
|---|---:|---:|
| RandomSolution | 261,353.1 (238,111 – 291,409) | 207,261.0 (186,229 – 225,881) |
| RandomStartSteepestEdges | 76,168.9 (73,580 – 80,490) | 52,662.4 (50,328 – 55,056) |
| RandomStartSteepestEdges10Candidates | 80,218.4 (75,324 – 85,156) | 52,723.0 (50,441 – 55,895) |
| RandomStartSteepestEdges13Candidates | 78,345.8 (75,325 – 83,113) | 52,600.8 (50,394 – 55,062) |
| ExtendEnd | 87,476.6 (84,362 – 90,075) | 59,397.3 (56,007 – 63,166) |
| ExtendAny | 74,347.0 (72,328 – 77,022) | 59,475.5 (52,915 – 66,164) |
| GreedyCycle | 73,793.4 (72,590 – 75,281) | 59,649.9 (55,655 – 63,734) |
| RegretOnlyExtendAny | 122,860.7 (111,527 – 133,181) | 74,763.4 (69,032 – 80,767) |
| RegretOnlyGreedyCycle | 123,320.3 (112,410 – 135,431) | 74,458.2 (69,474 – 79,437) |
| RegretWeightedExtendAny | 75,561.2 (72,419 – 78,504) | 55,698.3 (50,485 – 61,406) |
| RegretWeightedGreedyCycle | 73,335.3 (72,011 – 76,027) | 57,905.2 (52,860 – 63,406) |
| RandomStartSteepestNodes | 89,797.9 (83,476 – 99,440) | 66,139.6 (60,091 – 72,562) |
| GreedyStartSteepestNodes | 72,998.0 (71,994 – 75,688) | 57,536.9 (52,669 – 62,998) |
| GreedyStartSteepestEdges | 72,978.7 (71,994 – 75,204) | 57,145.6 (52,512 – 62,301) |
| RandomStartGreedyNodes | 87,245.0 (80,056 – 94,234) | 62,985.8 (58,118 – 69,313) |
| GreedyStartGreedyNodes | 72,997.2 (71,996 – 75,688) | 57,538.7 (52,510 – 62,933) |
| RandomStartGreedyEdges | 75,524.3 (73,376 – 78,145) | 51,869.9 (49,932 – 54,790) |
| GreedyStartGreedyEdges | 72,947.1 (71,996 – 75,204) | 57,213.5 (52,545 – 62,475) |

The first four rows are the Task 4 methods. The remaining rows are the previous methods from
the same extended run, included for context. The best previous averages are 72,947.1 on TSPA
(`GreedyStartGreedyEdges`) and 51,869.9 on TSPB (`RandomStartGreedyEdges`).

`RandomSolution` is only a baseline. Its average is 261,353 on TSPA and 207,261 on TSPB,
about 3.4 and 3.9 times the full-neighbourhood averages. It is kept in the tables to show the
scale but is not used in the plots.

The two instances behave differently:

* **TSPA.** 10 candidates are 4,049.5 worse on average (+5.32%) than the full search, 13
  candidates are 2,176.9 worse (+2.86%). The 10-candidate search loses all 200 paired runs; the
  13-candidate search wins 11 and loses 189.
* **TSPB.** 10 candidates are only 60.5 worse (+0.11%) than the full search, and 13 candidates
  are 61.6 *better* (−0.12%). The paired results are almost even: 96/103/1 and 107/90/3.

The best single solutions show the same. On TSPA the full search reaches 73,580, while the
candidate searches stop at 75,324 and 75,325. On TSPB the best solutions are 50,328, 50,441 and
50,394, so the candidate searches are within 113 units of the full search.

### 4.2 Running times

| Method | TSPA.csv av (min–max) | TSPB.csv av (min–max) |
|---|---:|---:|
| RandomSolution | 0.01 ms (0.01 – 0.55) | 0.01 ms (0.01 – 0.02) |
| RandomStartSteepestEdges | 8.31 ms (6.99 – 10.07) | 8.34 ms (7.22 – 10.35) |
| RandomStartSteepestEdges10Candidates | 1.54 ms (1.33 – 4.02) | 1.58 ms (1.35 – 3.42) |
| RandomStartSteepestEdges13Candidates | 2.06 ms (1.75 – 3.81) | 2.05 ms (1.74 – 3.74) |
| ExtendEnd | 0.02 ms (0.01 – 0.05) | 0.02 ms (0.01 – 0.03) |
| ExtendAny | 2.01 ms (1.92 – 2.99) | 2.03 ms (1.93 – 2.84) |
| GreedyCycle | 2.06 ms (1.94 – 3.04) | 2.01 ms (1.93 – 2.96) |
| RegretOnlyExtendAny | 2.28 ms (2.16 – 3.27) | 2.27 ms (2.15 – 3.39) |
| RegretOnlyGreedyCycle | 2.21 ms (1.97 – 3.43) | 2.18 ms (1.95 – 3.21) |
| RegretWeightedExtendAny | 2.29 ms (2.13 – 3.32) | 2.29 ms (2.08 – 3.28) |
| RegretWeightedGreedyCycle | 2.29 ms (2.15 – 3.57) | 2.26 ms (2.09 – 3.27) |
| RandomStartSteepestNodes | 11.57 ms (8.94 – 22.75) | 11.40 ms (9.50 – 15.60) |
| GreedyStartSteepestNodes | 2.59 ms (2.25 – 3.94) | 2.72 ms (2.28 – 3.91) |
| GreedyStartSteepestEdges | 2.51 ms (2.23 – 3.71) | 2.87 ms (2.34 – 3.85) |
| RandomStartGreedyNodes | 5.53 ms (2.94 – 8.07) | 5.21 ms (2.82 – 10.79) |
| GreedyStartGreedyNodes | 2.50 ms (2.26 – 3.73) | 2.49 ms (2.24 – 3.58) |
| RandomStartGreedyEdges | 4.16 ms (2.16 – 7.11) | 3.89 ms (2.01 – 6.89) |
| GreedyStartGreedyEdges | 2.48 ms (2.24 – 3.75) | 2.55 ms (2.25 – 3.37) |

The first four rows are the Task 4 methods; the remaining rows are the previous methods.

The candidate searches are about **5.3–5.4 times faster** than the full search with `k = 10`
and about **4.0 times faster** with `k = 13`. The first run of a method is a bit slower (for
example 4.02 ms against a median of 1.51 ms for `k = 10` on TSPA), probably because the first
run pays for allocation and cache warm-up. `RandomSolution` is almost free (0.01 ms) and is
only the baseline.

The speedup is not only from checking fewer moves. The iterators use O(1) `selected` and `pos`
arrays and generation stamps for duplicates. A naive version that scans the path for every
candidate, or evaluates the same move several times, can lose the whole advantage.

All times in table 4.2 come from one extended run that registers all 18 methods — the 16 from
Report 3 plus the two Task 4 candidate strategies — in the same process, so they are directly
comparable. Only times inside the extended run should be compared.

### 4.3 Paired comparisons

Every pair uses the same seed, so the difference is caused by the neighbourhood alone.
Only the Task 4 methods are paired here: they share the same random starts and differ only in
the candidate restriction. The previous methods are in the extended run, but they differ in more
than the neighbourhood (and some use greedy starts), so they are not included in this paired
comparison.

| Instance | Comparison | Mean difference | Better / worse / tie (of 200) |
|---|---|---:|---:|
| TSPA.csv | 10 candidates − full | +4,049.4 | 0 / 200 / 0 |
| TSPA.csv | 13 candidates − full | +2,176.9 | 11 / 189 / 0 |
| TSPA.csv | 13 candidates − 10 candidates | −1,872.5 | 177 / 23 / 0 |
| TSPB.csv | 10 candidates − full | +60.5 | 96 / 103 / 1 |
| TSPB.csv | 13 candidates − full | −61.6 | 107 / 90 / 3 |
| TSPB.csv | 13 candidates − 10 candidates | −122.2 | 104 / 93 / 3 |

On TSPA the restriction is a clear loss against the full search. On TSPB the candidate search
is not simply worse: it reaches a different local optimum and wins 96–107 of 200 starts against
the full search. A restricted neighbourhood can change the whole sequence of accepted moves, not
only the first step.

### 4.4 Where the objective goes

Component averages are rounded, so they may not add exactly to the objective. The first four
rows for each instance are the Task 4 methods; the remaining rows are the previous methods from
the same extended run, included for context.

| Instance | Method | Avg tour length | Avg node cost | Avg objective |
|---|---|---:|---:|---:|
| TSPA.csv | RandomSolution | 164,360.7 | 96,992.4 | 261,353.1 |
| TSPA.csv | RandomStartSteepestEdges | 23,040.2 | 53,128.7 | 76,168.9 |
| TSPA.csv | RandomStartSteepestEdges10Candidates | 23,663.5 | 56,554.8 | 80,218.4 |
| TSPA.csv | RandomStartSteepestEdges13Candidates | 23,358.0 | 54,987.9 | 78,345.8 |
| TSPA.csv | ExtendEnd | 33,641.8 | 53,834.8 | 87,476.6 |
| TSPA.csv | ExtendAny | 23,071.4 | 51,275.6 | 74,347.0 |
| TSPA.csv | GreedyCycle | 22,304.4 | 51,489.0 | 73,793.4 |
| TSPA.csv | RegretOnlyExtendAny | 22,557.7 | 100,303.0 | 122,860.7 |
| TSPA.csv | RegretOnlyGreedyCycle | 21,215.2 | 102,105.1 | 123,320.3 |
| TSPA.csv | RegretWeightedExtendAny | 23,352.2 | 52,209.0 | 75,561.2 |
| TSPA.csv | RegretWeightedGreedyCycle | 21,184.7 | 52,150.6 | 73,335.3 |
| TSPA.csv | RandomStartSteepestNodes | 37,458.6 | 52,339.2 | 89,797.9 |
| TSPA.csv | GreedyStartSteepestNodes | 21,139.4 | 51,858.6 | 72,998.0 |
| TSPA.csv | GreedyStartSteepestEdges | 21,121.6 | 51,857.1 | 72,978.7 |
| TSPA.csv | RandomStartGreedyNodes | 34,575.2 | 52,669.9 | 87,245.0 |
| TSPA.csv | GreedyStartGreedyNodes | 21,135.7 | 51,861.6 | 72,997.2 |
| TSPA.csv | RandomStartGreedyEdges | 22,737.3 | 52,787.0 | 75,524.3 |
| TSPA.csv | GreedyStartGreedyEdges | 21,092.9 | 51,854.2 | 72,947.1 |
| TSPB.csv | RandomSolution | 155,274.3 | 51,986.7 | 207,261.0 |
| TSPB.csv | RandomStartSteepestEdges | 20,847.6 | 31,814.8 | 52,662.4 |
| TSPB.csv | RandomStartSteepestEdges10Candidates | 21,074.1 | 31,648.9 | 52,723.0 |
| TSPB.csv | RandomStartSteepestEdges13Candidates | 20,884.4 | 31,716.4 | 52,600.8 |
| TSPB.csv | ExtendEnd | 26,882.0 | 32,515.3 | 59,397.3 |
| TSPB.csv | ExtendAny | 18,678.0 | 40,797.5 | 59,475.5 |
| TSPB.csv | GreedyCycle | 18,014.7 | 41,635.2 | 59,649.9 |
| TSPB.csv | RegretOnlyExtendAny | 22,584.5 | 52,178.9 | 74,763.4 |
| TSPB.csv | RegretOnlyGreedyCycle | 21,110.4 | 53,347.9 | 74,458.2 |
| TSPB.csv | RegretWeightedExtendAny | 20,849.9 | 34,848.4 | 55,698.3 |
| TSPB.csv | RegretWeightedGreedyCycle | 17,025.1 | 40,880.1 | 57,905.2 |
| TSPB.csv | RandomStartSteepestNodes | 35,185.3 | 30,954.3 | 66,139.6 |
| TSPB.csv | GreedyStartSteepestNodes | 17,100.1 | 40,436.8 | 57,536.9 |
| TSPB.csv | GreedyStartSteepestEdges | 16,704.9 | 40,440.7 | 57,145.6 |
| TSPB.csv | RandomStartGreedyNodes | 31,445.2 | 31,540.7 | 62,985.8 |
| TSPB.csv | GreedyStartGreedyNodes | 17,101.8 | 40,437.0 | 57,538.7 |
| TSPB.csv | RandomStartGreedyEdges | 20,884.6 | 30,985.3 | 51,869.9 |
| TSPB.csv | GreedyStartGreedyEdges | 16,772.7 | 40,440.8 | 57,213.5 |

The best previous averages are 72,947.1 on TSPA (`GreedyStartGreedyEdges`) and 51,869.9 on
TSPB (`RandomStartGreedyEdges`); the candidate searches are a faster restricted version of
the full steepest search, not the best overall method.

The random baseline is far behind, mostly because its tour is much longer: 164,361 against
23,040 on TSPA and 155,274 against 20,848 on TSPB.

On TSPA the candidate searches lose on both parts compared with the full search. With `k = 10`
the tour is 623 longer and the nodes cost 3,426 more; with `k = 13` the tour is 318 longer and
the nodes cost 1,859 more. The restricted search ends in a worse basin, and the cost-aware list
does not make up for the missing moves.

On TSPB the parts move in opposite directions compared with the full search. With `k = 10` the
search saves 166 on nodes but pays 227 more for the tour (net +61). With `k = 13` it saves 98 on
nodes and pays 37 more for the tour (net −62). The list shifts the selection towards cheaper
nodes, and the tour grows only a little.

The instance's node costs average 971 (sd 568) on TSPA and 519 (sd 287) on TSPB. The cost term
therefore has a larger effect on the candidate ordering on TSPA, while the 2-opt delta is
cost-blind and the inter-route criterion ignores the cost of the node that leaves. This is the
likely reason for the TSPA deterioration. The experiment does not separate these effects.

### 4.5 Candidate-list size

The choice `k = 13` is not special. The table below runs the same search for several list sizes;
the full-neighbourhood row is the reference.

| k | TSPA avg objective | TSPB avg objective | TSPA ms/run | TSPB ms/run |
|---:|---:|---:|---:|---:|
| full | 76,168.9 | 52,662.4 | 8.01 | 7.98 |
| 5 | 86,525.6 | 54,185.8 | 0.80 | 0.74 |
| 8 | 82,638.2 | 52,961.8 | 1.21 | 1.21 |
| 10 | 80,218.4 | 52,723.0 | 1.49 | 1.51 |
| 13 | 78,345.8 | 52,600.8 | 1.98 | 1.99 |
| 20 | 76,432.7 | 52,668.6 | 3.14 | 3.06 |
| 30 | 76,093.9 | 52,688.9 | 4.73 | 4.60 |
| 50 | 76,150.6 | 52,668.0 | 7.99 | 7.76 |

Timings in this table come from the sweep run and differ slightly from section 4.2 because
timing varies; the objectives are deterministic.

![Candidate-list size — quality and time](report/figures_task4/candidate_sweep.png)

* **TSPA needs about 20–30 candidates.** At `k = 20` the average is 0.35% above the full search
  and the search is 2.5 times faster; at `k = 30` it is 0.10% below the full search and 1.7 times
  faster. The curve is still improving at 13.
* **TSPB is already fine with 10 candidates.** At `k = 10` the average is 0.11% above the full
  search and the search is 5.3 times faster. Larger lists change the average by less than 0.25%
  and only cost time.

So 13 is an awkward middle point: slower than 10 without closing the TSPA gap, and unnecessary
on TSPB. For speed, 10 is the better choice on TSPB; for quality on TSPA, 20–30 is the useful
range.

### 4.6 Pictures of the best solutions

Node colour is the node cost (light blue cheap, dark blue expensive) and the dark line is the
closed tour. The first panel is the full-neighbourhood reference, the other two are the candidate
searches. `RandomSolution` is left out of the plots because its objective is about three times
larger and would make the panels hard to read; it is in the tables. The figures show only the
Task 4 methods; the previous methods are in the tables.

![Best solution per method — TSPA.csv](report/figures_task4/best_solutions_TSPA.png)

![Best solution per method — TSPB.csv](report/figures_task4/best_solutions_TSPB.png)

![Objective ranges — TSPA.csv](report/figures_task4/objective_ranges_TSPA.png)

![Objective ranges — TSPB.csv](report/figures_task4/objective_ranges_TSPB.png)

On TSPA the candidate panels have more long edges and the selected nodes are on average more
expensive, which matches section 4.4. On TSPB the three panels look close.

### 4.7 Best runs

The first four rows for each instance are the Task 4 methods. The remaining rows are the previous
methods from the same extended run, included for context.

| Instance | Method | Best objective | Run | Tour length | Node cost |
|---|---|---:|---:|---:|---:|
| TSPA.csv | RandomSolution | 238,111 | 154 | 146,623 | 91,488 |
| TSPA.csv | RandomStartSteepestEdges | 73,580 | 171 | 21,726 | 51,854 |
| TSPA.csv | RandomStartSteepestEdges10Candidates | 75,324 | 70 | 22,896 | 52,428 |
| TSPA.csv | RandomStartSteepestEdges13Candidates | 75,325 | 15 | 22,694 | 52,631 |
| TSPA.csv | ExtendEnd | 84,362 | 54 | 31,319 | 53,043 |
| TSPA.csv | ExtendAny | 72,328 | 11 | 21,816 | 50,512 |
| TSPA.csv | GreedyCycle | 72,590 | 31 | 21,570 | 51,020 |
| TSPA.csv | RegretOnlyExtendAny | 111,527 | 85 | 19,974 | 91,553 |
| TSPA.csv | RegretOnlyGreedyCycle | 112,410 | 31 | 21,509 | 90,901 |
| TSPA.csv | RegretWeightedExtendAny | 72,419 | 69 | 20,127 | 52,292 |
| TSPA.csv | RegretWeightedGreedyCycle | 72,011 | 31 | 20,248 | 51,763 |
| TSPA.csv | RandomStartSteepestNodes | 83,476 | 94 | 32,174 | 51,302 |
| TSPA.csv | GreedyStartSteepestNodes | 71,994 | 126 | 20,157 | 51,837 |
| TSPA.csv | GreedyStartSteepestEdges | 71,994 | 126 | 20,157 | 51,837 |
| TSPA.csv | RandomStartGreedyNodes | 80,056 | 63 | 27,207 | 52,849 |
| TSPA.csv | GreedyStartGreedyNodes | 71,996 | 31 | 20,233 | 51,763 |
| TSPA.csv | RandomStartGreedyEdges | 73,376 | 19 | 21,634 | 51,742 |
| TSPA.csv | GreedyStartGreedyEdges | 71,996 | 31 | 20,233 | 51,763 |
| TSPB.csv | RandomSolution | 186,229 | 165 | 132,442 | 53,787 |
| TSPB.csv | RandomStartSteepestEdges | 50,328 | 101 | 20,812 | 29,516 |
| TSPB.csv | RandomStartSteepestEdges10Candidates | 50,441 | 149 | 21,229 | 29,212 |
| TSPB.csv | RandomStartSteepestEdges13Candidates | 50,394 | 149 | 20,711 | 29,683 |
| TSPB.csv | ExtendEnd | 56,007 | 32 | 24,632 | 31,375 |
| TSPB.csv | ExtendAny | 52,915 | 50 | 20,068 | 32,847 |
| TSPB.csv | GreedyCycle | 55,655 | 174 | 19,634 | 36,021 |
| TSPB.csv | RegretOnlyExtendAny | 69,032 | 185 | 21,267 | 47,765 |
| TSPB.csv | RegretOnlyGreedyCycle | 69,474 | 169 | 21,486 | 47,988 |
| TSPB.csv | RegretWeightedExtendAny | 50,485 | 192 | 18,835 | 31,650 |
| TSPB.csv | RegretWeightedGreedyCycle | 52,860 | 5 | 16,480 | 36,380 |
| TSPB.csv | RandomStartSteepestNodes | 60,091 | 191 | 29,813 | 30,278 |
| TSPB.csv | GreedyStartSteepestNodes | 52,669 | 5 | 16,589 | 36,080 |
| TSPB.csv | GreedyStartSteepestEdges | 52,512 | 17 | 17,298 | 35,214 |
| TSPB.csv | RandomStartGreedyNodes | 58,118 | 97 | 24,165 | 33,953 |
| TSPB.csv | GreedyStartGreedyNodes | 52,510 | 175 | 17,010 | 35,500 |
| TSPB.csv | RandomStartGreedyEdges | 49,932 | 105 | 18,676 | 31,256 |
| TSPB.csv | GreedyStartGreedyEdges | 52,545 | 175 | 16,391 | 36,154 |

For context, the best previous run is 71,994 on TSPA (`GreedyStartSteepestNodes` and
`GreedyStartSteepestEdges`) and 49,932 on TSPB (`RandomStartGreedyEdges`).

The Task 4 tours are in `results_task4.json`; the previous methods' tours are in Report 3's
`results.json`. `go run .` prints the best tour of every Task 4 method, and the run column
identifies the random start.

## 5. Solution checker

All 1600 runs were checked with the course-provided **`Solution checker.xlsx`** through
`tools/check_with_excel.py`. The workbook writes the tour into `F3:F102`, uses `LOOKUP` to fetch
the coordinates and costs, computes the rounded edges in `J3:J102` (`F103 = F3` closes the cycle),
and reports the node cost in `I2`, the tour length in `J2` and the objective in `K2`. The script
only checks that the tour has exactly `M = 100` distinct in-range ids and reads those cells back
from a temporary copy, so the provided file is not modified.

| Instance | Method | Runs checked | Feasible | Objective matches | Components match |
|---|---|---:|---:|---:|---:|
| TSPA.csv | RandomSolution | 200 | 200/200 | 200/200 | 200/200 |
| TSPA.csv | RandomStartSteepestEdges | 200 | 200/200 | 200/200 | 200/200 |
| TSPA.csv | RandomStartSteepestEdges10Candidates | 200 | 200/200 | 200/200 | 200/200 |
| TSPA.csv | RandomStartSteepestEdges13Candidates | 200 | 200/200 | 200/200 | 200/200 |
| TSPB.csv | RandomSolution | 200 | 200/200 | 200/200 | 200/200 |
| TSPB.csv | RandomStartSteepestEdges | 200 | 200/200 | 200/200 | 200/200 |
| TSPB.csv | RandomStartSteepestEdges10Candidates | 200 | 200/200 | 200/200 | 200/200 |
| TSPB.csv | RandomStartSteepestEdges13Candidates | 200 | 200/200 | 200/200 | 200/200 |

**1600 runs checked, 0 failures.**

The extended run's 7200 solutions are identical to the solutions checked earlier: the 1600
Task 4 runs checked here and all 6400 Report 3 runs (the two baselines appear in both checks).
The previous methods were therefore not rechecked again.

## 6. Source code

| File | Contents |
|---|---|
| `solver/problem.go` | loading the instance, the distance matrix, `TourLength` / `NodesCost` / `Objective` |
| `solver/solver.go` | the `Strategy` interface, the optional `Preparer` interface and the benchmark loop |
| `task1/strategies.go` | `RandomSolution`, used for the starts and as the baseline |
| `task3/task3.go` | the full-neighbourhood steepest search |
| `task4/task4.go` | candidate lists, the candidate iterators and `improvePathSteepestWithCandidates` |
| `task4/strategies.go` | `CandidateSteepest`, which prepares the lists and runs the search |
| `main.go` | command line, the `-candidates` option, the table and the JSON export |
| `tools/check_with_excel.py` | drives `Solution checker.xlsx` |
| `tools/make_report_task4.py` | Task 4 figures and tables for this report |
| `tools/oeviz.py` | plotting helpers |

The candidate search is one type, `CandidateSteepest`, with the list size `K` as a field.
`Prepare` builds the lists before the timed loop and `Solve` runs the search. The `-candidates`
flag makes the list size an experiment parameter, which is how the sweep in section 4.5 is
produced without changing the code.

## 7. Conclusions

1. **Candidate moves give the expected speedup.** With `k = 10` the search is about 5.3–5.4 times
   faster than the full search in the same run, with `k = 13` about 4.0 times. Constant-time
   membership and position lookups and duplicate removal are needed; a naive implementation can
   lose the gain.
2. **The quality cost depends on the instance.** On TSPA 10 candidates are 5.32% worse than the
   full search and lose all 200 paired runs; 13 candidates are 2.86% worse and lose 189. On TSPB
   the differences are +0.11% and −0.12%, with almost even paired results.
3. **`k = 13` is not a special point.** TSPA needs about 20–30 candidates to recover the quality
   of the full search, while TSPB is already fine with 10. Thirteen is slower than 10 without
   closing the TSPA gap, and it is unnecessary on TSPB.
4. **A restricted neighbourhood is not simply worse.** On TSPB the candidate search wins 96–107
   of 200 paired runs against the full search. It reaches a different local optimum, which can be
   better than the one found by the full search.
5. **The candidate criterion has limits.** The list uses `dist + cost`, but the 2-opt delta is
   cost-blind and the inter-route criterion ignores the cost of the node that leaves. On TSPA,
   where costs are larger, the search loses on both tour length and node cost compared with the
   full search. A criterion made for each move type, or learned from earlier solutions, could
   help.
6. **What to keep.** Candidate lists are useful when speed matters, especially on TSPB-like
   instances with 10 candidates. On TSPA-like instances, use 20–30 candidates or the full search
   within this steepest random-start family. If the goal is the best objective regardless of
   method, the best methods from Report 3 are `GreedyStartGreedyEdges` on TSPA (72,947.1) and
   `RandomStartGreedyEdges` on TSPB (51,869.9). The next step is to put candidate lists inside a
   higher-level method such as MSLS or ILS.
