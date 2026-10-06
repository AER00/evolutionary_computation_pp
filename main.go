package main

import (
	"math/rand"
	"time"

	"OE/task1"
)

func main() {
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	task1.SolveInstance("TSPA.csv", 60*time.Second, rng)
	task1.SolveInstance("TSPB.csv", 60*time.Second, rng)
}
