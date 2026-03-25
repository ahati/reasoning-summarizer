// Package llama provides CGo bindings to llama.cpp.
// Before building, run `go generate` to fetch and build llama.cpp:
//
//	CGO_ENABLED=1 go generate github.com/ahati/reasoning-summarizer/llama
//
// This will clone llama.cpp and build the static libraries needed for CGo.
package llama

//go:generate bash build-llama.sh
