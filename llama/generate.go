// Package llama provides CGo bindings to llama.cpp.
// llama.cpp is included as a git submodule.
// Run `git submodule update --init --recursive` to fetch it.
// Run `go generate ./...` to build the static libraries.
package llama

//go:generate bash build-llama.sh
