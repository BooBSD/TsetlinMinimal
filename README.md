# TsetlinMinimal
Minimal Tsetlin Machine implementation

How to run MNIST example
------------------------

0. Make sure that you have installed the latest version of the [Julia language](https://julialang.org/downloads/).
1. Run `julia --project=. -O3 -t 32 --gcthreads=32,1 mnist.jl` where `32` is the number of your logical CPU cores.
