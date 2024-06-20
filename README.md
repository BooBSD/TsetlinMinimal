# TsetlinMinimal
Minimal Tsetlin Machine implementation in just **200** lines of code.

How to run MNIST example
------------------------

0. Make sure that you have installed the latest version of the [Julia language](https://julialang.org/downloads/).
1. Run `julia --project=. -O3 -t 32 --gcthreads=32,1 mnist.jl` where `32` is the number of your logical CPU cores.
   Do not forget to uncomment the `@threads` macros in the code to enable multithreading for training and inference.
