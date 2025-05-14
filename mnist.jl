include("Tsetlin.jl")

try
    using MLDatasets: MNIST
catch LoadError
    import Pkg
    Pkg.add("MLDatasets")
end

using Printf: @printf
using MLDatasets: MNIST
using .Tsetlin: TMInput, TMClassifier, train!, predict, accuracy, unzip


x_train, y_train = unzip([MNIST(:train)...])
x_test, y_test = unzip([MNIST(:test)...])

function booleanize(x::AbstractArray{Float32}, thresholds::Number...)::TMInput
    return TMInput(vcat((x .> t for t in thresholds)...))
end

# Booleanizing input data (3 bits per pixel):
x_train = [booleanize(x, 0, 0.33, 0.66) for x in x_train]
x_test = [booleanize(x, 0, 0.33, 0.66) for x in x_test]

CLAUSES = 128
T = 4
S = 24
L = 12

# CLAUSES = 2048
# T = 32
# S = 30
# L = 12  # 10

EPOCHS = 500

# Training the TM model
tm = TMClassifier(CLAUSES, T, S, L=L, states_num=256, include_limit=200)
tm_best = train!(tm, x_train, y_train, x_test, y_test, EPOCHS, shuffle=true, verbose=1)
