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

# 3-bit booleanization
x_train = [TMInput(vec([
    [x > 0 ? true : false for x in i];
    [x > 0.33 ? true : false for x in i];
    [x > 0.66 ? true : false for x in i];
])) for i in x_train]
x_test = [TMInput(vec([
    [x > 0 ? true : false for x in i];
    [x > 0.33 ? true : false for x in i];
    [x > 0.66 ? true : false for x in i];
])) for i in x_test]

const EPOCHS = 500
const CLAUSES = 128
const T = 8
const R = 0.89
const L = 16

# Training the TM model
tm = TMClassifier(CLAUSES, T, R, L=L, states_num=256, include_limit=128)
tm_best = train!(tm, x_train, y_train, x_test, y_test, EPOCHS, shuffle=true, verbose=1)
