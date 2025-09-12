module Tsetlin

using Dates
using Random
using Base.Threads
using Printf: @printf

unzip(a) = (getfield.(a, x) for x in fieldnames(eltype(a)))

mutable struct TATeam
    const include_limit::UInt8
    const state_min::UInt8
    const state_max::UInt8
    positive_clauses::Matrix{UInt8}
    negative_clauses::Matrix{UInt8}
    positive_included_literals::Vector{Vector{UInt16}}
    negative_included_literals::Vector{Vector{UInt16}}
    const clause_size::Int64

    function TATeam(clause_size::Int64, clauses_num::Int64, include_limit::Int64, state_min::Int64, state_max::Int64)
        positive_clauses = fill(UInt8(include_limit - 1), clause_size, floor(Int, clauses_num / 2))
        negative_clauses = fill(UInt8(include_limit - 1), clause_size, floor(Int, clauses_num / 2))
        positive_included_literals = [UInt16[] for _ in 1:floor(Int, clauses_num / 2)]
        negative_included_literals = [UInt16[] for _ in 1:floor(Int, clauses_num / 2)]
        return new(include_limit, state_min, state_max, positive_clauses, negative_clauses, positive_included_literals, negative_included_literals, clause_size)
    end
end

mutable struct TMClassifier
    clauses_num::Int64
    T::Int64
    S::Int64
    s::Int64
    L::Int64
    const include_limit::Int64
    const state_min::Int64
    const state_max::Int64
    const clauses::Dict{Any, TATeam}

    function TMClassifier(clauses_num::Int64, T::Int64, S::Int64; states_num::Int64=256, include_limit::Int64=128, L::Int64=16)
        return new(clauses_num, T, S, 0, L, include_limit, typemin(UInt8), states_num - 1, Dict())
    end
end

struct TMInput <: AbstractVector{Bool}
    x::BitVector

    function TMInput(x::AbstractArray{Bool}; negate::Bool=true)
        return new(vec(negate ? [x; [!_x for _x in x]] : x))
    end
end

Base.size(x::TMInput)::Tuple{Int64} = size(x.x)
Base.getindex(x::TMInput, i::Int)::Bool = x.x[i]

function initialize!(tm::TMClassifier, X::Vector{TMInput}, Y::Vector)
    tm.s = round(Int, length(first(X)) / tm.S)
    for cls in unique(Y)
        tm.clauses[cls] = TATeam(length(first(X)), tm.clauses_num, tm.include_limit, tm.state_min, tm.state_max)
    end
end

function check_clause(x::TMInput, literals::Vector{UInt16})::Bool
    @inbounds for l in literals
        if !x[l]
            return false
        end
    end
    return true
end

function vote(ta::TATeam, x::TMInput)::Tuple{Int64, Int64}
    pos = sum(check_clause(x, l) for l in ta.positive_included_literals)
    neg = sum(check_clause(x, l) for l in ta.negative_included_literals)
    return pos, neg
end

function feedback!(tm::TMClassifier, ta::TATeam, x::TMInput, clauses1::Matrix{UInt8}, clauses2::Matrix{UInt8}, literals1::Vector{Vector{UInt16}}, literals2::Vector{Vector{UInt16}}, positive::Bool)
    v::Int64 = clamp(-(vote(ta, x)...), -tm.T, tm.T)
    update::Float64 = (positive ? (tm.T - v) : (tm.T + v)) / (tm.T * 2)

    # Feedback 1
    @inbounds for (j, c) in enumerate(eachcol(clauses1))
        if rand() < update
            if check_clause(x, literals1[j])
                if length(literals1[j]) <= tm.L
                    @inbounds for i = 1:ta.clause_size
                        if (x[i] == true) && (c[i] < ta.state_max)
                            c[i] += one(UInt8)
                        end
                    end
                end
                @inbounds for i = 1:ta.clause_size
                    if (x[i] == false) && (c[i] < ta.include_limit) && (c[i] > ta.state_min)
                        c[i] -= one(UInt8)
                    end
                end
            else
                @inbounds for _ in 1:tm.s
                    i = rand(1:ta.clause_size)  # Here's one random only.
                    if c[i] > ta.state_min
                        c[i] -= one(UInt8)
                    end
                end
            end
            literals1[j] = [@inbounds i for i = 1:ta.clause_size if c[i] >= ta.include_limit]
        end
    end
    # Feedback 2
    @inbounds for (j, c) in enumerate(eachcol(clauses2))
        if rand() < update
            if check_clause(x, literals2[j])
                @inbounds for i = 1:ta.clause_size
                    if (x[i] == false) && (c[i] < ta.include_limit)
                        c[i] += one(UInt8)
                    end
                end
                literals2[j] = [@inbounds i for i = 1:ta.clause_size if c[i] >= ta.include_limit]
            end
        end
    end
end

function predict(tm::TMClassifier, x::TMInput)::Any
    best_vote::Int64 = typemin(Int64)
    best_cls::Any = nothing
    @inbounds for (cls, ta) in tm.clauses
        v::Int64 = -(vote(ta, x)...)
        if v > best_vote
            best_vote = v
            best_cls = cls
        end
    end
    return best_cls
end

function predict(tm::TMClassifier, X::Vector{TMInput})::Vector
    predicted::Vector = Vector{eltype(first(keys(tm.clauses)))}(undef, length(X))  # Predefine vector for @threads access
    @threads for i in eachindex(X)
        predicted[i] = predict(tm, X[i])
    end
    return predicted
end

function train!(tm::TMClassifier, x::TMInput, y::Any; shuffle::Bool=true)
    classes = shuffle ? Random.shuffle(collect(keys(tm.clauses))) : keys(tm.clauses)
    for cls in classes
        if cls != y
            feedback!(tm, tm.clauses[y], x, tm.clauses[y].positive_clauses, tm.clauses[y].negative_clauses, tm.clauses[y].positive_included_literals, tm.clauses[y].negative_included_literals, true)
            feedback!(tm, tm.clauses[cls], x, tm.clauses[cls].negative_clauses, tm.clauses[cls].positive_clauses, tm.clauses[cls].negative_included_literals, tm.clauses[cls].positive_included_literals, false)
        end
    end
end

function train!(tm::TMClassifier, X::Vector{TMInput}, Y::Vector; shuffle::Bool=true)
    if length(tm.clauses) == 0
        initialize!(tm, X, Y)
    end
    if shuffle
        X, Y = unzip(Random.shuffle(collect(zip(X, Y))))
    end
    @threads for i in eachindex(Y)
        train!(tm, X[i], Y[i], shuffle=shuffle)
    end
end

function accuracy(predicted::Vector, Y::Vector)::Float64
    @assert eltype(predicted) == eltype(Y)
    @assert length(predicted) == length(Y)
    return sum(@inbounds 1 for (p, y) in zip(predicted, Y) if p == y; init=0) / length(Y)
end

function train!(tm::TMClassifier, x_train::Vector, y_train::Vector, x_test::Vector, y_test::Vector, epochs::Int64; shuffle::Bool=true, verbose::Int=1)::TMClassifier
    if length(tm.clauses) == 0
        initialize!(tm, x_train, y_train)
    end
    if verbose > 0
        println("\nRunning in $(nthreads()) threads.")
        println("Clauses: $(tm.clauses_num), T: $(tm.T), S: $(tm.S) (s: $(tm.s)), L: $(tm.L), states_num: $(tm.state_max + 1), include_limit: $(tm.include_limit).")
        println("Classes: $(length(tm.clauses)). Input vector size: $(length(x_train[1])) bits. Training dataset size: $(length(y_train)). Testing dataset size: $(length(y_test)).")
        println("Accuracy over $(epochs) epochs:\n")
    end
    best_tm = (0.0, nothing)
    all_time = @elapsed begin
        for i in 1:epochs
            training_time = @elapsed train!(tm, x_train, y_train, shuffle=shuffle)
            testing_time = @elapsed begin
                acc = accuracy(predict(tm, x_test), y_test)
            end
            if acc >= first(best_tm)
                best_tm = (acc, deepcopy(tm))
            end
            if verbose > 0
                @printf("#%s  Accuracy: %.2f%%  Best: %.2f%%  Training: %.3fs  Testing: %.3fs\n", i, acc * 100, best_tm[1] * 100, training_time, testing_time)
            end
        end
    end
    if verbose > 0
        elapsed = Time(0) + Second(floor(Int, all_time))
        @printf("\n%s epochs done in %s. Best accuracy: %.2f%%.\n", epochs, elapsed, best_tm[1] * 100)
        println("Clauses: $(tm.clauses_num), T: $(tm.T), S: $(tm.S) (s: $(tm.s)), L: $(tm.L), states_num: $(tm.state_max + 1), include_limit: $(tm.include_limit).")
        println("Classes: $(length(tm.clauses)). Input vector size: $(length(x_train[1])) bits. Training dataset size: $(length(y_train)). Testing dataset size: $(length(y_test)).\n")
    end
    return best_tm[2]
end

end # module
