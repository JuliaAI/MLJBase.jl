module TestStatic

using Test, MLJBase

## SIMPLE UNIVARIATE FUNCTION

mutable struct Scale <: MLJBase.Static
    scaling::Float64
end

function MLJBase.transform(s::Scale, _, X) 
    X isa AbstractVecOrMat && return X * s.scaling
    MLJBase.table(s.scaling * MLJBase.matrix(X), prototype=X)
end

function MLJBase.inverse_transform(s::Scale, _, X)
    X isa AbstractVecOrMat && return X / s.scaling
    MLJBase.table(MLJBase.matrix(X) / s.scaling, prototype=X)
end

s = Scale(2)
X = randn(2, 3)
Xt = MLJBase.table(X)

R = transform(s, nothing, X)
IR = inverse_transform(s, nothing, R)
@test IR ≈ X

R = transform(s, nothing, Xt)
IR = inverse_transform(s, nothing, R)
@test MLJBase.matrix(IR) ≈ X


## MULTIVARIATE FUNCTION

mutable struct PermuteArgs <: MLJBase.Static
    permutation::NTuple{N,Int} where N
end

MLJBase.transform(p::PermuteArgs, _, args...) =
    Tuple([args[i] for i in p.permutation])
MLJBase.inverse_transform(p::PermuteArgs, _, args...) =
    Tuple([args[i] for i in sortperm(p.permutation |> collect)])

p = PermuteArgs((2, 3, 1))
@test transform(p, nothing, 10, 20, 30) == (20, 30, 10)
@test inverse_transform(p, nothing, 20, 30, 10) == (10, 20, 30)

# no-op
fitresult, _, _ = MLJBase.fit(p, 1, (1, 2, 3))

@test transform(p, fitresult, 10, 20, 30) == (20, 30, 10)
@test inverse_transform(p, fitresult, 20, 30, 10) == (10, 20, 30)

end
true
