module TestStatic

using Test, MLJBase

## SIMPLE UNIVARIATE FUNCTION

mutable struct Scale <: MLJBase.Static
    scaling::Float64
end

(s::Scale)(X) = s.scaling*X
MLJBase.inv(s::Scale, X) = X/s.scaling

s = Scale(2)
X = rand(2,3)

# no-op:
fitresult, _, _ = MLJBase.fit(s, 1, X)

@test transform(s, 1, X) ≈ 2*X
@test inverse_transform(s, 1, X) ≈ 0.5*X


## MULTIVARIATE FUNCTION

mutable struct PermuteArgs <: MLJBase.Static
    permutation::NTuple{<:Any,Int}
end

(p::PermuteArgs)(args...) =
    Tuple([args[i] for i in p.permutation])

p = PermuteArgs((2,3,1))
@test p(10, 20, 30) == (20, 30, 10)

MLJBase.inv(p::PermuteArgs, args...) =
    Tuple([args[i] for i in sortperm(p.permutation |> collect)])

@test inv(p, p(10, 20, 30)...) == (10, 20, 30)

# no-op
fitresult, _, _ = MLJBase.fit(p, 1, (1, 2, 3))

@test MLJBase.transform(p, fitresult, 10, 20, 30) == (20, 30, 10) 
@test MLJBase.inverse_transform(p, fitresult, 20, 30, 10) == (10, 20, 30)

end
true
