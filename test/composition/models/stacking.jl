module TestStacking

using Test
using StableRNGs
using MLJBase
using ..Models


rng = StableRNG(698790187)


@testset "Testing the Binary Outcome stack" begin
    library = [KNNRegressor(), 
                DecisionTreeRegressor(), 
                FooBarRegressor(;lambda=0.1), 
                FooBarRegressor(;lambda=0)]
    stack = Stack(library, 
                FooBarRegressor(), 
                CV(;nfolds=3))
    X, y = make_regression(200, 5;rng=rng)

    mach = machine(stack, X, y)
    fit!(mach)

    fp = fitted_params(mach)

end

end