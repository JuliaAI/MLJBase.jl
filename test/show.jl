using .Models

module FooFoo

struct FooBar{T}
    x::T
end

end

using .FooFoo

@testset "simple_repr" begin
    @test MLJBase.simple_repr(typeof(fill(3, 1))) == "Vector"
    @test MLJBase.simple_repr(typeof(FooFoo.FooBar(3))) == "FooBar"
    # with module qualifiers within type parameters
    object = FooFoo.FooBar(FooFoo.FooBar(rand(3)))
    T = typeof(object) # Main.FooFoo.FooBar{Main.FooFoo.FooBar{Vector{Float64}}}
    @test MLJBase.simple_repr(T) == "FooBar"
end

@testset "display of models" begin
    io = IOBuffer()
    show(io, KNNRegressor())
    @test String(take!(io)) == "KNNRegressor(K = 5, …)"
    show(io, MIME("text/plain"), KNNRegressor())
    @test String(take!(io)) ==
        "KNNRegressor(\n  K = 5, \n  algorithm = :kdtree, \n  "*
        "metric = Distances.Euclidean(0.0), \n  leafsize = 10, \n  "*
        "reorder = true, \n  weights = :uniform)"
end

true
