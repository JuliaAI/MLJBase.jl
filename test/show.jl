using .Models

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
