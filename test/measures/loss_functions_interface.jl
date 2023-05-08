rng = StableRNG(614)

# convert a Binary vector into vector of +1 or -1 values
# (for testing only):
pm1(y) = Int8(2) .* (Int8.(MLJBase.int(y))) .- Int8(3)

const MARGIN_LOSSES = MLJBase.MARGIN_LOSSES
const DISTANCE_LOSSES = MLJBase.DISTANCE_LOSSES

# using `WeightedSum` instead of `WeightedMean`; see
# https://github.com/JuliaML/LossFunctions.jl/issues/149
WeightedSum(w) = LossFunctions.AggMode.WeightedMean(w, normalize=false)

@testset "naked" begin
    @test MLJBase.naked(MLJBase.LossFunctions.PeriodicLoss{Float64}) ==
        :PeriodicLoss
    @test MLJBase.naked(MLJBase.LossFunctions.PeriodicLoss) ==
        :PeriodicLoss
end

@testset "LossFunctions.jl - binary" begin
    y = categorical(["yes", "yes", "no", "yes"])
    yes, no = y[1], y[3]
    dyes = MLJBase.UnivariateFinite([yes, no], [0.6, 0.4])
    dno =  MLJBase.UnivariateFinite([yes, no], [0.3, 0.7])
    yhat = [dno, dno, dyes, dyes]
    w = [1, 2, 3, 4]

    @test MLJBase.ZeroOneLoss()(yhat, y) ≈ [1, 1, 1, 0]
    @test MLJBase.zero_one_loss(yhat,y, w) ≈ [1, 2, 3, 0]

    N = 10
    y = categorical(rand(rng, ["yes", "no"], N), ordered=true)
    levels!(y, ["no", "yes"])
    no, yes = MLJBase.classes(y[1])
    @test pm1([yes, no]) in [[+1, -1], [-1, +1]]
    ym = pm1(y) # observations for raw LossFunctions measure
    p_vec = rand(N)
    yhat = MLJBase.UnivariateFinite([no, yes], p_vec, augment=true)
    yhatm = MLJBase._scale.(p_vec) # predictions for raw LossFunctions measure
    w = rand(rng, N)

    for M_ex in MARGIN_LOSSES
        m = eval(:(MLJBase.$M_ex()))
        @test m(yhat, y) ≈ (getfield(m, :loss)).(yhatm, ym)
        @test m(yhat, y, w) ≈
            w .* (getfield(m, :loss)).(yhatm, ym)
    end
end

@testset "LossFunctions.jl - continuous" begin
    # losses for continuous targets:
    N    = 10
    y    = randn(rng, N)
    yhat = randn(rng, N)
    X    = nothing
    w    = rand(rng, N)

    for M_ex in DISTANCE_LOSSES
        m = eval(:(MLJBase.$M_ex()))
        m_ex = MLJBase.snakecase(M_ex)
        @test m == eval(:(MLJBase.$m_ex))
        @test m(yhat, y) ≈
            (getfield(m, :loss)).(yhat, y)
        @test m(yhat ,y, w) ≈
            w .* (getfield(m, :loss)).(yhat, y)
    end
end
