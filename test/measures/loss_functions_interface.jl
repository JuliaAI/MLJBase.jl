rng = StableRNG(614)

# convert a Binary vector into vector of +1 or -1 values
# (for testing only):
pm1(y) = Int8(2) .* (Int8.(MLJBase.int(y))) .- Int8(3)

const MARGIN_LOSSES = MLJBase.MARGIN_LOSSES
const DISTANCE_LOSSES = MLJBase.DISTANCE_LOSSES

@testset "LossFunctions.jl - binary" begin
    y = categorical(["yes", "yes", "no", "yes"])
    yes, no = y[1], y[3]
    dyes = MLJBase.UnivariateFinite([yes, no], [0.6, 0.4])
    dno =  MLJBase.UnivariateFinite([yes, no], [0.3, 0.7])
    yhat = [dno, dno, dyes, dyes]
    X = nothing
    w = [1, 2, 3, 4]

    @test MLJBase.value(MLJBase.ZeroOneLoss(), yhat, X, y, nothing) ≈
        [1, 1, 1, 0]
    @test MLJBase.value(MLJBase.zero_one_loss, yhat, X, y, w) ≈
        [1, 2, 3, 0] ./10 .* 4

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
    X = nothing

    for M_ex in MARGIN_LOSSES
        m = eval(:(MLJBase.$M_ex()))
        @test MLJBase.value(m, yhat, X, y, nothing) ≈
            LossFunctions.value(m, yhatm, ym)
        @test mean(MLJBase.value(m, yhat, X, y, w)) ≈
            LossFunctions.value(m, yhatm, ym,
                                LossFunctions.AggMode.WeightedMean(w))
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
        @test MLJBase.value(m, yhat, X, y, nothing) ≈
            LossFunctions.value(m, yhat, y)
        @test mean(MLJBase.value(m, yhat, X, y, w)) ≈
            LossFunctions.value(m, yhat, y,
                                LossFunctions.AggMode.WeightedMean(w))
    end
end
