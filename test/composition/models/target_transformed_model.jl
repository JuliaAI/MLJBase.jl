module TestTargetTransformed

using MLJBase
using Test
using ..Models
using StableRNGs
const MMI = MLJBase.MLJModelInterface

rng = StableRNG(698790187)

atom = DeterministicConstantRegressor()
p_atom = ConstantRegressor()
whitener = UnivariateStandardizer()

@testset "constructor and clean!" begin

    model = @test_logs TargetTransformed(atom, target=whitener)
    @test model.model == atom
    @test model.inverse == nothing
    @test model.target == whitener
    @test model.cache

    model = @test_logs TargetTransformed(model=atom, target=whitener)
    @test model.model == atom
    @test model.inverse == nothing

    model = @test_logs(
        TargetTransformed(atom, target=whitener, inverse=identity))
    @test model.model == atom
    @test model.inverse == identity

    @test_throws(MLJBase.ERR_MODEL_UNSPECIFIED,
                 TargetTransformed(target=whitener))

    @test_throws(MLJBase.ERR_TOO_MANY_ARGUMENTS,
                 TargetTransformed(atom, whitener))

    @test_throws(MLJBase.err_tt_unsupported(whitener),
                 TargetTransformed(whitener, target=whitener))

    model = @test_logs((:warn, MLJBase.WARN_IDENTITY_INVERSE),
                       TargetTransformed(p_atom, target=whitener))
    @test model.inverse == identity

    model = @test_logs((:warn, MLJBase.WARN_MISSING_INVERSE),
                       TargetTransformed(atom, target=y->log.(y)))
    @test model.inverse == identity

end

# a function for transforming and it's inverse:
f(y) = sin.(y)
g(y) = asin.(y)

# implemented as a static model:
mutable struct Nonlinear <: Static
    λ::Float64 # ignored
end
Nonlinear(; λ=1) = Nonlinear(λ)
MMI.transform(::Nonlinear, _, y) = f(y)
MMI.inverse_transform(::Nonlinear, _, z) = g(z)

# some data:
X, _ = make_regression(5)
z = rand(rng, 5)
y = asin.(z)
@test sin.(y) ≈ z

# average of y on untransformed scale:
avg = mean(y)

# average of y on the non-linear scale defined by f:
avg_nonlinear = g(mean(f(y))) # = g(mean(z))

@testset begin "fit and predict"

    # Remember that `atom` predicts the constant mean of the training
    # target on all new observations. Let's
    # check it's expected behaviour before wrapping:
    fr, _, _ = MMI.fit(atom, 0, X, y)
    @test predict(atom, fr, X) ≈ fill(avg, 5)

    # Test wrapping using f and g:
    model = TargetTransformed(atom, target=f, inverse=g)
    fr1, _, _ = MMI.fit(model, 0, X, y)
    @test predict(model, fr1, X) ≈ fill(avg_nonlinear, 5)

    # Test wrapping using a `Static` transformer:
    model = TargetTransformed(atom, target=Nonlinear())
    fr1, _, _ = MMI.fit(model, 0, X, y)
    @test predict(model, fr1, X) ≈ fill(avg_nonlinear, 5)

    # Test wrapping using a non-static `Unsupervised` model:
    model = TargetTransformed(atom, target=whitener)
    fr1, _, _ = MMI.fit(model, 0, X, y)
    @test predict(model, fr1, X) ≈ fill(avg, 5) # whitener is linear

    # Test with `inverse=identity`:
    model = TargetTransformed(atom, target=Nonlinear(), inverse=identity)
    fr1, _, _ = MMI.fit(model, 0, X, y)
    @test predict(model, fr1, X) ≈ fill(mean(z), 5)

    # Test a probablistic model:
    model = TargetTransformed(p_atom, target=whitener, inverse=identity)
    fr1, _, _ = MMI.fit(model, 0, X, y)
    yhat = predict(model, fr1, X)
    @test isapprox(first(yhat).μ, 0, atol=1e-15)
end

MMI.iteration_parameter(::Type{DeterministicConstantRegressor}) = :n

@testset "traits" begin
    model = TargetTransformed(atom, target=Nonlinear())
    @test input_scitype(model) == input_scitype(atom)
    @test target_scitype(model) == target_scitype(atom)
    @test is_wrapper(model)
    @test iteration_parameter(model) == :(model.n)
end

@testset "integration" begin
    model = TargetTransformed(atom, target=Nonlinear())
    mach = machine(model, X, y)
    fit!(mach, verbosity=0)
    @test predict(mach, X) ≈ fill(avg_nonlinear, 5)
    @test issubset([:model, :target], keys(report(mach)))
    @test issubset([:model, :target], keys(fitted_params(mach)))
    @test fitted_params(mach).model.fitresult ≈ mean(z)
end

end

true
