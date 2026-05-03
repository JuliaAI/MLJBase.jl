module TestFreezable

using MLJBase
using Test
using ..Models
using ..TestUtilities
using StableRNGs
const MMI = MLJBase.MLJModelInterface

@testset "Freezable types and constructor" begin
    @testset "deterministic model wrapping" begin
        atom = DeterministicConstantRegressor()
        model = Freezable(atom)
        @test model isa MLJBase.FreezableDeterministic
        @test model isa MLJBase.DeterministicNetworkComposite
    end

    @testset "probabilistic model wrapping" begin
        atom = ConstantRegressor()
        model = Freezable(atom)
        @test model isa MLJBase.FreezableProbabilistic
        @test model isa MLJBase.ProbabilisticNetworkComposite
    end

    @testset "unsupervised model wrapping" begin
        atom = UnivariateStandardizer()
        model = Freezable(atom)
        @test model isa MLJBase.FreezableUnsupervised
        @test model isa MLJBase.UnsupervisedNetworkComposite
    end

    @testset "default values" begin
        atom = DeterministicConstantRegressor()
        model = Freezable(atom)
        @test model.frozen == true
        @test model.cache == true
    end

    @testset "custom keyword arguments" begin
        atom = DeterministicConstantRegressor()
        model = Freezable(atom; frozen=false, cache=false)
        @test model.frozen == false
        @test model.cache == false
    end

    @testset "error: no model" begin
        @test_throws(
            MLJBase.ERR_FREEZABLE_MODEL_UNSPECIFIED,
            Freezable(),
        )
    end

    @testset "error: unsupported model type (Static)" begin
        static_model = Averager(mix=0.5)
        @test_throws(ArgumentError, Freezable(static_model))
    end

    @testset "error: too many arguments" begin
        @test_throws(
            MLJBase.ERR_FREEZABLE_TOO_MANY_ARGUMENTS,
            Freezable(DeterministicConstantRegressor(), ConstantRegressor()),
        )
    end

    @testset "model field identity" begin
        atom = DeterministicConstantRegressor()
        model = Freezable(atom)
        @test model.model === atom
    end

end

@testset "constructor type correctness across model types" begin
    EXPECTED_SUPER = Dict(
        MLJBase.Deterministic  => MLJBase.DeterministicNetworkComposite,
        MLJBase.Probabilistic  => MLJBase.ProbabilisticNetworkComposite,
        MLJBase.Unsupervised   => MLJBase.UnsupervisedNetworkComposite,
    )
    atoms = [
        DeterministicConstantRegressor(),
        ConstantRegressor(),
        ConstantClassifier(),
        DecisionTreeClassifier(),
        DecisionTreeRegressor(),
        UnivariateStandardizer(),
        Standardizer(),
    ]
    for atom in atoms
        wrapped = Freezable(atom)
        abstract_atom = MMI.abstract_type(atom)
        expected_super = EXPECTED_SUPER[abstract_atom]
        @test wrapped isa expected_super
        @test wrapped.model === atom
    end
end

# Define iteration_parameter for DeterministicConstantRegressor within this module
# so we can test that Freezable prepends :model to the path.
MMI.iteration_parameter(::Type{DeterministicConstantRegressor}) = :n

@testset "trait delegation" begin
    atom_det = DeterministicConstantRegressor()
    wrapped_det = Freezable(atom_det)

    atom_clf = DecisionTreeClassifier()
    wrapped_clf = Freezable(atom_clf)

    @testset "input_scitype delegation" begin
        @test MLJBase.input_scitype(typeof(wrapped_det)) ==
              MLJBase.input_scitype(typeof(atom_det))
        @test MLJBase.input_scitype(typeof(wrapped_clf)) ==
              MLJBase.input_scitype(typeof(atom_clf))
    end

    @testset "target_scitype delegation" begin
        @test MLJBase.target_scitype(typeof(wrapped_det)) ==
              MLJBase.target_scitype(typeof(atom_det))
        @test MLJBase.target_scitype(typeof(wrapped_clf)) ==
              MLJBase.target_scitype(typeof(atom_clf))
    end

    @testset "is_wrapper" begin
        @test MLJBase.is_wrapper(typeof(wrapped_det)) == true
        @test MLJBase.is_wrapper(typeof(wrapped_clf)) == true
    end

    @testset "package_name" begin
        @test MLJBase.package_name(typeof(wrapped_det)) == "MLJBase"
    end

    @testset "load_path" begin
        @test MLJBase.load_path(typeof(wrapped_det)) == "MLJBase.Freezable"
    end

    @testset "constructor" begin
        @test MLJBase.constructor(typeof(wrapped_det)) == Freezable
    end

    @testset "iteration_parameter prepends :model" begin
        @test MLJBase.iteration_parameter(typeof(wrapped_det)) == :(model.n)
    end
end

@testset "trait delegation across model types" begin
    DELEGATED_TRAITS = [
        MLJBase.input_scitype,
        MLJBase.target_scitype,
        MLJBase.fit_data_scitype,
        MLJBase.predict_scitype,
        MLJBase.transform_scitype,
        MLJBase.is_pure_julia,
        MLJBase.supports_weights,
        MLJBase.supports_class_weights,
        MLJBase.supports_training_losses,
        MLJBase.is_supervised,
        MLJBase.prediction_type,
    ]
    atoms = [
        DeterministicConstantRegressor(),
        ConstantRegressor(),
        ConstantClassifier(),
        DecisionTreeClassifier(),
        DecisionTreeRegressor(),
        UnivariateStandardizer(),
        Standardizer(),
    ]
    for atom in atoms
        wrapped = Freezable(atom)
        for trait_fn in DELEGATED_TRAITS
            @test trait_fn(typeof(wrapped)) == trait_fn(typeof(atom))
        end
    end
end

@testset "Freezable export" begin
    @test :Freezable in names(MLJBase)
end

@testset "supervised Freezable end-to-end" begin
    rng = StableRNG(42)
    X = MLJBase.table(randn(rng, 20, 3))
    y = randn(rng, 20)

    # Freezable-wrapped model (frozen=false so it behaves like unwrapped)
    fmodel = Freezable(DeterministicConstantRegressor(); frozen=false)
    fmach = machine(fmodel, X, y)
    fit!(fmach; verbosity=0)
    fpreds = predict(fmach, X)

    # Unwrapped model
    atom = DeterministicConstantRegressor()
    amach = machine(atom, X, y)
    fit!(amach; verbosity=0)
    apreds = predict(amach, X)

    @test fpreds ≈ apreds

    fp = fitted_params(fmach)
    @test :model in keys(fp)

    rep = report(fmach)
    @test rep === nothing || rep isa NamedTuple
end

@testset "unsupervised Freezable end-to-end" begin
    rng = StableRNG(123)
    v = randn(rng, 50)

    fmodel = Freezable(UnivariateStandardizer(); frozen=false)
    fmach = machine(fmodel, v)
    fit!(fmach; verbosity=0)
    ftransformed = transform(fmach, v)

    atom = UnivariateStandardizer()
    amach = machine(atom, v)
    fit!(amach; verbosity=0)
    atransformed = transform(amach, v)

    @test ftransformed ≈ atransformed
end

@testset "initial training always proceeds when frozen" begin
    rng = StableRNG(77)
    X = MLJBase.table(randn(rng, 20, 3))
    y = randn(rng, 20)

    model = Freezable(DeterministicConstantRegressor(); frozen=true)
    mach = machine(model, X, y)
    @test mach.state == 0

    fit!(mach; verbosity=0)
    @test mach.state > 0

    preds = predict(mach, X)
    @test length(preds) == 20
    @test all(isfinite, preds)
end

@testset "frozen training behavior" begin
    rng = StableRNG(99)

    # Rows 1:10 have mean 1.0, rows 11:20 have mean 100.0.
    # DeterministicConstantRegressor predicts the mean, so we can detect
    # whether retraining happened by checking the predicted value.
    X = MLJBase.table(randn(rng, 20, 2))
    y = vcat(fill(1.0, 10), fill(100.0, 10))

    model = Freezable(DeterministicConstantRegressor(), frozen=true)
    @test model.frozen == true

    # Initial fit with rows 1:10 — should proceed even though frozen=true
    mach = machine(model, X, y)
    fit!(mach; rows=1:10, verbosity=0)
    @test mach.state > 0
    preds_initial = predict(mach, X)
    @test all(p -> p ≈ 1.0, preds_initial)

    # Second fit with rows 11:20 — should be a no-op (frozen skip)
    fit!(mach; rows=11:20, verbosity=0)
    preds_frozen = predict(mach, X)
    @test preds_frozen == preds_initial

    # Thaw and retrain
    model.frozen = false
    @test model.frozen == false
    fit!(mach; rows=11:20, verbosity=0)
    preds_thawed = predict(mach, X)
    @test all(p -> p ≈ 100.0, preds_thawed)
    @test preds_thawed != preds_initial
end

@testset "freeze! and thaw! on models" begin
    model = Freezable(DeterministicConstantRegressor(); frozen=false)
    @test model.frozen == false

    # freeze! sets frozen=true and returns the model
    ret = freeze!(model)
    @test model.frozen == true
    @test ret === model

    # thaw! sets frozen=false and returns the model
    ret = thaw!(model)
    @test model.frozen == false
    @test ret === model
end

@testset "thaw! triggers retraining" begin
    rng = StableRNG(88)
    X = MLJBase.table(randn(rng, 20, 3))
    y = randn(rng, 20)

    model = Freezable(DeterministicConstantRegressor(); frozen=true)
    mach = machine(model, X, y)
    fit!(mach; verbosity=0)
    state_after_initial = mach.state
    @test state_after_initial > 0

    thaw!(model)
    fit!(mach; verbosity=0)
    @test mach.state > state_after_initial
end

@testset "frozen skip with different data sizes" begin
    for n in [5, 10, 30, 50]
        rng = StableRNG(hash(n))
        X = MLJBase.table(randn(rng, n, 2))
        y = randn(rng, n)

        model = Freezable(DeterministicConstantRegressor(); frozen=true)
        mach = machine(model, X, y)
        fit!(mach; verbosity=0)
        preds1 = predict(mach, X)

        half = max(1, n ÷ 2)
        fit!(mach; rows=1:half, verbosity=0)
        preds2 = predict(mach, X)

        @test preds1 == preds2
    end
end

@testset "pipeline with frozen component" begin
    rng = StableRNG(555)

    X_part1 = fill(1.0, 10, 2)
    X_part2 = fill(100.0, 10, 2)
    X = MLJBase.table(vcat(X_part1, X_part2))
    y = vcat(fill(10.0, 10), fill(200.0, 10))

    frozen_std = Freezable(Standardizer(), frozen=true)
    pipe = Pipeline(
        std = frozen_std,
        reg = DeterministicConstantRegressor(),
    )

    mach = machine(pipe, X, y)
    fit!(mach; rows=1:10, verbosity=0)
    preds_first = predict(mach, X)
    @test all(p -> p ≈ 10.0, preds_first)

    # Retrain with rows 11:20: frozen Standardizer skips, predictor retrains
    fit!(mach; rows=11:20, verbosity=0)
    preds_second = predict(mach, X)
    @test all(p -> p ≈ 200.0, preds_second)
    @test length(preds_second) == 20
end

@testset "pipeline frozen component is not retrained on row change" begin
    # Regression test: when a `Freezable` component is inside a pipeline and the
    # outer machine is re-fitted with new rows, the parent composite rebuilds
    # its learning network. The frozen inner model must NOT be retrained — its
    # fitted_params must match the first fit byte-for-byte.
    rng = StableRNG(777)
    X = (x1 = randn(rng, 200), x2 = randn(rng, 200))
    y = randn(rng, 200)

    frozen_std = Freezable(Standardizer(), frozen=true)
    pipe = Pipeline(
        scaler = frozen_std,
        reg    = DeterministicConstantRegressor(),
    )

    mach = machine(pipe, X, y)
    fit!(mach; rows=1:100, verbosity=0)
    fp_first = fitted_params(mach).scaler.model

    fit!(mach; rows=101:200, verbosity=0)
    fp_second = fitted_params(mach).scaler.model

    @test fp_first == fp_second
end

@testset "pipeline frozen component fit/skip sequence" begin
    # Use @test_mach_sequence to assert the inner Standardizer's machine is
    # trained once and not retrained on a subsequent row-change refit.
    rng = StableRNG(778)
    X = (x1 = randn(rng, 60), x2 = randn(rng, 60))
    y = randn(rng, 60)

    frozen_std = Freezable(Standardizer(), frozen=true)
    pipe = Pipeline(
        scaler = frozen_std,
        reg    = DeterministicConstantRegressor(),
    )
    mach = machine(pipe, X, y)

    # Drive the channel once to discover the machine objects, then assert.
    fit!(mach; rows=1:30, verbosity=-5000)
    seq1 = MLJBase.flush!(MLJBase.MACHINE_CHANNEL)
    @test any(t -> t[1] === :train && t[2].model isa Symbol && t[2].model === :model, seq1)

    # Second fit on different rows must not produce a :train event for the inner
    # Standardizer machine (the one whose symbolic model is `:model`, owned by the
    # FreezableUnsupervised composite).
    fit!(mach; rows=31:60, verbosity=-5000)
    seq2 = MLJBase.flush!(MLJBase.MACHINE_CHANNEL)
    inner_train_events =
        filter(t -> t[1] === :train && t[2].model isa Symbol && t[2].model === :model, seq2)
    @test isempty(inner_train_events)
end

@testset "pipeline frozen component skip with varying sizes" begin
    for n in [10, 20, 40]
        rng = StableRNG(hash(n))
        half = n ÷ 2

        X = MLJBase.table(randn(rng, n, 3))
        y = vcat(fill(1.0, half), fill(100.0, n - half))

        frozen_std = Freezable(Standardizer(), frozen=true)
        pipe = Pipeline(
            std = frozen_std,
            reg = DeterministicConstantRegressor(),
        )

        mach = machine(pipe, X, y)
        fit!(mach; rows=1:half, verbosity=0)
        preds_first = predict(mach, X)
        @test all(p -> p ≈ 1.0, preds_first)

        fit!(mach; rows=(half+1):n, verbosity=0)
        preds_second = predict(mach, X)
        @test all(p -> p ≈ 100.0, preds_second)
        @test length(preds_second) == n
    end
end

@testset "thawed supervised equivalence" begin
    rng = StableRNG(200)
    X = MLJBase.table(randn(rng, 30, 4))
    y = randn(rng, 30)

    fmodel = Freezable(DeterministicConstantRegressor(); frozen=false)
    fmach = machine(fmodel, X, y)
    fit!(fmach; verbosity=0)
    fpreds = predict(fmach, X)

    atom = DeterministicConstantRegressor()
    amach = machine(atom, X, y)
    fit!(amach; verbosity=0)
    apreds = predict(amach, X)

    @test fpreds ≈ apreds
end

@testset "thawed unsupervised equivalence" begin
    rng = StableRNG(300)
    v = randn(rng, 40)

    fmodel = Freezable(UnivariateStandardizer(); frozen=false)
    fmach = machine(fmodel, v)
    fit!(fmach; verbosity=0)
    ftransformed = transform(fmach, v)

    atom = UnivariateStandardizer()
    amach = machine(atom, v)
    fit!(amach; verbosity=0)
    atransformed = transform(amach, v)

    @test ftransformed ≈ atransformed
end

end # module
true
