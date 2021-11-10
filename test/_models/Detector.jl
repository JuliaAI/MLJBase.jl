# dummy detectors that predict outliers with equal probability
struct DummyProbabilisticUnsupervisedDetector <: MMI.ProbabilisticUnsupervisedDetector end
struct DummyDeterministicUnsupervisedDetector <: MMI.DeterministicUnsupervisedDetector end
struct DummyProbabilisticSupervisedDetector <: MMI.ProbabilisticSupervisedDetector end
struct DummyDeterministicSupervisedDetector <: MMI.DeterministicSupervisedDetector end

export DummyProbabilisticUnsupervisedDetector,
       DummyDeterministicUnsupervisedDetector,
       DummyProbabilisticSupervisedDetector,
       DummyDeterministicSupervisedDetector

const DummyDetector = Union{
    DummyProbabilisticUnsupervisedDetector,
    DummyDeterministicUnsupervisedDetector,
    DummyProbabilisticSupervisedDetector,
    DummyDeterministicSupervisedDetector}
MMI.transform(::DummyDetector, verbosity, X) = fill(0.5, MLJBase.nrows(X))
MMI.input_scitype(::Type{<:DummyDetector}) = MMI.Table

const DummyProbabilisticDetector = Union{
    DummyProbabilisticUnsupervisedDetector,
    DummyProbabilisticSupervisedDetector}
MMI.predict(m::DummyProbabilisticDetector, verbosity, X) =
    MLJBase.UnivariateFinite(["in", "out"],
                            MMI.transform(m, verbosity, X),
                            augment=true, pool=missing)

const DummyDeterministicDetector = Union{
    DummyDeterministicUnsupervisedDetector,
    DummyDeterministicSupervisedDetector}
MMI.predict(::DummyDeterministicDetector, verbosity, X) =
    MLJBase.categorical(fill("in", MLJBase.nrows(X)), ordered=true, levels=["in", "out"])

const DummyUnsupervisedDetector = Union{
    DummyProbabilisticUnsupervisedDetector,
    DummyDeterministicUnsupervisedDetector}
MMI.fit(::DummyUnsupervisedDetector, verbosity, X) = nothing, nothing, nothing

const DummySupervisedDetector = Union{
    DummyProbabilisticSupervisedDetector,
    DummyDeterministicSupervisedDetector}
MMI.fit(::DummySupervisedDetector, verbosity, X, y) = nothing, nothing, nothing
