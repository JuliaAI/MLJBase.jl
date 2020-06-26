# true composite models:
abstract type DeterministicComposite <: Deterministic end
abstract type ProbabilisticComposite <: Probabilistic end
abstract type IntervalComposite <: Interval end
abstract type  UnsupervisedComposite <: Unsupervised end
abstract type  StaticComposite <: Static end

# surrogate composite models:
struct DeterministicSurrogate <: Deterministic end
struct ProbabilisticSurrogate <: Probabilistic end
struct IntervalSurrogate <: Interval end
struct UnsupervisedSurrogate <: Unsupervised end
struct StaticSurrogate <: Static end

Deterministic() = DeterministicSurrogate()
Probabilistic() = ProbabilisticSurrogate()
Interval() = IntervalSurrogate()
Unsupervised() = UnsupervisedSurrogate()
Static() = StaticSurrogate()

const SupervisedComposite =
    Union{DeterministicComposite,ProbabilisticComposite,IntervalComposite}

const SupervisedSurrogate =
    Union{DeterministicSurrogate,ProbabilisticSurrogate,IntervalSurrogate}

const Surrogate = Union{DeterministicSurrogate,
                        ProbabilisticSurrogate,
                        IntervalSurrogate,
                        UnsupervisedSurrogate,
                        StaticSurrogate}

const Composite = Union{SupervisedComposite,UnsupervisedComposite,
                        StaticComposite}

for T in [:DeterministicComposite,
          :ProbabilisticComposite,
          :IntervalComposite,
          :UnsupervisedComposite,
          :StaticComposite,
          :DeterministicSurrogate,
          :ProbabilisticSurrogate,
          :IntervalSurrogate,
          :UnsupervisedSurrogate,
          :StaticSurrogate]
    eval(:(MMI.is_wrapper(::Type{$T}) = true))
    eval(:(MMI.package_name(::Type{$T}) = "MLJBase"))
    str = string(T)
    eval(:(MMI.load_path(::Type{$T}) = "MLJBase."*$str))
end

# aliases for legacy code:
const DeterministicNetwork = DeterministicComposite
const ProbabilisticNetwork = ProbabilisticComposite
const UnsupervisedNetwork = UnsupervisedComposite
