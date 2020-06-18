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

# to suppress inclusion of these types in models():
MMI.is_wrapper(::Type{DeterministicComposite}) = true
MMI.is_wrapper(::Type{ProbabilisticComposite}) = true
MMI.is_wrapper(::Type{IntervalComposite}) = true
MMI.is_wrapper(::Type{UnsupervisedComposite}) = true
MMI.is_wrapper(::Type{StaticComposite}) = true
MMI.is_wrapper(::Type{DeterministicSurrogate}) = true
MMI.is_wrapper(::Type{ProbabilisticSurrogate}) = true
MMI.is_wrapper(::Type{IntervalSurrogate}) = true
MMI.is_wrapper(::Type{UnsupervisedSurrogate}) = true
MMI.is_wrapper(::Type{StaticSurrogate}) = true

# aliases for legacy code:
const DeterministicNetwork = DeterministicComposite
const ProbabilisticNetwork = ProbabilisticComposite
const UnsupervisedNetwork = UnsupervisedComposite
