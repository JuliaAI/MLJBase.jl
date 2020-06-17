# true composite models:
abstract type DeterministicComposite <: Deterministic end
abstract type ProbabilisticComposite <: Probabilistic end
abstract type  UnsupervisedComposite <: Unsupervised end

# surrogate composite models:
struct DeterministicSurrogate <: Deterministic
    input_scitype::DataType
    target_scitype::DataType
    output_scitype::DataType
end
struct ProbabilisticSurrogate <: Probabilistic
    input_scitype::DataType
    target_scitype::DataType
    output_scitype::DataType
end
struct UnsupervisedSurrogate <: Unsupervised
    input_scitype::DataType
    target_scitype::DataType
    output_scitype::DataType
end

Deterministic(; input_scitype=Unknown
              , target_scitype=Unknown
              , output_scitype=Unknown) =
                  DeterministicSurrogate(input_scitype,
                                         target_scitype,
                                         output_scitype)

Probabilistic(; input_scitype=Unknown
              , target_scitype=Unknown
              , output_scitype=Unknown) =
                  ProbabilisticSurrogate(input_scitype,
                                         target_scitype,
                                         output_scitype)

Unsupervised(; input_scitype=Unknown
              , target_scitype=Unknown
              , output_scitype=Unknown) =
                  UnsupervisedSurrogate(input_scitype,
                                         target_scitype,
                                         output_scitype)

const SupervisedComposite =
    Union{DeterministicComposite,ProbabilisticComposite}

const SupervisedSurrogate =
    Union{DeterministicSurrogate,ProbabilisticSurrogate}

const Surrogate = Union{DeterministicSurrogate,
                        ProbabilisticSurrogate,
                        UnsupervisedSurrogate}

const Composite = Union{SupervisedComposite,UnsupervisedComposite}

# to suppress inclusion of these types in models():
MMI.is_wrapper(::Type{DeterministicComposite}) = true
MMI.is_wrapper(::Type{ProbabilisticComposite}) = true
MMI.is_wrapper(::Type{UnsupervisedComposite}) = true
MMI.is_wrapper(::Type{DeterministicSurrogate}) = true
MMI.is_wrapper(::Type{ProbabilisticSurrogate}) = true
MMI.is_wrapper(::Type{UnsupervisedSurrogate}) = true

# aliases for legacy code:
const DeterministicNetwork = DeterministicComposite
const ProbabilisticNetwork = ProbabilisticComposite
const UnsupervisedNetwork = UnsupervisedComposite
