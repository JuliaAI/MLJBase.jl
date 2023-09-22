export  SimpleDeterministicNetworkCompositeModel,
    SimpleProbabilisticNetworkCompositeModel

using MLJBase

const COMPOSITE_MODELS = [
    :SimpleDeterministicNetworkCompositeModel,
    :SimpleProbabilisticNetworkCompositeModel
]
const REGRESSORS = Dict(
    :SimpleDeterministicNetworkCompositeModel => :DeterministicConstantRegressor,
    :SimpleProbabilisticNetworkCompositeModel => :ConstantRegressor,
)

const REGRESSOR_SUPERTYPES = Dict(
    :SimpleDeterministicNetworkCompositeModel => :Deterministic,
    :SimpleProbabilisticNetworkCompositeModel => :Probabilistic,
)

const COMPOSITE_SUPERTYPES = Dict(
    :SimpleDeterministicNetworkCompositeModel => :DeterministicNetworkComposite,
    :SimpleProbabilisticNetworkCompositeModel => :ProbabilisticNetworkComposite,
)

for model in COMPOSITE_MODELS
    regressor = REGRESSORS[model]
    regressor_supertype = REGRESSOR_SUPERTYPES[model]
    composite_supertype = COMPOSITE_SUPERTYPES[model]
    quote
        """
            (model)(; regressor=$($(regressor))(), transformer=FeatureSelector())

        Construct a composite model consisting of a transformer
        (`Unsupervised` model) followed by a `$($(regressor_supertype))` model.
        Intended for internal testing only.

        """
        mutable struct $(model){
            L<:$(regressor_supertype),
            T<:Unsupervised
        } <: $(composite_supertype)
            model::L
            transformer::T
        end

        function $(model)(;
            model=$(regressor)(), transformer=FeatureSelector()
        )
            composite =  $(model)(model, transformer)
            message = MLJBase.clean!(composite)
            isempty(message) || @warn message
            return composite
        end

        MLJBase.metadata_pkg(
            $(model);
            package_url = "https://github.com/alan-turing-institute/MLJBase.jl",
            is_pure_julia = true,
            is_wrapper = true
        )

        MLJBase.input_scitype(::Type{<:$(model){L,T}}) where {L,T} =
            MLJBase.input_scitype(T)
        MLJBase.target_scitype(::Type{<:$(model){L,T}}) where {L,T} =
            MLJBase.target_scitype(L)

    end |> eval
end

## FIT METHODS

for model in COMPOSITE_MODELS
    @eval function MLJBase.prefit(
        composite::$(model),
        verbosity::Integer,
        Xtrain,
        ytrain
    )
        X = source(Xtrain) # instantiates a source node
        y = source(ytrain)

        t = machine(:transformer, X)
        Xt = transform(t, X)

        l = machine(:model, Xt, y)
        yhat = predict(l, Xt)

        (predict=yhat,)
    end
end
