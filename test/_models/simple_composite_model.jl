export SimpleDeterministicCompositeModel, SimpleDeterministicNetworkCompositeModel,
    SimpleProbabilisticCompositeModel, SimpleProbabilisticNetworkCompositeModel

using MLJBase

const COMPOSITE_MODELS = [
    :SimpleDeterministicCompositeModel,
    :SimpleProbabilisticCompositeModel,
    :SimpleDeterministicNetworkCompositeModel,
    :SimpleProbabilisticNetworkCompositeModel
]
const REGRESSORS = Dict(
    :SimpleDeterministicCompositeModel => :DeterministicConstantRegressor,
    :SimpleDeterministicNetworkCompositeModel => :DeterministicConstantRegressor,
    :SimpleProbabilisticCompositeModel => :ConstantRegressor,
    :SimpleProbabilisticNetworkCompositeModel => :ConstantRegressor,
)

const REGRESSOR_SUPERTYPES = Dict(
    :SimpleDeterministicCompositeModel => :Deterministic,
    :SimpleDeterministicNetworkCompositeModel => :Deterministic,
    :SimpleProbabilisticCompositeModel => :Probabilistic,
    :SimpleProbabilisticNetworkCompositeModel => :Probabilistic,
)

const COMPOSITE_SUPERTYPES = Dict(
    :SimpleDeterministicCompositeModel => :DeterministicComposite,
    :SimpleDeterministicNetworkCompositeModel => :DeterministicNetworkComposite,
    :SimpleProbabilisticCompositeModel => :ProbabilisticComposite,
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
        (`Unsupervised` model) followed by a `$($(regressor_supertype))` model. Mainly
        intended for internal testing .

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
for model in COMPOSITE_MODELS[1:2]
    @eval function MLJBase.fit(
        composite::$(model), verbosity::Integer, Xtrain, ytrain
    )
        X = source(Xtrain) # instantiates a source node
        y = source(ytrain)

        t = machine(composite.transformer, X)
        Xt = transform(t, X)

        l = machine(composite.model, Xt, y)
        yhat = predict(l, Xt)

        mach = machine($(REGRESSOR_SUPERTYPES[model])(), X, y; predict=yhat)

        return!(mach, composite, verbosity)
    end
end

for model in COMPOSITE_MODELS[3:4]
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
