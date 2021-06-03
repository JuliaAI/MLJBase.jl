############################################
################ Structures ################ 
############################################

function is_glb(potential_glb, models)
    for model in models
        if !(potential_glb <: input_scitype(model))
            return false
        end
    end
    return true
end

function glb(models)
    for model in models
        potential_glb = input_scitype(model)
        if is_glb(potential_glb, models)
            return potential_glb
        end
    end
    return Unknown
end


mutable struct DeterministicStack{modelnames, input_scitype, target_scitype} <: DeterministicComposite
   models::Vector{Supervised}
   metalearner::Deterministic
   resampling
   function DeterministicStack(modelnames, models, metalearner, resampling)
        target_scitype = MMI.target_scitype(metalearner)
        input_scitype = glb(models)
        return new{modelnames, input_scitype, target_scitype}(models, metalearner, resampling)
   end
end

mutable struct ProbabilisticStack{modelnames, input_scitype, target_scitype} <: ProbabilisticComposite
    models::Vector{Supervised}
    metalearner::Probabilistic
    resampling
    function ProbabilisticStack(modelnames, models, metalearner, resampling)
        target_scitype = MMI.target_scitype(metalearner)
        input_scitype = glb(models)
        return new{modelnames, input_scitype, target_scitype}(models, metalearner, resampling)
    end
 end


const Stack{modelnames, input_scitype, target_scitype} = 
    Union{DeterministicStack{modelnames, input_scitype, target_scitype}, 
            ProbabilisticStack{modelnames, input_scitype, target_scitype}}

"""
    Stack(;metalearner=nothing, resampling=CV(), named_models...)

Implements the generalized Stack algorithm introduced by Wolpert 
in https://www.sciencedirect.com/science/article/abs/pii/S0893608005800231 and 
generalized by Van der Laan et al in https://biostats.bepress.com/ucbbiostat/paper222/.

Instead of using your favorite model, use them all! The Stack is a metalearning algorithm 
covered by theoretical guarantees. 

The stack in a nutshell:

- The data is split in training/validation sets 
- Each model in the library is trained on each training set and outputs predictions on the validation sets
- The metalearner is subsequently trained on those predictions and finds the best combination of the models
in the library.
- Each model is then retrained on the full data 
- "Stacking" those models and the metalearner results in an end to end fully trained model

You are not exempt from evaluating the stack.

We currently provide two different stack types the `DeterministicStack` and the `ProbabilisticStack`.
The type of which is automatically chosen by the constructor based on the provided metalearner.

# Arguments
- `metalearner::Model`: The model that will optimize the desired criterion based on its internals. 
                        For instance, a LinearRegression model will optimize the squared error.
- `resampling::Union{CV, StratifiedCV}`: The resampling strategy used to train the metalearner.
- `named_models`: The models that will be part of the library

# Examples

Let's build a simple DeterministicStack for a continuous target, we show that:  

- It composes easily with pipelines in the library.
- Some members of the library may very well be `Probabilistic` models even though the stack 
is `Deterministic`, the expected value will be taken.
- Rather than running a hyperparameter search, you can integrate each model in the stack


```julia
using MLJ
using MLJDecisionTreeInterface
using MLJLinearModels
using NearestNeighborModels
using EvoTrees
using MLJXGBoostInterface

X, y = make_regression(500, 5)

std_lr = @pipeline Standardizer() LinearRegressor()
library = (constant=ConstantRegressor(),
            tree_2=DecisionTreeRegressor(max_depth=2), 
            tree_3=DecisionTreeRegressor(max_depth=3),
            evo=EvoTreeRegressor(),
            knn=KNNRegressor(),
            xgb=XGBoostRegressor(),
            std_lr=std_lr)

stack = Stack(;metalearner=LinearRegressor(),
                resampling=CV(),
                library...)

mach = machine(stack, X, y)
evaluate!(mach; resampling=Holdout(), measure=rmse)

```

"""
function Stack(;metalearner=nothing, resampling=CV(), named_models...)
    metalearner === nothing && 
        throw(ArgumentError("metalearner=$metalearner argument should be overrided"))

    nt = NamedTuple(named_models)
    modelnames = keys(nt)
    models = collect(nt)

    if metalearner isa Deterministic
        stack =  DeterministicStack(modelnames, models, metalearner, resampling)
    elseif metalearner isa Probabilistic
        stack = ProbabilisticStack(modelnames, models, metalearner, resampling)
    else
        throw(ArgumentError("The metalearner should be a subtype 
                    of $(Union{Deterministic, Probabilistic})"))
    end
    message = MMI.clean!(stack)
    isempty(message) || @warn message

    return stack
end


function MMI.clean!(stack::Stack)
    # We only carry checks and don't try to correct the arguments here
    message = ""

    target_scitype(stack.metalearner) <: Union{AbstractArray{<:Continuous}, AbstractArray{<:Finite}} ||
        throw(ArgumentError("The metalearner should have target_scitype: 
                $(Union{AbstractArray{<:Continuous}, AbstractArray{<:Finite}})"))

    return message
end


Base.propertynames(::Stack{modelnames}) where modelnames = tuple(:resampling, :metalearner, modelnames...)


function Base.getproperty(stack::Stack{modelnames}, name::Symbol) where modelnames
    name === :metalearner && return getfield(stack, :metalearner)
    name === :resampling && return getfield(stack, :resampling)
    models = getfield(stack, :models)
    for j in eachindex(modelnames)
        name === modelnames[j] && return models[j]
    end
    error("type Stack has no property $name")
end


function Base.setproperty!(stack::Stack{modelnames}, _name::Symbol, val) where modelnames
    _name === :metalearner && return setfield!(stack, :metalearner, val)
    _name === :resampling && return setfield!(stack, :resampling, val)
    idx = findfirst(==(_name), modelnames)
    idx isa Nothing || return getfield(stack, :models)[idx] = val
    error("type Stack has no property $name")
end


MMI.target_scitype(::Type{<:Stack{modelnames, input_scitype, target_scitype}}) where 
    {modelnames, input_scitype, target_scitype} = target_scitype


MMI.input_scitype(::Type{<:Stack{modelnames, input_scitype, target_scitype}}) where 
    {modelnames, input_scitype, target_scitype} = input_scitype



###########################################################
################# Node operations Methods ################# 
###########################################################


function getfolds(y::AbstractNode, cv::CV, n::Int)
    folds = source(train_test_pairs(cv, 1:n))
end


function getfolds(y::AbstractNode, cv::StratifiedCV, n::Int)
    node(YY->train_test_pairs(cv, 1:n, YY), y)
end


function trainrows(X::AbstractNode, folds::AbstractNode, nfold)
    node((XX, ff) -> selectrows(XX, ff[nfold][1]), X, folds)
end


function testrows(X::AbstractNode, folds::AbstractNode, nfold)
    node((XX, ff) -> selectrows(XX, ff[nfold][2]), X, folds)
end


pre_judge_transform(ŷ::Node, ::Type{<:Probabilistic}, ::Type{<:AbstractArray{<:Finite}}) = 
    node(ŷ->pdf.(ŷ, levels.(ŷ)), ŷ)

pre_judge_transform(ŷ::Node, ::Type{<:Probabilistic}, ::Type{<:AbstractArray{<:Continuous}}) = 
    node(ŷ->mean.(ŷ), ŷ)

pre_judge_transform(ŷ::Node, ::Type{<:Deterministic}, ::Type{<:AbstractArray{<:Continuous}}) = 
    ŷ

#######################################
################# Fit ################# 
#######################################
"""
    fit(m::Stack, verbosity::Int, X, y)
"""
function fit(m::Stack, verbosity::Int, X, y)
    n = nrows(y)

    X = source(X)
    y = source(y)

    Zval = []
    yval = []

    folds = getfolds(y, m.resampling, n)
    # Loop over the cross validation folds to build a training set for the metalearner.
    for nfold in 1:m.resampling.nfolds
        Xtrain = trainrows(X, folds, nfold)
        ytrain = trainrows(y, folds, nfold)
        Xtest = testrows(X, folds, nfold)
        ytest = testrows(y, folds, nfold)
        
        # Train each model on the train fold and predict on the validation fold
        # predictions are subsequently used as an input to the metalearner
        Zfold = []
        for model in getfield(m, :models)
            mach = machine(model, Xtrain, ytrain)
            ypred = predict(mach, Xtest)
            # Dispatch the computation of the expected mean based on 
            # the model type and target_scytype
            ypred = pre_judge_transform(ypred, typeof(model), target_scitype(model))
            push!(Zfold, ypred)
        end

        Zfold = hcat(Zfold...)
        
        push!(Zval, Zfold)
        push!(yval, ytest)
    end

    Zval = MLJBase.table(vcat(Zval...))
    yval = vcat(yval...)

    metamach = machine(m.metalearner, Zval, yval)

    # Each model is retrained on the original full training set
    Zpred = []
    for model in getfield(m, :models)
        mach = machine(model, X, y)
        ypred = predict(mach, X)
        ypred = pre_judge_transform(ypred, typeof(model), target_scitype(model))
        push!(Zpred, ypred)
    end

    Zpred = MLJBase.table(hcat(Zpred...))
    ŷ = predict(metamach, Zpred)

    # We can infer the Surrogate by two calls to supertype
    mach = machine(supertype(supertype(typeof(m)))(), X, y; predict=ŷ)

    return!(mach, m, verbosity)

end

