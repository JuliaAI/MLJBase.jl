############################################
################ Structures ################ 
############################################


function glb(types...)
    # If a lower bound is in the types then it is greatest
    # else we just return Unknown for now
    for type in types
        all(type <: t_ for t_ in types) && return type
    end
    return Unknown
end


function input_target_scitypes(models, metalearner)
    # The target scitype is defined as the greatest lower bound of the 
    # metalearner and the base models in the library
    all_tg_scitypes = [target_scitype(metalearner), 
                        [target_scitype(m) for m in models]...]
    tg_scitype = glb(all_tg_scitypes...)
    # The input scitype is defined as the greatest lower bound of the 
    # base models in the library
    inp_scitype = glb([input_scitype(m) for m in models]...)

    return inp_scitype, tg_scitype
end


mutable struct DeterministicStack{modelnames, inp_scitype, tg_scitype} <: DeterministicComposite
   models::Vector{Supervised}
   metalearner::Deterministic
   resampling
   function DeterministicStack(modelnames, models, metalearner, resampling)
        inp_scitype, tg_scitype = input_target_scitypes(models, metalearner)
        return new{modelnames, inp_scitype, tg_scitype}(models, metalearner, resampling)
   end
end

mutable struct ProbabilisticStack{modelnames, inp_scitype, tg_scitype} <: ProbabilisticComposite
    models::Vector{Supervised}
    metalearner::Probabilistic
    resampling
    function ProbabilisticStack(modelnames, models, metalearner, resampling)
        inp_scitype, tg_scitype = input_target_scitypes(models, metalearner)
        return new{modelnames, inp_scitype, tg_scitype}(models, metalearner, resampling)
    end
 end


const Stack{modelnames, inp_scitype, tg_scitype} = 
    Union{DeterministicStack{modelnames, inp_scitype, tg_scitype}, 
            ProbabilisticStack{modelnames, inp_scitype, tg_scitype}}

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

- Some members of the library may very well be `Probabilistic` models even though the stack 
is `Deterministic`, the expected value will be taken.
- Rather than running a hyperparameter search, you can integrate each model in the stack


```julia
using MLJ

DecisionTreeRegressor = @load DecisionTreeRegressor pkg=DecisionTree
EvoTreeRegressor = @load EvoTreeRegressor
XGBoostRegressor = @load XGBoostRegressor
KNNRegressor = @load KNNRegressor pkg=NearestNeighborModels
LinearRegressor = @load LinearRegressor pkg=MLJLinearModels

X, y = make_regression(500, 5)

stack = Stack(;metalearner=LinearRegressor(),
                resampling=CV(),
                constant=ConstantRegressor(),
                tree_2=DecisionTreeRegressor(max_depth=2), 
                tree_3=DecisionTreeRegressor(max_depth=3),
                evo=EvoTreeRegressor(),
                knn=KNNRegressor(),
                xgb=XGBoostRegressor())

mach = machine(stack, X, y)
evaluate!(mach; resampling=Holdout(), measure=rmse)

```

"""
function Stack(;metalearner=nothing, resampling=CV(), named_models...)
    metalearner === nothing && 
        throw(ArgumentError("No metalearner specified. Use Stack(metalearner=...)"))

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
    # Issuing clean! statement
    message = MMI.clean!(stack)
    isempty(message) || @warn message

    # Warning is either input_scitype/target_scitype is 
    # Unknown at construction time
    params = typeof(stack).parameters
    params[end-1] == Unknown && @warn "Could not infer input_scitype of the stack"
    params[end] == Unknown && @warn "Could not infer target_scitype of the stack"

    return stack
end


function MMI.clean!(stack::Stack)
    # We only carry checks and don't try to correct the arguments here
    message = ""
    # Checking target_scitype and input_scitype have not been changed from the original stack
    params = typeof(stack).parameters
    inp_scitype, tg_scitype = input_target_scitypes(getfield(stack, :models), stack.metalearner)
    inp_scitype == params[end-1] || 
            throw(DomainError(inp_scitype, "The newly inferred input_scitype of the stack doesn't 
            match its original one. You have probably changed one of the model or the metalearner 
            to a non compatible type."))
    tg_scitype == params[end] || 
            throw(DomainError(tg_scitype, "The newly inferred target_scitype of the stack doesn't 
            match its original one. You have probably changed one of the model or the metalearner 
            to a non compatible type."))
    # Checking the target scitype is consistent with either Probabilistic/Deterministic Stack
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


MLJBase.load_path(::Type{<:ProbabilisticStack}) = "MLJBase.ProbabilisticStack"
MLJBase.load_path(::Type{<:DeterministicStack}) = "MLJBase.DeterministicStack"
MLJBase.package_name(::Type{<:Stack}) = "MLJBase"
MLJBase.package_uuid(::Type{<:Stack}) = "a7f614a8-145f-11e9-1d2a-a57a1082229d"
MLJBase.package_url(::Type{<:Stack}) = "https://github.com/alan-turing-institute/MLJBase.jl"
MLJBase.package_license(::Type{<:Stack}) = "MIT"

###########################################################
################# Node operations Methods ################# 
###########################################################


getfolds(y::AbstractNode, cv::CV, n::Int) = 
    source(train_test_pairs(cv, 1:n))

getfolds(y::AbstractNode, cv::StratifiedCV, n::Int) =
    node(YY->train_test_pairs(cv, 1:n, YY), y)

trainrows(X::AbstractNode, folds::AbstractNode, nfold) =
    node((XX, ff) -> selectrows(XX, ff[nfold][1]), X, folds)

testrows(X::AbstractNode, folds::AbstractNode, nfold) = 
    node((XX, ff) -> selectrows(XX, ff[nfold][2]), X, folds)

    
pre_judge_transform(ŷ::Node, ::Type{<:Probabilistic}, ::Type{<:AbstractArray{<:Finite}}) = 
    node(ŷ -> pdf(ŷ, levels(first(ŷ))), ŷ)

pre_judge_transform(ŷ::Node, ::Type{<:Probabilistic}, ::Type{<:AbstractArray{<:Continuous}}) = 
    node(ŷ->mean.(ŷ), ŷ)

pre_judge_transform(ŷ::Node, ::Type{<:Deterministic}, ::Type{<:AbstractArray{<:Continuous}}) = 
    ŷ


function oos_set(m::Stack, folds::AbstractNode, Xs::Source, ys::Source)
    Zval = []
    yval = []
    # Loop over the cross validation folds to build a training set for the metalearner.
    for nfold in 1:m.resampling.nfolds
        Xtrain = trainrows(Xs, folds, nfold)
        ytrain = trainrows(ys, folds, nfold)
        Xtest = testrows(Xs, folds, nfold)
        ytest = testrows(ys, folds, nfold)
        
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

    Zval, yval
end

#######################################
################# Fit ################# 
#######################################
"""
    fit(m::Stack, verbosity::Int, X, y)
"""
function fit(m::Stack, verbosity::Int, X, y)
    n = nrows(y)

    Xs = source(X)
    ys = source(y)

    folds = getfolds(ys, m.resampling, n)
    
    Zval, yval = oos_set(m, folds, Xs, ys)

    metamach = machine(m.metalearner, Zval, yval)

    # Each model is retrained on the original full training set
    Zpred = []
    for model in getfield(m, :models)
        mach = machine(model, Xs, ys)
        ypred = predict(mach, Xs)
        ypred = pre_judge_transform(ypred, typeof(model), target_scitype(model))
        push!(Zpred, ypred)
    end

    Zpred = MLJBase.table(hcat(Zpred...))
    ŷ = predict(metamach, Zpred)

    # We can infer the Surrogate by two calls to supertype
    mach = machine(supertype(supertype(typeof(m)))(), Xs, ys; predict=ŷ)

    return!(mach, m, verbosity)

end

