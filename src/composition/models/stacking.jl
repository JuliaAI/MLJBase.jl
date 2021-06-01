
###########################################
################ Structure ################ 
###########################################

# The type hierarchy is not very satisfying as it is as I have to 
# define almost the same struct twice for DeterministicStack and ProbabilisticStack
# The problem is that I can't subtype Composite to define the abstract Stack

mutable struct DeterministicStack{modelnames} <: DeterministicComposite
   models::NTuple{<:Any, Supervised}
   metalearner::Deterministic
   cv_strategy::Union{CV, StratifiedCV} 
   DeterministicStack(modelnames, models, metalearner, cv_strategy) = new{modelnames}(models, metalearner, cv_strategy)
end

mutable struct ProbabilisticStack{modelnames} <: ProbabilisticComposite
    models::NTuple{<:Any, Supervised}
    metalearner::Probabilistic
    cv_strategy::Union{CV, StratifiedCV} 
    ProbabilisticStack(modelnames, models, metalearner, cv_strategy) = new{modelnames}(models, metalearner, cv_strategy)
 end


const Stack{modelnames} = Union{DeterministicStack{modelnames}, ProbabilisticStack{modelnames}}

"""

Implements the generalized Stack algorithm introduced by Wolpert in https://www.sciencedirect.com/science/article/abs/pii/S0893608005800231
"""
function stack(metalearner; cv_strategy=CV(), named_models...)
    nt = NamedTuple(named_models)
    modelnames = keys(nt)
    models = values(nt)

    # input_scitype(metalearner) <: Table{<:Continuous} || 
    #     error("The metalearner should have input_scitype<:Table(Continuous)")

    # I think this suffices to ensure adequation between models and the metalearner
    # to the best of the constructor's knowledge (unknown data)
    # as I'm not considering deterministic classifiers for now
    target_scitype(metalearner) <: Union{AbstractArray{<:Continuous}, AbstractArray{<:Finite}} ||
        throw(ArgumentError("The metalearner should have target_scitype: 
                $(Union{AbstractArray{<:Continuous}, AbstractArray{<:Finite}})"))

    for model in models 
        target_scitype(model) == target_scitype(metalearner) ||
            throw(ArgumentError("target_scitype of $model should be the same as the metalearner's, 
                    ie $(target_scitype(metalearner))"))
    end

    if metalearner isa Deterministic
        return DeterministicStack(modelnames, models, metalearner, cv_strategy)
    elseif metalearner isa Probabilistic
        return ProbabilisticStack(modelnames, models, metalearner, cv_strategy)
    else
        error("The metalearner should be a subtype of $(Union{Deterministic, Probabilistic})")
    end
end


Base.propertynames(::Stack{modelnames}) where modelnames = tuple(:cv_strategy, :metalearner, :models, modelnames...)


function Base.getproperty(stack::Stack{modelnames}, name::Symbol) where modelnames
    name === :metalearner && return getfield(stack, :metalearner)
    name === :cv_strategy && return getfield(stack, :cv_strategy)
    name === :models && return getfield(stack, :models)
    models = getfield(stack, :models)
    for j in eachindex(modelnames)
        name === modelnames[j] && return models[j]
    end
    error("type Stack has no field $name")
end



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
    node(ŷ->ŷ, ŷ)
    


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

    folds = getfolds(y, m.cv_strategy, n)
    # Loop over the cross validation folds to build a training set for the metalearner.
    for nfold in 1:m.cv_strategy.nfolds
        Xtrain = trainrows(X, folds, nfold)
        ytrain = trainrows(y, folds, nfold)
        Xtest = testrows(X, folds, nfold)
        ytest = testrows(y, folds, nfold)
        
        # Train each model on the train fold and predict on the validation fold
        # predictions are subsequently used as an input to the metalearner
        Zfold = []
        for model in m.models
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
    for model in m.models
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

