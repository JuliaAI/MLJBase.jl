
###########################################
################ Structure ################ 
###########################################

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



###########################################
################# Methods ################# 
###########################################


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


#############################################################
################# Pre-judge transformations ################# 
#############################################################

pre_judge_transform(ŷ, ::Type{<:Probabilistic}, ::Type{<:AbstractArray{<:Finite}}) = 
    node(ŷ->pdf(ŷ, levels(ŷ)), ŷ)

pre_judge_transform(ŷ, ::Type{<:Probabilistic}, ::Type{<:AbstractArray{<:Continuous}}) = 
    node(ŷ->mean.(ŷ), ŷ)

pre_judge_transform(ŷ, ::Type{<:Deterministic}, ::Type{<:AbstractArray{<:Continuous}}) = 
    node(ŷ->ŷ, ŷ)
    

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
    for nfold in 1:m.cv_strategy.nfolds
        Xtrain = trainrows(X, folds, nfold)
        ytrain = trainrows(y, folds, nfold)
        Xtest = testrows(X, folds, nfold)
        ytest = testrows(y, folds, nfold)

        Zfold = []
        for model in m.models
            mach = machine(model, Xtrain, ytrain)
            ypred = predict(mach, Xtest)
            # Dispatch the computation of the expected mean based on 
            # the model type and target_scytype
            println(ypred)
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

    Zpred = []
    for model in m.models
        mach = machine(model, X, y)
        ypred = predict(mach, X)
        println(ypred)
        ypred = pre_judge_transform(ypred, typeof(model), target_scitype(model))
        push!(Zpred, ypred)
    end

    Zpred = MLJBase.table(hcat(Zpred...))

    ŷ = predict(metamach, Zpred)

    mach = machine(supertype(supertype(typeof(m)))(), X, y; predict=ŷ)

    return!(mach, m, verbosity)

end

