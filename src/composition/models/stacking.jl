
###########################################
################ Structure ################ 
###########################################


"""

Implements the generalized Stack algorithm introduced by Wolpert in https://www.sciencedirect.com/science/article/abs/pii/S0893608005800231
"""
mutable struct Stack{modelnames} <: DeterministicComposite
   models::NTuple{<:Any, Supervised}
   metalearner::Supervised
   cv_strategy 
   Stack(modelnames, models, metalearner, cv_strategy) = new{modelnames}(models, metalearner, cv_strategy)
end


function Stack(metalearner;cv_strategy=CV(), named_models...)
    nt = NamedTuple(named_models)
    modelnames = keys(nt)
    models = values(nt)
    return Stack(modelnames, models, metalearner, cv_strategy)
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


function getfolds(y::AbstractNode, m::Model, n::Int)
    if m.cv_strategy isa StratifiedCV
        folds = node(YY->MLJBase.train_test_pairs(m.cv_strategy, 1:n, YY), y)
    else
        folds = source(MLJBase.train_test_pairs(m.cv_strategy, 1:n))
    end
    folds
end


function trainrows(X::AbstractNode, folds::AbstractNode, nfold)
    node((XX, ff) -> selectrows(XX, ff[nfold][1]), X, folds)
end


function testrows(X::AbstractNode, folds::AbstractNode, nfold)
    node((XX, ff) -> selectrows(XX, ff[nfold][2]), X, folds)
end


function expected_value(X::AbstractNode)
    node(XX->hasproperty(XX, :prob_given_ref) ? XX.prob_given_ref[2] : XX, X)
end


"""
    fit(m::Stack, verbosity::Int, X, y)
"""
function fit(m::Stack, verbosity::Int, X, y)
    n = nrows(y)

    X = source(X)
    y = source(y)

    Zval = []
    yval = []

    folds = getfolds(y, m, n)
    for nfold in 1:m.cv_strategy.nfolds
        Xtrain = trainrows(X, folds, nfold)
        ytrain = trainrows(y, folds, nfold)
        Xtest = testrows(X, folds, nfold)
        ytest = testrows(y, folds, nfold)

        Zfold = []
        for model in m.models
            mach = machine(model, Xtrain, ytrain)
            ypred = expected_value(predict(mach, Xtest))
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
        push!(Zpred, expected_value(predict(mach, X)))
    end

    Zpred = MLJBase.table(hcat(Zpred...))

    ŷ = expected_value(predict(metamach, Zpred))

    mach = machine(Deterministic(), X, y; predict=ŷ)

    return!(mach, m, verbosity)

end

