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
    all_tg_scitypes = [target_scitype(m) for m in models]
    tg_scitype = glb(target_scitype(metalearner), all_tg_scitypes...)
    # The input scitype is defined as the greatest lower bound of the
    # base models in the library
    inp_scitype = glb([input_scitype(m) for m in models]...)

    return inp_scitype, tg_scitype
end


mutable struct DeterministicStack{
    modelnames,
    inp_scitype,
    tg_scitype
} <: DeterministicComposite

    models::Vector{Supervised}
    metalearner::Deterministic
    resampling
    measures::Union{Nothing,AbstractVector}
    cache::Bool
    acceleration::AbstractResource
    function DeterministicStack(
        modelnames,
        models,
        metalearner,
        resampling,
        measures,
        cache,
        acceleration
    )
        inp_scitype, tg_scitype = input_target_scitypes(models, metalearner)
        return new{modelnames, inp_scitype, tg_scitype}(
            models,
            metalearner,
            resampling,
            measures,
            cache,
            acceleration
        )
    end
end

mutable struct ProbabilisticStack{
    modelnames,
    inp_scitype,
    tg_scitype
} <: ProbabilisticComposite

    models::Vector{Supervised}
    metalearner::Probabilistic
    resampling
    measures::Union{Nothing,AbstractVector}
    cache::Bool
    acceleration::AbstractResource
    function ProbabilisticStack(
        modelnames,
        models,
        metalearner,
        resampling,
        measures,
        cache,
        acceleration
    )

        inp_scitype, tg_scitype = input_target_scitypes(models, metalearner)
        return new{modelnames, inp_scitype, tg_scitype}(
            models,
            metalearner,
            resampling,
            measures,
            cache,
            acceleration
        )
    end
 end


const Stack{modelnames, inp_scitype, tg_scitype} = Union{
    DeterministicStack{modelnames, inp_scitype, tg_scitype},
    ProbabilisticStack{modelnames, inp_scitype, tg_scitype}
}

"""
    Stack(; metalearner=nothing, name1=model1, name2=model2, ..., keyword_options...)

Implements the two-layer generalized stack algorithm introduced by
[Wolpert
(1992)](https://www.sciencedirect.com/science/article/abs/pii/S0893608005800231)
and generalized by [Van der Laan et al
(2007)](https://biostats.bepress.com/ucbbiostat/paper222/). Returns an
instance of type `ProbabilisticStack` or `DeterministicStack`,
depending on the prediction type of `metalearner`.

When training a machine bound to such an instance:

- The data is split into training/validation sets according to the
  specified `resampling` strategy.

- Each base model `model1`, `model2`, ... is trained on each training
  subset and outputs predictions on the corresponding validation
  sets. The multi-fold predictions are spliced together into a
  so-called out-of-sample prediction for each model.

- The adjudicating model, `metalearner`, is subsequently trained on
  the out-of-sample predictions to learn the best combination of base
  model predictions.

- Each base model is retrained on all supplied data for purposes of
  passing on new production data onto the adjudicator for making new
  predictions

### Arguments

- `metalearner::Supervised`: The model that will optimize the desired
  criterion based on its internals.  For instance, a LinearRegression
  model will optimize the squared error.

- `resampling`: The resampling strategy used
  to prepare out-of-sample predictions of the base learners.

- `measures`: A measure or iterable over measures, to perform an internal
  evaluation of the learners in the Stack while training. This is not for the
  evaluation of the Stack itself.

- `cache`: Whether machines created in the learning network will cache data or not.

- `acceleration`: A supported `AbstractResource` to define the training parallelization
  mode of the stack.

- `name1=model1, name2=model2, ...`: the `Supervised` model instances
  to be used as base learners.  The provided names become properties
  of the instance created to allow hyper-parameter access


### Example

The following code defines a `DeterministicStack` instance for
learning a `Continuous` target, and demonstrates that:

- Base models can be `Probabilistic` models even if the stack
  itself is `Deterministic` (`predict_mean` is applied in such cases).

- As an alternative to hyperparameter optimization, one can stack
  multiple copies of given model, mutating the hyper-parameter used in
  each copy.


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
                measures=rmse,
                constant=ConstantRegressor(),
                tree_2=DecisionTreeRegressor(max_depth=2),
                tree_3=DecisionTreeRegressor(max_depth=3),
                evo=EvoTreeRegressor(),
                knn=KNNRegressor(),
                xgb=XGBoostRegressor())

mach = machine(stack, X, y)
evaluate!(mach; resampling=Holdout(), measure=rmse)

```

The internal evaluation report can be accessed like this
and provides a PerformanceEvaluation object for each model:

```julia
report(mach).cv_report
```

"""
function Stack(
    ;metalearner=nothing,
    resampling=CV(),
    measure=nothing,
    measures=measure,
    cache=true,
    acceleration=CPU1(),
    named_models...
)

    metalearner === nothing &&
        throw(ArgumentError("No metalearner specified. Use Stack(metalearner=...)"))

    nt = NamedTuple(named_models)
    modelnames = keys(nt)
    models = collect(nt)
    if (measures !== nothing) && !(measures isa AbstractVector)
        measures = [measures, ]
    end

    if metalearner isa Deterministic
        stack =  DeterministicStack(
            modelnames,
            models,
            metalearner,
            resampling,
            measures,
            cache,
            acceleration
        )
    elseif metalearner isa Probabilistic
        stack = ProbabilisticStack(
            modelnames,
            models,
            metalearner,
            resampling,
            measures,
            cache,
            acceleration,
        )
    else
        throw(ArgumentError("The metalearner should be a subtype
                    of $(Union{Deterministic, Probabilistic})"))
    end
    # Issuing clean! statement
    message = MMI.clean!(stack)
    isempty(message) || @warn message

    # Warning if either input_scitype/target_scitype is
    # Unknown at construction time
    params = typeof(stack).parameters
    params[end-1] == Unknown && @warn "Could not infer input_scitype of the stack"
    params[end] == Unknown && @warn "Could not infer target_scitype of the stack"

    return stack
end


function MMI.clean!(stack::Stack{modelnames, inp_scitype, tg_scitype}) where {
    modelnames,
    inp_scitype,
    tg_scitype
}

    # We only carry checks and don't try to correct the arguments here
    message = ""
    # Checking target_scitype and input_scitype have not been changed from the original
    # stack:
    glb_inp_scitype, glb_tg_scitype =
        input_target_scitypes(getfield(stack, :models), stack.metalearner)
    glb_inp_scitype == inp_scitype ||throw(DomainError(
            inp_scitype,
            "The newly inferred input_scitype of the stack doesn't "*
            "match its original one. You have probably changed one of "*
            "the base models or the metalearner to a non compatible type."
        ))
    glb_tg_scitype == tg_scitype || throw(DomainError(
            tg_scitype,
            "The newly inferred target_scitype of the stack doesn't "*
            "match its original one. You have probably changed one of "*
            "the base model or the metalearner to a non compatible type."
        ))
    # Checking the target scitype is consistent with either Probabilistic/Deterministic
    # Stack:
    target_scitype(stack.metalearner) <: Union{
        AbstractArray{<:Union{Missing,Continuous}},
        AbstractArray{<:Union{Missing,Finite}},
    } || throw(ArgumentError(
        "The metalearner should have target_scitype: "*
        "$(Union{AbstractArray{<:Continuous}, AbstractArray{<:Finite}})"
    ))

    return message
end


Base.propertynames(::Stack{modelnames}) where modelnames =
    tuple(:metalearner, :resampling, :measures, :cache, :acceleration, modelnames...)


function Base.getproperty(stack::Stack{modelnames}, name::Symbol) where modelnames
    name === :metalearner && return getfield(stack, :metalearner)
    name === :resampling && return getfield(stack, :resampling)
    name == :measures && return getfield(stack, :measures)
    name === :cache && return getfield(stack, :cache)
    name == :acceleration && return getfield(stack, :acceleration)
    models = getfield(stack, :models)
    for j in eachindex(modelnames)
        name === modelnames[j] && return models[j]
    end
    error("type Stack has no property $name")
end


function Base.setproperty!(stack::Stack{modelnames}, _name::Symbol, val) where modelnames
    _name === :metalearner && return setfield!(stack, :metalearner, val)
    _name === :resampling && return setfield!(stack, :resampling, val)
    _name === :measures && return setfield!(stack, :measures, val)
    _name === :cache && return setfield!(stack, :cache, val)
    _name === :acceleration && return setfield!(stack, :acceleration, val)
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

pre_judge_transform(
    ŷ::Node,
    ::Type{<:Probabilistic},
    ::Type{<:AbstractArray{<:Union{Missing,Finite}}},
) =  node(ŷ -> pdf(ŷ, levels(first(ŷ))), ŷ)

pre_judge_transform(
    ŷ::Node,
    ::Type{<:Probabilistic},
    ::Type{<:AbstractArray{<:Union{Missing,Continuous}}},
) = node(ŷ->mean.(ŷ), ŷ)

pre_judge_transform(
    ŷ::Node,
    ::Type{<:Deterministic},
    ::Type{<:AbstractArray{<:Union{Missing,Continuous}}},
) = ŷ


store_for_evaluation(
    mach::Machine,
    Xtest::AbstractNode,
    ytest::AbstractNode,
    measures::Nothing,
) = nothing

store_for_evaluation(
    mach::Machine,
    Xtest::AbstractNode,
    ytest::AbstractNode,
    measures,
) = node((ytest, Xtest) -> [mach, Xtest, ytest], ytest, Xtest)


"""
    internal_stack_report(
        m::Stack,
        verbosity::Int,
        y::AbstractNode,
        folds_evaluations::Vararg{Nothing},
    )

When measure/measures is a Nothing, the folds_evaluation won't have been filled by
`store_for_evaluation` and we thus return an empty NamedTuple.

"""
internal_stack_report(
    m::Stack,
    verbosity::Int,
    tt_pairs,
    folds_evaluations::Vararg{Nothing},
) = NamedTuple{}()

"""
internal_stack_report(
    m::Stack,
    verbosity::Int,
    y::AbstractNode,
    folds_evaluations::Vararg{AbstractNode},
)

When measure/measures is provided, the folds_evaluation will have been filled by
`store_for_evaluation`. This function is not doing any heavy work (not constructing nodes
corresponding to measures) but just unpacking all the folds_evaluations in a single node
that can be evaluated later.

"""
function internal_stack_report(
    m::Stack,
    verbosity::Int,
    tt_pairs,
    folds_evaluations::Vararg{AbstractNode}
)
    _internal_stack_report(folds_evaluations...) =
        internal_stack_report(m, verbosity, tt_pairs, folds_evaluations...)
    return (report=(cv_report=node(_internal_stack_report, folds_evaluations...),),)
end

"""
    internal_stack_report(
        stack::Stack{modelnames,},
        verbosity::Int,
        y,
        folds_evaluations...
    ) where modelnames

Returns a `NamedTuple` of `PerformanceEvaluation` objects, one for each model. The
folds_evaluations are built in a flatten array respecting the order given by:
(fold_1:(model_1:[mach, Xtest, ytest], model_2:[mach, Xtest, ytest], ...), fold_2:(model_1,
model_2, ...), ...)

"""
function internal_stack_report(
    stack::Stack{modelnames,},
    verbosity::Int,
    tt_pairs,
    folds_evaluations...
) where modelnames

    n_measures = length(stack.measures)
    nfolds = length(tt_pairs)

    # For each model we record the results mimicking the fields PerformanceEvaluation
    results = NamedTuple{modelnames}(
        [(
            measure = stack.measures,
            measurement = Vector{Any}(undef, n_measures),
            operation = _actual_operations(nothing, stack.measures, model, verbosity),
            per_fold = [Vector{Any}(undef, nfolds) for _ in 1:n_measures],
            per_observation = Vector{Union{Missing, Vector{Any}}}(missing, n_measures),
            fitted_params_per_fold = [],
            report_per_fold = [],
            train_test_pairs = tt_pairs
        )
         for model in getfield(stack, :models)
         ]
    )

    # Update the results
    index = 1
    for foldid in 1:nfolds
        for modelname in modelnames
            model_results = results[modelname]
            mach, Xtest, ytest = folds_evaluations[index]
            # Update report and fitted_params per fold
            push!(model_results.fitted_params_per_fold, fitted_params(mach))
            push!(model_results.report_per_fold, report(mach))
            # Loop over measures to update per_observation and per_fold
            for (i, (measure, operation)) in enumerate(zip(
                stack.measures,
                model_results.operation,
            ))
                ypred = operation(mach, Xtest)
                loss = measure(ypred, ytest)
                # Update per_observation
                if reports_each_observation(measure)
                    if model_results.per_observation[i] === missing
                        model_results.per_observation[i] = Vector{Any}(undef, nfolds)
                    end
                    model_results.per_observation[i][foldid] = loss
                end

                # Update per_fold
                model_results.per_fold[i][foldid] =
                    reports_each_observation(measure) ?
                    MLJBase.aggregate(loss, measure) : loss
            end
            index += 1
        end
    end

    # Update measurement field by aggregation
    for modelname in modelnames
        for (i, measure) in enumerate(stack.measures)
            model_results = results[modelname]
            model_results.measurement[i] =
                MLJBase.aggregate(model_results.per_fold[i], measure)
        end
    end

    return NamedTuple{modelnames}([PerformanceEvaluation(r...) for r in results])
end


check_stack_measures(stack, verbosity::Int, measures::Nothing, y) = nothing
"""
    check_stack_measures(stack, measures, y)

Check the measures compatibility for each model in the Stack.

"""
function check_stack_measures(stack, verbosity::Int, measures, y)
    for model in getfield(stack, :models)
        operations = _actual_operations(nothing, measures, model, verbosity)
        _check_measures(measures, operations, model, y)
    end
end

"""
    oos_set(m::Stack, folds::AbstractNode, Xs::Source, ys::Source)

This function is building the out-of-sample dataset that is later used by the `judge` for
its own training. It also returns the folds_evaluations object if internal cross-validation
results are requested.

"""
function oos_set(m::Stack, Xs::Source, ys::Source, tt_pairs)
    Zval = []
    yval = []
    folds_evaluations = []
    # Loop over the cross validation folds to build a training set for the metalearner.
    for (training_rows, test_rows) in tt_pairs
        Xtrain = selectrows(Xs, training_rows)
        ytrain = selectrows(ys, training_rows)
        Xtest = selectrows(Xs, test_rows)
        ytest = selectrows(ys, test_rows)

        # Train each model on the train fold and predict on the validation fold
        # predictions are subsequently used as an input to the metalearner
        Zfold = []
        for model in getfield(m, :models)
            mach = machine(model, Xtrain, ytrain, cache=m.cache)
            ypred = predict(mach, Xtest)
            # Internal evaluation on the fold if required
            push!(folds_evaluations, store_for_evaluation(mach, Xtest, ytest, m.measures))
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

    Zval, yval, folds_evaluations
end

#######################################
################# Fit #################
#######################################
"""
    fit(m::Stack, verbosity::Int, X, y)
"""
function fit(m::Stack, verbosity::Int, X, y)
    check_stack_measures(m, verbosity, m.measures, y)
    tt_pairs = train_test_pairs(m.resampling, 1:nrows(y), X, y)

    Xs = source(X)
    ys = source(y)

    Zval, yval, folds_evaluations = oos_set(m, Xs, ys, tt_pairs)

    metamach = machine(m.metalearner, Zval, yval, cache=m.cache)

    # Each model is retrained on the original full training set
    Zpred = []
    for model in getfield(m, :models)
        mach = machine(model, Xs, ys, cache=m.cache)
        ypred = predict(mach, Xs)
        ypred = pre_judge_transform(ypred, typeof(model), target_scitype(model))
        push!(Zpred, ypred)
    end

    Zpred = MLJBase.table(hcat(Zpred...))
    ŷ = predict(metamach, Zpred)

    internal_report = internal_stack_report(m, verbosity, tt_pairs, folds_evaluations...)

    # We can infer the Surrogate by two calls to supertype
    mach = machine(supertype(supertype(typeof(m)))(), Xs, ys; predict=ŷ, internal_report...)

    return!(mach, m, verbosity, acceleration=m.acceleration)
end
