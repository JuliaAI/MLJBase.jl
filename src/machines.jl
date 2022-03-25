## MACHINE TYPE

struct NotTrainedError{M} <: Exception
    mach::M
    operation::Symbol
end

Base.showerror(io::IO, e::NotTrainedError) =
    print(io, "$(e.mach) has not been trained. "*
          "Call `fit!` on the machine, or, "*
          "if you meant to create a "*
          "learning network `Node`, "*
          "use the syntax `node($(e.operation), mach::Machine)`. ")

caches_data_by_default(::Type{<:Model}) = true
caches_data_by_default(m::M) where M<:Model = caches_data_by_default(M)

mutable struct Machine{M<:Model,C} <: MLJType

    model::M
    old_model::M # for remembering the model used in last call to `fit!`
    fitresult
    cache

    # training arguments (`Node`s or user-specified data wrapped in
    # `Source`s):
    args::Tuple{Vararg{AbstractNode}}

    # cached model-specific reformatting of args (for C=true):
    data

    # cached subsample of data (for C=true):
    resampled_data

    report
    frozen::Bool
    old_rows
    state::Int
    old_upstream_state

    # cleared by fit!(::Node) calls; put! by `fit_only!(machine, true)` calls:
    fit_okay::Channel{Bool}

    function Machine(model::M, args::AbstractNode...;
                     cache=caches_data_by_default(M)) where M<:Model
        mach = new{M,cache}(model)
        mach.frozen = false
        mach.state = 0
        mach.args = args
        mach.old_upstream_state = upstream(mach)
        mach.fit_okay = Channel{Bool}(1)
        return mach
    end

end

upstream(mach::Machine) = Tuple(m.state for m in ancestors(mach))

"""
    ancestors(mach::Machine; self=false)

All ancestors of `mach`, including `mach` if `self=true`.

"""
function ancestors(mach::Machine; self=false)
    ret = Machine[]
    self && push!(ret, mach)
    return vcat(ret, (machines(N) for N in mach.args)...) |> unique
end


## CONSTRUCTORS

# In these checks the args are abstract nodes but `full=true` only
# makes sense if they are actually source nodes.

# helper

# Here `F` is some fit_data_scitype, and so is tuple of scitypes, or a
# union of such tuples:
_contains_unknown(F) = false
_contains_unknown(F::Type{Unknown}) = true
_contains_unknown(F::Union) = any(_contains_unknown, Base.uniontypes(F))
function _contains_unknown(F::Type{<:Tuple})
    # the first line seems necessary; see https://discourse.julialang.org/t/a-union-of-tuple-types-isa-tuple-type/75339?u=ablaom
    F isa Union && return any(_contains_unknown, Base.uniontypes(F))
    return any(_contains_unknown, F.parameters)
end

warn_generic_scitype_mismatch(S, F, T) =
    "The number and/or types of data arguments do not " *
    "match what the specified model supports.\n"*
    "Run `@doc $T` to learn more about your model's requirements.\n\n"*
    "Commonly, but non exclusively, supervised models are constructed " *
    "using the syntax `machine(model, X, y)` or `machine(model, X, y, w)` " *
    "while most other models with `machine(model, X)`. " *
    "Here `X` are features, `y` a target, and `w` sample or class weights.\n\n" *
    "In general, data in `machine(model, data...)` must satisfy " *
    "`scitype(data) <: MLJ.fit_data_scitype(model)` unless the " *
    "right-hand side contains `Unknown` scitypes.\n"*
    "In the present case:\n"*
    "scitype(data) = $S\n"*
    "fit_data_scitype(model) = $F\n"

const INFO_UNKNOWN_SCITYPE =
    "Some data contains `Unknown` scitypes, which means scitype checks are skipped. "

err_length_mismatch(model) = DimensionMismatch(
    "Differing number of observations "*
    "in input and target. ")

check(model::Any, args...) =
    throw(ArgumentError("Expected a `Model` instance, got $model. "))
function check(model::Model, check_level, args...)

    is_okay = true

    check_level >= 1 || return is_okay

    F = fit_data_scitype(model)

    # if there are `Unknown`s an error will never be thrown and at
    # worst we log this fact to `Info`:
    if _contains_unknown(F)
        check_level >= 2 && @info INFO_UNKNOWN_SCITYPE
        return is_okay
    end

    # we use `elscitype` here instead of `scitype` because the data is
    # wrapped in source nodes:
    S = Tuple{elscitype.(args)...}
    if !(S <: F)
        is_okay = false
        message = warn_generic_scitype_mismatch(S, F, typeof(model))
        if check_level >= 3
            throw(ArgumentError(message))
        else
            @warn message
        end
    end

    if length(args) > 1 && is_supervised(model)
        X, y = args[1:2]

        # checks on dimension matching:
        scitype(X) == CallableReturning{Nothing} || nrows(X()) == nrows(y()) ||
            throw(err_length_mismatch(model))
    end
    return is_okay
end

"""
    machine(model, args...; cache=true, check_level)

Construct a `Machine` object binding a `model`, storing
hyper-parameters of some machine learning algorithm, to some data,
`args`. Calling `fit!` on a `Machine` object stores in the machine
object the outcomes of applying the algorithm. This in turn enables
generalization to new data using operations such as `predict` or
`transform`:

```julia
using MLJModels
X, y = make_regression()

PCA = @load PCA pkg=MultivariateStats
model = PCA()
mach = machine(model, X)
fit!(mach, rows=1:50)
transform(mach, selectrows(X, 51:100)) # or transform(mach, rows=51:100)

DecisionTreeRegressor = @load DecisionTreeRegressor pkg=DecisionTree
model = DecisionTreeRegressor()
mach = machine(model, X, y)
fit!(mach, rows=1:50)
predict(mach, selectrows(X, 51:100)) # or predict(mach, rows=51:100)
```

Specify `cache=false` to prioritize memory management over speed.

If `args` contain `Unknown` scientific types, the `machine`
constructor will not check that the specified `model` explicitly
supports that type of data, and does so silently by default. Specify
`check_level >= 2` to be notified in the event of check
skipping. Specify `check_level=3` if you wish an error to be thrown if
scientific type mismatches occur. Otherwise a warning only is
issued.

When building a learning network, `Node` objects can be substituted
for the concrete data. 

### Learning network machines

    machine(Xs; oper1=node1, oper2=node2, ...)
    machine(Xs, ys; oper1=node1, oper2=node2, ...)
    machine(Xs, ys, extras...; oper1=node1, oper2=node2, ...)

Construct a special machine called a *learning network machine*, that
wraps a learning network, usually in preparation to export the network
as a stand-alone composite model type. The keyword arguments declare
what nodes are called when operations, such as `predict` and
`transform`, are called on the machine. An advanced option allows one
to additionally pass the output of any node to the machine's report;
see below.

In addition to the operations named in the constructor, the methods
`fit!`, `report`, and `fitted_params` can be applied as usual to the
machine constructed.

    machine(Probabilistic(), args...; kwargs...)
    machine(Deterministic(), args...; kwargs...)
    machine(Unsupervised(), args...; kwargs...)
    machine(Static(), args...; kwargs...)

Same as above, but specifying explicitly the kind of model the
learning network is to meant to represent.

Learning network machines are not to be confused with an ordinary
machine that happens to be bound to a stand-alone composite model
(i.e., an *exported* learning network).


### Examples of learning network machines

Supposing a supervised learning network's final predictions are
obtained by calling a node `yhat`, then the code

```julia
mach = machine(Deterministic(), Xs, ys; predict=yhat)
fit!(mach; rows=train)
predictions = predict(mach, Xnew) # `Xnew` concrete data
```

is  equivalent to

```julia
fit!(yhat, rows=train)
predictions = yhat(Xnew)
```

Here `Xs` and `ys` are the source nodes receiving, respectively, the
input and target data.

In a unsupervised learning network for clustering, with single source
node `Xs` for inputs, and in which the node `Xout` delivers the output
of dimension reduction, and `yhat` the class labels, one can write

```julia
mach = machine(Unsupervised(), Xs; transform=Xout, predict=yhat)
fit!(mach)
transformed = transform(mach, Xnew) # `Xnew` concrete data
predictions = predict(mach, Xnew)
```

which is equivalent to

```julia
fit!(Xout)
fit!(yhat)
transformed = Xout(Xnew)
predictions = yhat(Xnew)
```
### Including a node's output in the report

The return value of a node called with no arguments can be included in
a learning network machine's report, and so in the report of any
composite model type constructed by exporting a learning network. This
is useful for exposing byproducts of network training that are not
readily deduced from the `report`s and `fitted_params` of the
component machines (which are automatically exposed).

The following example shows how to expose `err1()` and `err2()`, where
`err1` are `err2` are nodes in the network delivering training errors.

```julia
X, y = make_moons()
Xs = source(X)
ys = source(y)

model = ConstantClassifier()
mach = machine(model, Xs, ys)
yhat = predict(mach, Xs)
err1 = @node auc(yhat, ys)
err2 = @node accuracy(yhat, ys)

network_mach = machine(Probabilistic(),
                       Xs,
                       ys,
                       predict=yhat,
                       report=(auc=err1, accuracy=err2))

fit!(network_mach)
r = report(network_mach)
@assert r.auc == auc(yhat(), ys())
@assert r.accuracy == accuracy(yhat(), ys())
```
"""
function machine end

machine(T::Type{<:Model}, args...; kwargs...) =
    throw(ArgumentError("Model *type* provided where "*
                        "model *instance* expected. "))

static_error() =
    throw(ArgumentError("A `Static` transformer "*
                        "has no training arguments. "*
                        "Use `machine(model)`. "))

function machine(model::Static, args...; kwargs...)
    isempty(args) || static_error()
    return Machine(model; kwargs...)
end

function machine(model::Static, args::AbstractNode...; kwargs...)
    isempty(args) || static_error()
    return Machine(model; kwargs...)
end

machine(model::Model, raw_arg1, arg2::AbstractNode, args::AbstractNode...;
        kwargs...) =
    error("Mixing concrete data with `Node` training arguments "*
          "is not allowed. ")

machine(model::Model, arg1::AbstractNode, arg2, args...; kwargs...) =
    error("Mixing concrete data with `Node` training arguments "*
          "is not allowed. ")

function machine(model::Model,
                 raw_arg1,
                 raw_args...;
                 check_level=1,
                 kwargs...)
    args = source.((raw_arg1, raw_args...))
    check(model, check_level, args...;)
    return Machine(model, args...; kwargs...)
end

function machine(model::Model, arg1::AbstractNode, args::AbstractNode...;
                 kwargs...)
    return Machine(model, arg1, args...; kwargs...)
end


warn_bad_deserialization(state) =
    "Deserialized machine state is not -1 (got $state). "*
    "This means that the machine has not been saved by a conventional MLJ routine.\n"
    "For example, it's possible original training data is accessible from the deserialised object. "

"""
    machine(file::Union{String, IO})

Rebuild from a file a machine that has been serialized using the default
Serialization module.
"""
function machine(file::Union{String, IO})
    smach = deserialize(file)
    smach.state == -1 ||
        @warn warn_bad_deserialization(smach.state)
    restore!(smach)
    return smach
end

## INSPECTION AND MINOR MANIPULATION OF FIELDS

# Note: freeze! and thaw! are possibly not used within MLJ itself.

"""
    freeze!(mach)

Freeze the machine `mach` so that it will never be retrained (unless
thawed).

See also [`thaw!`](@ref).
"""
function freeze!(machine::Machine)
    machine.frozen = true
end

"""
    thaw!(mach)

Unfreeze the machine `mach` so that it can be retrained.

See also [`freeze!`](@ref).
"""
function thaw!(machine::Machine)
    machine.frozen = false
end

params(mach::Machine) = params(mach.model)

machines(::Source) = Machine[]


## DISPLAY

_cache_status(::Machine{<:Any,true}) = " caches data"
_cache_status(::Machine{<:Any,false}) = " does not cache data"

function Base.show(io::IO, ::MIME"text/plain", mach::Machine{M}) where M
    show(io, mach)
    print(io, " trained $(mach.state) time")
    if mach.state == 1
        print(io, ";")
    else
        print(io, "s;")
    end
    println(io, _cache_status(mach))
    println(io, "  model: $M")
    println(io, "  args: ")
    for i in eachindex(mach.args)
        arg = mach.args[i]
        print(io, "    $i:\t$arg")
        if arg isa Source
            println(io, " \u23CE `$(elscitype(arg))`")
        else
            println(io)
        end
    end
end


## FITTING

# Not one, but *two*, fit methods are defined for machines here,
# `fit!` and `fit_only!`.

# - `fit_only!`: trains a machine without touching the learned
#   parameters (`fitresult`) of any other machine. It may error if
#   another machine on which it depends (through its node training
#   arguments `N1, N2, ...`) has not been trained.

# - `fit!`: trains a machine after first progressively training all
#   machines on which the machine depends. Implicitly this involves
#   making `fit_only!` calls on those machines, scheduled by the node
#   `glb(N1, N2, ... )`, where `glb` means greatest lower bound.)


function fitlog(mach, action::Symbol, verbosity)
    if verbosity < -1000
        put!(MACHINE_CHANNEL, (action, mach))
    elseif verbosity > -1 && action == :frozen
        @warn "$mach not trained as it is frozen."
    elseif verbosity > 0
        action == :train && (@info "Training $mach."; return)
        action == :update && (@info "Updating $mach."; return)
        action == :skip && begin
            @info "Not retraining $mach. Use `force=true` to force."
            return
        end
    end
end

# for getting model specific representation of the row-restricted
# training data from a machine, according to the value of the machine
# type parameter `C` (`true` or `false`):
_resampled_data(mach::Machine{<:Model,true}, rows) = mach.resampled_data
function _resampled_data(mach::Machine{<:Model,false}, rows)
    raw_args = map(N -> N(), mach.args)
    data = MMI.reformat(mach.model, raw_args...)
    return selectrows(mach.model, rows, data...)
end

"""
    MLJBase.fit_only!(mach::Machine; rows=nothing, verbosity=1, force=false)

Without mutating any other machine on which it may depend, perform one of
the following actions to the machine `mach`, using the data and model
bound to it, and restricting the data to `rows` if specified:

- *Ab initio training.* Ignoring any previous learned parameters and
  cache, compute and store new learned parameters. Increment `mach.state`.

- *Training update.* Making use of previous learned parameters and/or
   cache, replace or mutate existing learned parameters. The effect is
   the same (or nearly the same) as in ab initio training, but may be
   faster or use less memory, assuming the model supports an update
   option (implements `MLJBase.update`). Increment `mach.state`.

- *No-operation.* Leave existing learned parameters untouched. Do not
   increment `mach.state`.


### Training action logic

For the action to be a no-operation, either `mach.frozen == true` or
or none of the following apply:

- (i) `mach` has never been trained (`mach.state == 0`).

- (ii) `force == true`.

- (iii) The `state` of some other machine on which `mach` depends has
  changed since the last time `mach` was trained (ie, the last time
  `mach.state` was last incremented).

- (iv) The specified `rows` have changed since the last retraining and
  `mach.model` does not have `Static` type.

- (v) `mach.model` has changed since the last retraining.

In any of the cases (i) - (iv), `mach` is trained ab initio. If only
(v) fails, then a training update is applied.

To freeze or unfreeze `mach`, use `freeze!(mach)` or `thaw!(mach)`.


### Implementation detail

The data to which a machine is bound is stored in `mach.args`. Each
element of `args` is either a `Node` object, or, in the case that
concrete data was bound to the machine, it is concrete data wrapped in
a `Source` node. In all cases, to obtain concrete data for actual
training, each argument `N` is called, as in `N()` or `N(rows=rows)`,
and either `MLJBase.fit` (ab initio training) or `MLJBase.update`
(training update) is dispatched on `mach.model` and this data. See the
"Adding models for general use" section of the MLJ documentation for
more on these lower-level training methods.

"""
function fit_only!(mach::Machine{<:Model,cache_data};
                   rows=nothing,
                   verbosity=1,
                   force=false) where cache_data

    if mach.frozen
        # no-op; do not increment `state`.
        fitlog(mach, :frozen, verbosity)
        return mach
    end

    # catch deserialized machines not bound to data:
    !(mach.model isa Static) && isempty(mach.args) &&
        error("This machine is not bound to any data and so "*
              "cannot be trained. ")

    # take action if model has been mutated illegally:
    warning = clean!(mach.model)
    isempty(warning) || verbosity < 0 || @warn warning

    upstream_state = upstream(mach)

    rows === nothing && (rows = (:))
    rows_is_new = !isdefined(mach, :old_rows) || rows != mach.old_rows

    condition_iv = rows_is_new && !(mach.model isa Static)

    upstream_has_changed = mach.old_upstream_state != upstream_state

    data_is_valid = isdefined(mach, :data) && !upstream_has_changed

    # build or update cached `data` if necessary:
    if cache_data && !data_is_valid
        raw_args = map(N -> N(), mach.args)
        mach.data = MMI.reformat(mach.model, raw_args...)
    end

    # build or update cached `resampled_data` if necessary:
    if cache_data && (!data_is_valid || condition_iv)
        mach.resampled_data = selectrows(mach.model, rows, mach.data...)
    end

    # `fit`, `update`, or return untouched:
    if mach.state == 0 ||       # condition (i)
        force == true ||        # condition (ii)
        upstream_has_changed || # condition (iii)
        condition_iv

        # fit the model:
        fitlog(mach, :train, verbosity)
        mach.fitresult, mach.cache, mach.report =
            try
                fit(mach.model, verbosity, _resampled_data(mach, rows)...)
            catch exception
                @error "Problem fitting the machine $mach. "
                _sources = sources(glb(mach.args...))
                length(_sources) > 2 ||
                    mach.model isa Composite ||
                    all((!isempty).(_sources)) ||
                    @warn "Some learning network source nodes are empty. "
                @info "Running type checks... "
                raw_args = map(N -> N(), mach.args)
                check_level = 1
                if check(mach.model, check_level, source.(raw_args)...)
                    @info "Type checks okay. "
                else
                    @info "It seems an upstream node in a learning "*
                        "network is providing data of incompatible scitype. See "*
                        "above. "
                end
                rethrow()
            end

    elseif mach.model != mach.old_model # condition (v)

        # update the model:
        fitlog(mach, :update, verbosity)
        mach.fitresult, mach.cache, mach.report =
            update(mach.model,
                   verbosity,
                   mach.fitresult,
                   mach.cache,
                   _resampled_data(mach, rows)...)

    else

        # don't fit the model and return without incrementing `state`:
        fitlog(mach, :skip, verbosity)
        return mach

    end

    # If we get to here it's because we have run `fit` or `update`!

    if rows_is_new
        mach.old_rows = deepcopy(rows)
    end

    mach.old_model = deepcopy(mach.model)
    mach.old_upstream_state = upstream_state
    mach.state = mach.state + 1

    return mach
end

"""

    fit!(mach::Machine, rows=nothing, verbosity=1, force=false)

Fit the machine `mach`. In the case that `mach` has `Node` arguments,
first train all other machines on which `mach` depends.

To attempt to fit a machine without touching any other machine, use
`fit_only!`. For more on the internal logic of fitting see
[`fit_only!`](@ref)

"""
function fit!(mach::Machine; kwargs...)
    glb_node = glb(mach.args...) # greatest lower bound node of arguments
    fit!(glb_node; kwargs...)
    fit_only!(mach; kwargs...)
end

# version of fit_only! for calling by scheduler (a node), which waits
# on the specified `machines` to fit:
function fit_only!(mach::Machine, wait_on_downstream::Bool; kwargs...)

    wait_on_downstream || fit_only!(mach; kwargs...)

    upstream_machines = machines(glb(mach.args...))

    # waiting on upstream machines to fit:
    for m in upstream_machines
        fit_okay = fetch(m.fit_okay)
        if !fit_okay
            put!(mach.fit_okay, false)
            return mach
        end
    end

    # try to fit this machine:
    try
        fit_only!(mach; kwargs...)
    catch e
        put!(mach.fit_okay, false)
        @error "Problem fitting $mach"
        throw(e)
    end
    put!(mach.fit_okay, true)
    return mach

end


## INSPECTION OF TRAINING OUTCOMES

"""
    fitted_params(mach)

Return the learned parameters for a machine `mach` that has been
`fit!`, for example the coefficients in a linear model.

This is a named tuple and human-readable if possible.

If `mach` is a machine for a composite model, such as a model
constructed using `@pipeline`, then the returned named tuple has the
composite type's field names as keys. The corresponding value is the
fitted parameters for the machine in the underlying learning network
bound to that model. (If multiple machines share the same model, then the
value is a vector.)

```julia
using MLJ
@load LogisticClassifier pkg=MLJLinearModels
X, y = @load_crabs;
pipe = @pipeline Standardizer LogisticClassifier
mach = machine(pipe, X, y) |> fit!

julia> fitted_params(mach).logistic_classifier
(classes = CategoricalArrays.CategoricalValue{String,UInt32}["B", "O"],
 coefs = Pair{Symbol,Float64}[:FL => 3.7095037897680405, :RW => 0.1135739140854546, :CL => -1.6036892745322038, :CW => -4.415667573486482, :BD => 3.238476051092471],
 intercept = 0.0883301599726305,)
```

Additional keys, `machines` and `fitted_params_given_machine`, give a
list of *all* machines in the underlying network, and a dictionary of
fitted parameters keyed on those machines.

```

"""
function fitted_params(mach::Machine)
    if isdefined(mach, :fitresult)
        return fitted_params(mach.model, mach.fitresult)
    else
        throw(NotTrainedError(mach, :fitted_params))
    end
end

"""
    report(mach)

Return the report for a machine `mach` that has been
`fit!`, for example the coefficients in a linear model.

This is a named tuple and human-readable if possible.

If `mach` is a machine for a composite model, such as a model
constructed using `@pipeline`, then the returned named tuple has the
composite type's field names as keys. The corresponding value is the
report for the machine in the underlying learning network
bound to that model. (If multiple machines share the same model, then the
value is a vector.)

```julia
using MLJ
@load LinearBinaryClassifier pkg=GLM
X, y = @load_crabs;
pipe = @pipeline Standardizer LinearBinaryClassifier
mach = machine(pipe, X, y) |> fit!

julia> report(mach).linear_binary_classifier
(deviance = 3.8893386087844543e-7,
 dof_residual = 195.0,
 stderror = [18954.83496713119, 6502.845740757159, 48484.240246060406, 34971.131004997274, 20654.82322484894, 2111.1294584763386],
 vcov = [3.592857686311793e8 9.122732393971942e6 … -8.454645589364915e7 5.38856837634321e6; 9.122732393971942e6 4.228700272808351e7 … -4.978433790526467e7 -8.442545425533723e6; … ; -8.454645589364915e7 -4.978433790526467e7 … 4.2662172244975924e8 2.1799125705781363e7; 5.38856837634321e6 -8.442545425533723e6 … 2.1799125705781363e7 4.456867590446599e6],)

```

Additional keys, `machines` and `report_given_machine`, give a
list of *all* machines in the underlying network, and a dictionary of
reports keyed on those machines.

```

"""
function report(mach::Machine)
    if isdefined(mach, :report)
        return mach.report
    else
        throw(NotTrainedError(mach, :report))
    end
end


"""
    training_losses(mach::Machine)

Return a list of training losses, for models that make these
available. Otherwise, returns `nothing`.

"""

function training_losses(mach::Machine)
    if isdefined(mach, :report)
        return training_losses(mach.model, mach.report)
    else
        throw(NotTrainedError(mach, :training_losses))
    end
end


###############################################################################
#####    SERIALIZABLE, RESTORE!, SAVE AND A FEW UTILITY FUNCTIONS         #####
###############################################################################


"""
    serializable(mach::Machine)

Returns a shallow copy of the machine to make it serializable. In particular,
all training data is removed and, if necessary, learned parameters are replaced
with persistent representations.

Any general purpose Julia serialization may be applied to the output of
`serializable` (eg, JLSO, BSON, JLD) but you must call `restore!(mach)` on
the deserialised object `mach` before using it. See the example below.

If using Julia's standard Serialization library, a shorter workflow is
available using the [`save`](@ref) method.

A machine returned by serializable is characterized by the property `mach.state == -1`.

### Example using [JLSO](https://invenia.github.io/JLSO.jl/stable/)

    using MLJ
    using JLSO
    Tree = @load DecisionTreeClassifier
    tree = Tree()
    X, y = @load_iris
    mach = fit!(machine(tree, X, y))

    # This machine can now be serialized
    smach = serializable(mach)
    JLSO.save("machine.jlso", machine => smach)

    # Deserialize and restore learned parameters to useable form:
    loaded_mach = JLSO.load("machine.jlso")[:machine]
    restore!(loaded_mach)

    predict(loaded_mach, X)
    predict(mach, X)
"""
function serializable(mach::Machine{<:Any, C}) where C
    # Returns a shallow copy of the machine to make it serializable, in particular:
    # - Removes all data from caches, args and data fields
    # - Makes all `fitresults` serializable
    # - Annotates the state as -1
    copymach = machine(mach.model, mach.args..., cache=C)

    for fieldname in fieldnames(Machine)
        if fieldname ∈ (:model, :report, :cache, :data, :resampled_data, :old_rows)
            continue
        elseif  fieldname == :state
            setfield!(copymach, :state, -1)
        elseif fieldname == :args
            setfield!(copymach, fieldname, ())
        # Make fitresult ready for serialization
        elseif fieldname == :fitresult
            copymach.fitresult = save(mach.model, getfield(mach, fieldname))
        else
            setfield!(copymach, fieldname, getfield(mach, fieldname))
        end
    end

    setreport!(copymach, mach.report)

    return copymach
end

"""
    restore!(mach::Machine)

Default method to restores the state of a machine that is currently serializable.
Such a machine is annotated with `state=1`
"""
function restore!(mach::Machine)
    mach.fitresult = restore(mach.model, mach.fitresult)
    mach.state = 1
    return mach
end


"""
    MLJ.save(filename, mach::Machine)
    MLJ.save(io, mach::Machine)

    MLJBase.save(filename, mach::Machine)
    MLJBase.save(io, mach::Machine)

Serialize the machine `mach` to a file with path `filename`, or to an
input/output stream `io` (at least `IOBuffer` instances are
supported) using the Serialization module.

To serialise using a different format, see `serializable`.

Machines are de-serialized using the `machine` constructor as shown in
the example below. Data (or nodes) may be optionally passed to the
constructor for retraining on new data using the saved model.


### Example

    using MLJ
    Tree = @load DecisionTreeClassifier
    X, y = @load_iris
    mach = fit!(machine(Tree(), X, y))

    MLJ.save("tree.jls", mach)
    mach_predict_only = machine("tree.jls")
    predict(mach_predict_only, X)

    mach2 = machine("tree.jls", selectrows(X, 1:100), y[1:100])
    predict(mach2, X) # same as above

    fit!(mach2) # saved learned parameters are over-written
    predict(mach2, X) # not same as above

    # using a buffer:
    io = IOBuffer()
    MLJ.save(io, mach)
    seekstart(io)
    predict_only_mach = machine(io)
    predict(predict_only_mach, X)

!!! warning "Only load files from trusted sources"
    Maliciously constructed JLS files, like pickles, and most other
    general purpose serialization formats, can allow for arbitrary code
    execution during loading. This means it is possible for someone
    to use a JLS file that looks like a serialized MLJ machine as a
    [Trojan
    horse](https://en.wikipedia.org/wiki/Trojan_horse_(computing)).

"""
function save(file::Union{String,IO},
              mach::Machine)
    isdefined(mach, :fitresult)  ||
        error("Cannot save an untrained machine. ")

    smach = serializable(mach)

    serialize(file, smach)
end


setreport!(mach::Machine, report) =
    setfield!(mach, :report, report)
