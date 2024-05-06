## SCITYPE CHECK LEVEL

"""
    default_scitype_check_level()

Return the current global default value for scientific type checking
when constructing machines.

    default_scitype_check_level(i::Integer)

Set the global default value for scientific type checking to `i`.

The effect of the `scitype_check_level` option in calls of the form
`machine(model, data, scitype_check_level=...)` is summarized below:

`scitype_check_level` | Inspect scitypes? | If `Unknown` in scitypes | If other scitype mismatch |
|:--------------------|:-----------------:|:------------------------:|:-------------------------:|
0                     | ×                 |                          |                           |
1 (value at startup)  | ✓                 |                          | warning                   |
2                     | ✓                 | warning                  | warning                   |
3                     | ✓                 | warning                  | error                     |
4                     | ✓                 | error                    | error                     |

See also [`machine`](@ref)

"""
function default_scitype_check_level end
default_scitype_check_level() = DEFAULT_SCITYPE_CHECK_LEVEL[]
default_scitype_check_level(i) = (DEFAULT_SCITYPE_CHECK_LEVEL[] = i;)


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

caches_data_by_default(m) = caches_data_by_default(typeof(m))
caches_data_by_default(::Type) = true
caches_data_by_default(::Type{<:Symbol}) = false

mutable struct Machine{M,OM,C} <: MLJType

    model::M
    old_model::OM  # for remembering the model used in last call to `fit!`

    # the next two refer to objects returned by `MLJModlelInterface.fit(::M, ...)`.
    fitresult
    cache # relevant to `MLJModelInterface.update`, not to be confused with type param `C`

    # training arguments (`Node`s or user-specified data wrapped in
    # `Source`s):
    args::Tuple{Vararg{AbstractNode}}

    # cached model-specific reformatting of args (for C=true):
    data

    # cached subsample of data (for C=true):
    resampled_data

    # dictionary of named tuples keyed on method (:fit, :predict, etc):
    report
    frozen::Bool
    old_rows
    state::Int
    old_upstream_state

    # cleared by fit!(::Node) calls; put! by `fit_only!(machine, true)` calls:
    fit_okay::Channel{Bool}

    function Machine(
        model::M, args::AbstractNode...;
        cache=caches_data_by_default(model),
        ) where M
        # In the case of symbolic model, machine cannot know the type of model to be fit
        # at time of construction:
        OM = M == Symbol ? Any : M
        mach = new{M,OM,cache}(model) # (this `cache` is not the *field* `cache`)
        mach.frozen = false
        mach.state = 0
        mach.args = args
        mach.old_upstream_state = upstream(mach)
        mach.fit_okay = Channel{Bool}(1)
        return mach
    end

end

caches_data(::Machine{<:Any, <:Any, C}) where C = C

"""
    age(mach::Machine)

Return an integer representing the number of times `mach` has been trained or updated. For
more detail, see the discussion of training logic at [`fit_only!`](@ref).

"""
age(mach::Machine) = mach.state

"""
    replace(mach::Machine, field1 => value1, field2 => value2, ...)

**Private method.**

Return a shallow copy of the machine `mach` with the specified field
replacements. Undefined field values are preserved. Unspecified fields have identically
equal values, with the exception of `mach.fit_okay`, which is always a new instance
`Channel{Bool}(1)`.

The following example returns a machine with no traces of training data (but also removes
any upstream dependencies in a learning network):

```julia
replace(mach, :args => (), :data => (), :data_resampled_data => (), :cache => nothing)
```
"""
function Base.replace(mach::Machine{<:Any,<:Any,C}, field_value_pairs::Pair...) where C
    # determined new `model` and `args` and build replacement dictionary:
    newfield_given_old = Dict(field_value_pairs) # to be extended
    fields_to_be_replaced = keys(newfield_given_old)
    :fit_okay in fields_to_be_replaced && error("Cannot replace `:fit_okay` field. ")
    newmodel = :model in fields_to_be_replaced ? newfield_given_old[:model] : mach.model
    newargs = :args in fields_to_be_replaced ? newfield_given_old[:args] : mach.args

    # instantiate a new machine and make field replacements:
    clone = Machine(newmodel, newargs...; cache=C)
    for field in fieldnames(typeof(mach))
        if !(field in fields_to_be_replaced || isdefined(mach, field)) ||
             field in [:model, :args, :fit_okay]
            continue
        end
        value = field in fields_to_be_replaced ? newfield_given_old[field] :
            getproperty(mach, field)
        setproperty!(clone, field, value)
    end
    return clone
end

Base.copy(mach::Machine) = replace(mach)

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


# # CONSTRUCTORS

# In the checks `args` is expected to be `Vector{<:AbstractNode}` (eg, a vector of source
# nodes) not raw data.

# # Helpers

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

alert_generic_scitype_mismatch(S, F, T) =
    """
    The number and/or types of data arguments do not match what the specified model
    supports. Suppress this type check by specifying `scitype_check_level=0`.

    Run `@doc $(package_name(T)).$(name(T))` to learn more about your model's requirements.

    Commonly, but non exclusively, supervised models are constructed using the syntax
    `machine(model, X, y)` or `machine(model, X, y, w)` while most other models are
    constructed with `machine(model, X)`.  Here `X` are features, `y` a target, and `w`
    sample or class weights.

    In general, data in `machine(model, data...)` is expected to satisfy

        scitype(data) <: MLJ.fit_data_scitype(model)

    In the present case:

    scitype(data) = $S

    fit_data_scitype(model) = $F
    """

const WARN_UNKNOWN_SCITYPE =
    "Some data contains `Unknown` scitypes, which might lead to model-data mismatches. "

err_length_mismatch(model) = DimensionMismatch(
    "Differing number of observations in input and target. ")

function check(model::Model, scitype_check_level, args...)

    check_ismodel(model)

    is_okay = true

    scitype_check_level >= 1 || return is_okay

    F = fit_data_scitype(model)

    if _contains_unknown(F)
        scitype_check_level in [2, 3] && @warn WARN_UNKNOWN_SCITYPE
        scitype_check_level >= 4 && throw(ArgumentError(WARN_UNKNOWN_SCITYPE))
        return is_okay
    end

    # Sometimes (X, ) is a table, when X is a table, which leads to scitype((X,)) =
    # Table(...) where `Tuple{scitype(X)}` is wanted.  Also, we use `elscitype` here
    # instead of `scitype` because the data is wrapped in source nodes;
    S = Tuple{elscitype.(args)...}
    if !(S <: F)
        is_okay = false
        message = alert_generic_scitype_mismatch(S, F, typeof(model))
        if scitype_check_level >= 3
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


# # Constructors

"""
    machine(model, args...; cache=true, scitype_check_level=1)

Construct a `Machine` object binding a `model`, storing
hyper-parameters of some machine learning algorithm, to some data,
`args`. Calling [`fit!`](@ref) on a `Machine` instance `mach` stores
outcomes of applying the algorithm in `mach`, which can be inspected
using `fitted_params(mach)` (learned paramters) and `report(mach)`
(other outcomes). This in turn enables generalization to new data
using operations such as `predict` or `transform`:

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

When building a learning network, `Node` objects can be substituted
for the concrete data but no type or dimension checks are applied.

### Checks on the types of training data

A model articulates its data requirements using [scientific
types](https://juliaai.github.io/ScientificTypes.jl/dev/), i.e.,
using the [`scitype`](@ref) function instead of the `typeof` function.

If `scitype_check_level > 0` then the scitype of each `arg` in `args`
is computed, and this is compared with the scitypes expected by the
model, unless `args` contains `Unknown` scitypes and
`scitype_check_level < 4`, in which case no further action is
taken. Whether warnings are issued or errors thrown depends the
level. For details, see [`default_scitype_check_level`](@ref), a method
to inspect or change the default level (`1` at startup).

### Machines with model placeholders

A symbol can be substituted for a model in machine constructors to act as a placeholder
for a model specified at training time. The symbol must be the field name for a struct
whose corresponding value is a model, as shown in the following example:

```julia
mutable struct MyComposite
    transformer
    classifier
end

my_composite = MyComposite(Standardizer(), ConstantClassifier)

X, y = make_blobs()
mach = machine(:classifier, X, y)
fit!(mach, composite=my_composite)
```

The last two lines are equivalent to

```julia
mach = machine(ConstantClassifier(), X, y)
fit!(mach)
```

Delaying model specification is used when exporting learning networks as new stand-alone
model types. See [`prefit`](@ref) and the MLJ documentation on learning networks.


See also [`fit!`](@ref), [`default_scitype_check_level`](@ref),
[`MLJBase.save`](@ref), [`serializable`](@ref).

"""
function machine end

const ERR_STATIC_ARGUMENTS = ArgumentError(
    "A `Static` transformer "*
    "has no training arguments. "*
    "Use `machine(model)`. "
)

machine(T::Type{<:Model}, args...; kwargs...) =
    throw(ArgumentError("Model *type* provided where "*
                        "model *instance* expected. "))

function machine(model::Static, args...; cache=false, kwargs...)
    isempty(args) || throw(ERR_STATIC_ARGUMENTS)
    return Machine(model; cache=false, kwargs...)
end

function machine(
    model::Static,
    args::AbstractNode...;
    cache=false,
    kwargs...,
)
    isempty(args) || model isa Symbol || throw(ERR_STATIC_ARGUMENTS)
    mach = Machine(model; cache=false, kwargs...)
    return mach
end

machine(model::Symbol; cache=false, kwargs...) =
    Machine(model; cache, kwargs...)

machine(model::Union{Model,Symbol}, raw_arg1, arg2::AbstractNode, args::AbstractNode...;
        kwargs...) =
            error("Mixing concrete data with `Node` training arguments "*
                  "is not allowed. ")

function machine(
    model::Union{Model,Symbol},
    raw_arg1,
    raw_args...;
    scitype_check_level=default_scitype_check_level(),
    kwargs...,
)
    args = source.((raw_arg1, raw_args...))
    model isa Symbol || check(model, scitype_check_level, args...;)
    return Machine(model, args...; kwargs...)
end

function machine(model::Union{Model,Symbol}, arg1::AbstractNode, args::AbstractNode...;
                 kwargs...)
    return Machine(model, arg1, args...; kwargs...)
end

function machine(model::Symbol, arg1::AbstractNode, args::AbstractNode...;
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

_cache_status(::Machine{<:Any,<:Any,true}) = "caches model-specific representations of data"
_cache_status(::Machine{<:Any,<:Any,false}) = "does not cache data"

function Base.show(io::IO, mach::Machine)
    model = mach.model
    m = model isa Symbol ? ":$model" : model
    print(io, "machine($m, …)")
end

function Base.show(io::IO, ::MIME"text/plain", mach::Machine{M}) where M
    header =
        mach.state == -1 ? "serializable " :
        mach.state ==  0 ? "untrained " :
        "trained "
    header *= "Machine"
    mach.state >= 0 && (header *= "; "*_cache_status(mach))
    println(io, header)
    println(io, "  model: $(mach.model)")
    println(io, "  args: ")
    for i in eachindex(mach.args)
        arg = mach.args[i]
        print(io, "    $i:\t$arg")
        if arg isa Source
            println(io, " \u23CE $(elscitype(arg))")
        else
            println(io)
        end
    end
end


## FITTING

# Not one, but *two*, fit methods are defined for machines here,
# `fit!` and `fit_only!`.

# - `fit_only!`: trains a machine without touching the learned parameters (`fitresult`) of
#   any other machine. It may error if another machine on which it depends (through its node
#   training arguments `N1, N2, ...`) has not been trained. It's possible that a dependent
#   machine `mach` may have it's report mutated if `reporting_operations(mach.model)` is
#   non-empty.

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
_resampled_data(mach::Machine{<:Any,<:Any,true}, model, rows) = mach.resampled_data
function _resampled_data(mach::Machine{<:Any,<:Any,false}, model, rows)
    raw_args = map(N -> N(), mach.args)
    data = MMI.reformat(model, raw_args...)
    return selectrows(model, rows, data...)
end

err_no_real_model(mach) = ErrorException(
    """

    Cannot train or use $mach, which has a `Symbol` as model. Perhaps you
    forgot to specify `composite=... ` in a `fit!` call?

    """
)

err_missing_model(model) = ErrorException(
    "Specified `composite` model does not have `:$(model)` as a field."
)

"""
   last_model(mach::Machine)

Return the last model used to train the machine `mach`. This is a bona fide model, even if
`mach.model` is a symbol.

Returns `nothing` if `mach` has not been trained.
"""
last_model(mach) = isdefined(mach, :old_model) ? mach.old_model : nothing

"""
    MLJBase.fit_only!(
        mach::Machine;
        rows=nothing,
        verbosity=1,
        force=false,
        composite=nothing,
    )

Without mutating any other machine on which it may depend, perform one of the following
actions to the machine `mach`, using the data and model bound to it, and restricting the
data to `rows` if specified:

- *Ab initio training.* Ignoring any previous learned parameters and
  cache, compute and store new learned parameters. Increment `mach.state`.

- *Training update.* Making use of previous learned parameters and/or
   cache, replace or mutate existing learned parameters. The effect is
   the same (or nearly the same) as in ab initio training, but may be
   faster or use less memory, assuming the model supports an update
   option (implements `MLJBase.update`). Increment `mach.state`.

- *No-operation.* Leave existing learned parameters untouched. Do not
   increment `mach.state`.

If the model, `model`, bound to `mach` is a symbol, then instead perform the action using
the true model given by `getproperty(composite, model)`. See also [`machine`](@ref).


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

- (v) `mach.model` is a model and different from the last model used for training, but has
  the same type.

- (vi) `mach.model` is a model but has a type different from the last model used for
  training.

- (vii) `mach.model` is a symbol and `(composite, mach.model)` is different from the last
  model used for training, but has the same type.

- (viii) `mach.model` is a symbol and `(composite, mach.model)` has a different type from
  the last model used for training.

In any of the cases (i) - (iv), (vi), or (viii), `mach` is trained ab initio. If (v) or
(vii) is true, then a training update is applied.

To freeze or unfreeze `mach`, use `freeze!(mach)` or `thaw!(mach)`.


### Implementation details

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
function fit_only!(
    mach::Machine{<:Any,<:Any,cache_data};
    rows=nothing,
    verbosity=1,
    force=false,
    composite=nothing,
) where cache_data

    if mach.frozen
        # no-op; do not increment `state`.
        fitlog(mach, :frozen, verbosity)
        return mach
    end

    # catch deserialized machines not bound to data:
    if isempty(mach.args) && !(mach.model isa Static) && !(mach.model isa Symbol)
        error("This machine is not bound to any data and so "*
              "cannot be trained. ")
    end

    # If `mach.model` is a symbol, then we want to replace it with the bone fide model
    # `getproperty(composite, mach.model)`:
    model = if mach.model isa Symbol
        isnothing(composite) && throw(err_no_real_model(mach))
        mach.model in propertynames(composite) ||
            throw(err_missing_model(model))
        getproperty(composite, mach.model)
    else
        mach.model
    end

    modeltype_changed = !isdefined(mach, :old_model) ? true  :
            typeof(model) === typeof(mach.old_model) ? false :
                                                       true

    # take action if model has been mutated illegally:
    warning = clean!(model)
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
        mach.data = MMI.reformat(model, raw_args...)
    end

    # build or update cached `resampled_data` if necessary (`mach.data` is already defined
    # above if needed here):
    if cache_data && (!data_is_valid || condition_iv)
        mach.resampled_data = selectrows(model, rows, mach.data...)
    end

    # `fit`, `update`, or return untouched:
    if mach.state == 0 ||       # condition (i)
        force == true ||        # condition (ii)
        upstream_has_changed || # condition (iii)
        condition_iv ||         # condition (iv)
        modeltype_changed       # conditions (vi) or (vii)

        isdefined(mach, :report) || (mach.report = LittleDict{Symbol,Any}())

        # fit the model:
        fitlog(mach, :train, verbosity)
        mach.fitresult, mach.cache, mach.report[:fit] =
            try
                fit(model, verbosity, _resampled_data(mach, model, rows)...)
            catch exception
                @error "Problem fitting the machine $mach. "
                _sources = sources(glb(mach.args...))
                length(_sources) > 2 ||
                    all((!isempty).(_sources)) ||
                    @warn "Some learning network source nodes are empty. "
                @info "Running type checks... "
                raw_args = map(N -> N(), mach.args)
                scitype_check_level = 1
                if check(model, scitype_check_level, source.(raw_args)...)
                    @info "Type checks okay. "
                else
                    @info "It seems an upstream node in a learning "*
                        "network is providing data of incompatible scitype. See "*
                        "above. "
                end
                rethrow()
            end

    elseif model != mach.old_model # condition (v)

        # update the model:
        fitlog(mach, :update, verbosity)
        mach.fitresult, mach.cache, mach.report[:fit] =
            update(model,
                   verbosity,
                   mach.fitresult,
                   mach.cache,
                   _resampled_data(mach, model, rows)...)

    else

        # don't fit the model and return without incrementing `state`:
        fitlog(mach, :skip, verbosity)
        return mach

    end

    # If we get to here it's because we have run `fit` or `update`!

    if rows_is_new
        mach.old_rows = deepcopy(rows)
    end

    mach.old_model = deepcopy(model)
    mach.old_upstream_state = upstream_state
    mach.state = mach.state + 1

    return mach
end

# version of fit_only! for calling by scheduler (a node), which waits on all upstream
# `machines` to fit:
function fit_only!(mach::Machine, wait_on_upstream::Bool; kwargs...)

    wait_on_upstream || fit_only!(mach; kwargs...)

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

"""

    fit!(mach::Machine, rows=nothing, verbosity=1, force=false, composite=nothing)

Fit the machine `mach`. In the case that `mach` has `Node` arguments,
first train all other machines on which `mach` depends.

To attempt to fit a machine without touching any other machine, use
`fit_only!`. For more on options and the the internal logic of fitting see
[`fit_only!`](@ref)

"""
function fit!(mach::Machine; kwargs...)
    glb_node = glb(mach.args...) # greatest lower bound node of arguments
    fit!(glb_node; kwargs...)
    fit_only!(mach; kwargs...)
end


## INSPECTION OF TRAINING OUTCOMES

"""
    fitted_params(mach)

Return the learned parameters for a machine `mach` that has been
`fit!`, for example the coefficients in a linear model.

This is a named tuple and human-readable if possible.

If `mach` is a machine for a composite model, such as a model constructed using the
pipeline syntax `model1 |> model2 |> ...`, then the returned named tuple has the composite
type's field names as keys. The corresponding value is the fitted parameters for the
machine in the underlying learning network bound to that model. (If multiple machines
share the same model, then the value is a vector.)

```julia-repl
julia> using MLJ
julia> @load LogisticClassifier pkg=MLJLinearModels
julia> X, y = @load_crabs;
julia> pipe = Standardizer() |> LogisticClassifier();
julia> mach = machine(pipe, X, y) |> fit!;

julia> fitted_params(mach).logistic_classifier
(classes = CategoricalArrays.CategoricalValue{String,UInt32}["B", "O"],
 coefs = Pair{Symbol,Float64}[:FL => 3.7095037897680405, :RW => 0.1135739140854546, :CL => -1.6036892745322038, :CW => -4.415667573486482, :BD => 3.238476051092471],
 intercept = 0.0883301599726305,)
```

See also [`report`](@ref)

"""
function fitted_params(mach::Machine)
    if isdefined(mach, :fitresult)
        return fitted_params(last_model(mach), mach.fitresult)
    else
        throw(NotTrainedError(mach, :fitted_params))
    end
end

"""
    report(mach)

Return the report for a machine `mach` that has been
`fit!`, for example the coefficients in a linear model.

This is a named tuple and human-readable if possible.

If `mach` is a machine for a composite model, such as a model constructed using the
pipeline syntax `model1 |> model2 |> ...`, then the returned named tuple has the composite
type's field names as keys. The corresponding value is the report for the machine in the
underlying learning network bound to that model. (If multiple machines share the same
model, then the value is a vector.)

```julia-repl
julia> using MLJ
julia> @load LinearBinaryClassifier pkg=GLM
julia> X, y = @load_crabs;
julia> pipe = Standardizer() |> LinearBinaryClassifier();
julia> mach = machine(pipe, X, y) |> fit!;

julia> report(mach).linear_binary_classifier
(deviance = 3.8893386087844543e-7,
 dof_residual = 195.0,
 stderror = [18954.83496713119, 6502.845740757159, 48484.240246060406, 34971.131004997274, 20654.82322484894, 2111.1294584763386],
 vcov = [3.592857686311793e8 9.122732393971942e6 … -8.454645589364915e7 5.38856837634321e6; 9.122732393971942e6 4.228700272808351e7 … -4.978433790526467e7 -8.442545425533723e6; … ; -8.454645589364915e7 -4.978433790526467e7 … 4.2662172244975924e8 2.1799125705781363e7; 5.38856837634321e6 -8.442545425533723e6 … 2.1799125705781363e7 4.456867590446599e6],)

```

See also [`fitted_params`](@ref)

"""
function report(mach::Machine)
    if isdefined(mach, :report)
        return MMI.report(last_model(mach), mach.report)
    else
        throw(NotTrainedError(mach, :report))
    end
end

"""

    report_given_method(mach::Machine)

Same as `report(mach)` but broken down by the method (`fit`, `predict`, etc) that
contributed the report.

A specialized method intended for learning network applications.

The return value is a dictionary keyed on the symbol representing the method (`:fit`,
`:predict`, etc) and the values report contributed by that method.

"""
report_given_method(mach::Machine) = mach.report



"""
    training_losses(mach::Machine)

Return a list of training losses, for models that make these
available. Otherwise, return `nothing`.

"""

function training_losses(mach::Machine)
    if isdefined(mach, :report)
        return training_losses(last_model(mach), report_given_method(mach)[:fit])
    else
        throw(NotTrainedError(mach, :training_losses))
    end
end

"""
    feature_importances(mach::Machine)

Return a list of `feature => importance` pairs for a fitted machine, `mach`, for supported
models. Otherwise return `nothing`.

"""
function feature_importances(mach::Machine)
    if isdefined(mach, :report) && isdefined(mach, :fitresult)
        return _feature_importances(
            last_model(mach),
            mach.fitresult,
            report_given_method(mach)[:fit],
        )
    else
        throw(NotTrainedError(mach, :feature_importances))
    end
end

function _feature_importances(model, fitresult, report)
    if reports_feature_importances(model)
        return MMI.feature_importances(model, fitresult, report)
    else
        return nothing
    end
end
###############################################################################
#####    SERIALIZABLE, RESTORE!, SAVE AND A FEW UTILITY FUNCTIONS         #####
###############################################################################

const ERR_SERIALIZING_UNTRAINED = ArgumentError(
    "`serializable` called on untrained machine. "
)

"""
    serializable(mach::Machine)

Returns a shallow copy of the machine to make it serializable. In particular,
all training data is removed and, if necessary, learned parameters are replaced
with persistent representations.

Any general purpose Julia serializer may be applied to the output of
`serializable` (eg, JLSO, BSON, JLD) but you must call
`restore!(mach)` on the deserialised object `mach` before using
it. See the example below.

If using Julia's standard Serialization library, a shorter workflow is
available using the [`MLJBase.save`](@ref) (or `MLJ.save`) method.

A machine returned by `serializable` is characterized by the property
`mach.state == -1`.

### Example using [JLSO](https://invenia.github.io/JLSO.jl/stable/)
```julia
using MLJ
using JLSO
Tree = @load DecisionTreeClassifier
tree = Tree()
X, y = @load_iris
mach = fit!(machine(tree, X, y))

# This machine can now be serialized
smach = serializable(mach)
JLSO.save("machine.jlso", :machine => smach)

# Deserialize and restore learned parameters to useable form:
loaded_mach = JLSO.load("machine.jlso")[:machine]
restore!(loaded_mach)

predict(loaded_mach, X)
predict(mach, X)
```
See also [`restore!`](@ref), [`MLJBase.save`](@ref).

"""
function serializable(mach::Machine{<:Any,<:Any,C}, model=mach.model; verbosity=1) where C

    isdefined(mach, :fitresult) || throw(ERR_SERIALIZING_UNTRAINED)
    mach.state == -1 && return mach

    # The next line of code makes `serializable` recursive, in the case that `mach.model`
    # is a `Composite` model: `save` duplicates the underlying learning network, which
    # involves calls to `serializable` on the old machines in the network to create the
    # new ones.

    serializable_fitresult = save(model, mach.fitresult)

    # Duplication currenty needs to happen in two steps for this to work in case of
    # `Composite` models.

    clone = replace(
        mach,
        :state => -1,
        :args => (),
        :fitresult => serializable_fitresult,
        :old_rows => nothing,
        :data => nothing,
        :resampled_data => nothing,
        :cache => nothing,
    )

    report = report_for_serialization(clone)

    return replace(clone, :report => report)
end

"""
    restore!(mach::Machine)

Restore the state of a machine that is currently serializable but
which may not be otherwise usable. For such a machine, `mach`, one has
`mach.state=1`. Intended for restoring deserialized machine objects to a
useable form.

For an example see [`serializable`](@ref).

"""
function restore!(mach::Machine, model=mach.model)
    mach.state != -1 && return mach
    mach.fitresult = restore(model, mach.fitresult)
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

To serialise using a different format, see [`serializable`](@ref).

Machines are deserialized using the `machine` constructor as shown in
the example below.

> The implementation of `save` for machines changed in MLJ 0.18
>  (MLJBase 0.20). You can only restore a machine saved using older
>  versions of MLJ using an older version.

### Example

```julia
using MLJ
Tree = @load DecisionTreeClassifier
X, y = @load_iris
mach = fit!(machine(Tree(), X, y))

MLJ.save("tree.jls", mach)
mach_predict_only = machine("tree.jls")
predict(mach_predict_only, X)

# using a buffer:
io = IOBuffer()
MLJ.save(io, mach)
seekstart(io)
predict_only_mach = machine(io)
predict(predict_only_mach, X)
```

!!! warning "Only load files from trusted sources"
    Maliciously constructed JLS files, like pickles, and most other
    general purpose serialization formats, can allow for arbitrary code
    execution during loading. This means it is possible for someone
    to use a JLS file that looks like a serialized MLJ machine as a
    [Trojan
    horse](https://en.wikipedia.org/wiki/Trojan_horse_(computing)).

See also [`serializable`](@ref), [`machine`](@ref).

"""
function save(file::Union{String,IO}, mach::Machine)
    isdefined(mach, :fitresult)  ||
        error("Cannot save an untrained machine. ")

    smach = serializable(mach)

    serialize(file, smach)
end

report_for_serialization(mach) = mach.report

# NOTE. there is also a specialization of `report_for_serialization` for `Composite`
# models, defined in /src/composition/learning_networks/machines/
