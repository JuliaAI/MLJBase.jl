## MACHINE TYPE

abstract type AbstractMachine{M<:Model} <: MLJType end

mutable struct NodalMachine{M<:Model} <: AbstractMachine{M}
    model::M
    previous_model::M # for remembering the model used in last call to `fit!`
    fitresult
    cache
    args::Tuple{Vararg{AbstractNode}}
    report
    frozen::Bool
    previous_rows   # for remembering the rows used in last call to `fit!`
    state::Int      # number of times fit! has been called on machine
    upstream_state  # for remembering the upstream state in last call to `fit!`

    function NodalMachine(model::M, args::AbstractNode...) where M<:Model

        mach = new{M}(model)
        mach.frozen = false
        mach.state = 0
        mach.args = args

        mach.upstream_state = Tuple([state(arg) for arg in args])

        return mach
    end
end


## CONSTRUCTORS

# In these checks the args are abstract nodes but `full=true` only
# makes sense if they are actually source nodes.

function check(model::Supervised, args... ; full=false)
    nargs = length(args)
    if nargs == 2
        X, y = args
    elseif nargs == 3
        supports_weights(model) ||
            @info("$(typeof(model)) does not support sample weights and " *
                  "the supplied weights will be ignored in training.\n" *
                  "However, supplied weights will be passed to " *
                  "weight-supporting measures on calls to `evaluate!` " *
                  "and in tuning. ")
        X, y, w = args
        w isa AbstractVector{<:Real} ||
            throw(ArgumentError("Weights must be real."))
        nrows(w) == nrows(y) ||
            throw(DimensionMismatch("Weights and target differ in length."))
    else
        throw(ArgumentError("Use `machine(model, X, y)` or " *
                            "`machine(model, X, y, w)` for a supervised " *
                            "model."))
    end

    if full
        # checks on input type:
        input_scitype(model) <: Unknown ||
            elscitype(X) <: input_scitype(model) ||
            @warn "The scitype of `X`, in `machine(model, X, y)` or " *
            "`machine(model, X, y, w)` is incompatible with " *
            "`model`:\nscitype(X) = $(scitype(X))\n" *
            "input_scitype(model) = $(input_scitype(model))."

        # checks on target type:
        target_scitype(model) <: Unknown ||
            elscitype(y) <: target_scitype(model) ||
            @warn "The scitype of `y`, in `machine(model, X, y)` " *
            "or `machine(model, X, y, w)` is incompatible with " *
            "`model`:\nscitype(y) = $(scitype(y))\n" *
            "target_scitype(model) = $(target_scitype(model))."

        # checks on dimension matching:
        X() === nothing || # model fits a distribution to y
            nrows(X()) == nrows(y()) ||
            throw(DimensionMismatch("Differing number of observations "*
                                    "in input and target. "))
    end
    return nothing
end

function check(model::Unsupervised, args...; full=false)
    nargs = length(args)
    nargs <= 1 ||
        throw(ArgumentError("Wrong number of arguments. Use " *
                            "`machine(model, X)` for an unsupervised model, "*
                            "or `machine(model)` if there are no training "*
                            "arguments (`Static` tranformers).) "))
    if full && nargs == 1
        X = args[1]
        # check input scitype
        input_scitype(model) <: Unknown ||
            elscitype(X) <: input_scitype(model) ||
            @warn "The scitype of `X`, in `machine(model, X)` is "*
        "incompatible with `model`:\nscitype(X) = $(scitype(X))\n" *
            "input_scitype(model) = $(input_scitype(model))."
    end
    return nothing
end

function machine(model::Static, args...)
    isempty(args) ||
        error("A `Static` transformer has no training arguments. "*
              "Use `machine(model)`. ")
    return NodalMachine(model, args...)
end

function machine(model::Supervised, X, y)
    args = (source(X), source(y, kind=:target))
    check(model, X, y; full=true)
    return NodalMachine(model, args...)
end

function machine(model::Supervised, X, y, w)
    args = (source(X), source(y, kind=:target), source(w, kind=:weights))
    check(model, X, y; full=true)
    return NodalMachine(model, args...)
end

function machine(model::Unsupervised, X)
    args = (source(X, kind=:input),)
    check(model, X, full=true)
    return NodalMachine(model, args...)
end

function machine(model::Model, args::AbstractNodes...)
    check(model, args...)
    return NodalMachine(model, args...)
end


## INSPECTION AND MINOR MANIPULATION OF FIELDS

# Note: freeze! and thaw! are possibly not used within MLJ itself.

"""
    freeze!(mach)

Freeze the machine `mach` so that it will never be retrained (unless
thawed).

See also [`thaw!`](@ref).
"""
function freeze!(machine::NodalMachine)
    machine.frozen = true
end

"""
    thaw!(mach)

Unfreeze the machine `mach` so that it can be retrained.

See also [`freeze!`](@ref).
"""
function thaw!(machine::NodalMachine)
    machine.frozen = false
end

"""
    is_stale(mach)

Check if a machine `mach` is stale.

See also [`fit!`](@ref)
"""
function is_stale(machine::NodalMachine)
    !isdefined(machine, :fitresult) ||
        machine.model != machine.previous_model ||
        reduce(|,[is_stale(arg) for arg in machine.args])
end

"""
    state(mach)

Return the state of a machine, `mach` (0 if untrained).

"""
state(machine::NodalMachine) = machine.state

params(mach::AbstractMachine) = params(mach.model)


## FITTING

"""
    fit!(mach::Machine; rows=nothing, verbosity=1, force=false)

When called for the first time, attempt to call

    MLJBase.fit(mach.model, verbosity=verbosity, args...)`

where `args` represents the training arguments of the machine. These
are either the concrete data bound to the machine at construction, or
data obtained by calling abstract nodes bound to the machine at
construction.  If `rows` is specified, it is understood that each
argument `arg` has been replaced with `selectrows(arg)`.

In the case of node arguments, the call to `fit!` will fail if an
argument of the machine depends ultimately on some other untrained
machine for successful calling. This is resolved by instead
calling `fit!` any node `N` for which `mach in machines(N)` is true,
which trains all necessary machines in an appropriate
order.

Subsequent `fit!` calls do nothing unless:

- (i) `force=true`

- (ii) Some machine on which `mach` depends has been retrained since
  `mach` was last retrained (never true if `mach` was bound to conrete
  data)

- (iii) The specified `rows` have changed since the last retraining.

- (iv) `mach` is stale (see below).

In cases (i), (ii) or (iii), `MLJBase.fit` is called again for
retraining from scratch. Otherwise `MLJBase.update` is called.

A machine `mach` is *stale* if:

-  `mach.model` has changed since the last
time a fit-result was computed.

- The machine was bound to abstract nodes at construction and one of
  these is `stale`. A node `N` is stale if `N.machine` is stale or one
  of its arguments is stale. `Source` nodes are never stale.

"""
function fit!(mach::AbstractMachine; rows=nothing, verbosity=1, force=false)

    if mach.frozen
        verbosity < 0 || @warn "$mach not trained as it is frozen."
        return mach
    end

    # catch machines deserialized and not bound to data:
    !(mach.model isa Static} && isempty(mach.args) &&
        error("This machine is not bound to any data and so "*
              "cannot be trained. ")

    warning = clean!(mach.model)
    isempty(warning) || verbosity < 0 || @warn warning

    rows === nothing && (rows = (:))

    rows_have_changed = !isdefined(mach, :previous_rows) ||
                          rows != mach.previous_rows

    # determine if concrete data to be used in training may have changed:
    upstream_state = Tuple([state(arg) for arg in mach.args])
    data_has_changed =
        rows_have_changed || (upstream_state != mach.upstream_state)
    previously_fit = (mach.state > 0)

    raw_args = [arg(rows=rows) for arg in mach.args]

    # we fit, update, or return untouched:
    if !previously_fit || data_has_changed || force
        # fit the model:
        verbosity < 1 || @info "Training $mach."
        mach.fitresult, mach.cache, mach.report =
            fit(mach.model, verbosity, raw_args...)
    elseif !is_stale(mach)
        # don't fit the model:
        if verbosity > 0
            @info "Not retraining $mach.\n It appears up-to-date. " *
                  "Use `force=true` to force retraining."
        end
        return mach
    else
        # update the model:
        verbosity < 1 || @info "Updating $mach."
        mach.fitresult, mach.cache, mach.report =
            update(mach.model,
                   verbosity,
                   mach.fitresult,
                   mach.cache,
                   raw_args...)
    end

    if rows_have_changed
        mach.previous_rows = deepcopy(rows)
    end

    mach.previous_model = deepcopy(mach.model)

    mach.upstream_state = upstream_state
    mach.state = mach.state + 1

    return mach
end


## INSPECTION OF TRAINING OUTCOMES

"""
    fitted_params(mach)

Return the learned parameters for a machine `mach` that has been
`fit!`, for example the coefficients in a linear model.

This is a named tuple and human-readable if possible.

If `mach` is a machine for a composite model, then the returned value
has keys `machines` and `fitted_params_given_machine`, whose
corresponding values are a vector of machines appearing in the
underlying learning network, and a dictionary of reports keyed on
those machines.

```julia
using MLJ
X, y = @load_crabs;
pipe = @pipeline MyPipe(
    std = Standardizer(),
    clf = @load LinearBinaryClassifier pkg=GLM
)
mach = machine(MyPipe(), X, y) |> fit!
fp = fitted_params(mach)
machs = fp.machines
2-element Array{Any,1}:
 Machine{LinearBinaryClassifier{LogitLink}} @ 1…57
 Machine{Standardizer} @ 7…33

fp.fitted_params_given_machine[machs[1]]
(coef = [121.05433477939319, 1.5863921128182814,
         61.0770377473622, -233.42699281787324, 72.74253591435117],
 intercept = 10.384459260848505,)
```

"""
function fitted_params(machine::AbstractMachine)
    if isdefined(machine, :fitresult)
        return fitted_params(machine.model, machine.fitresult)
    else
        throw(error("$machine has not been trained."))
    end
end

"""
    report(mach)

Return the report for a machine `mach` that has been
`fit!`, for example the coefficients in a linear model.

This is a named tuple and human-readable if possible.

If `mach` is a machine for a composite model, then the returned value
has keys `machines` and `report_given_machine`, whose corresponding
values are a vector of machines appearing in the underlying learning
network, and a dictionary of reports keyed on those machines.

```julia
using MLJ
X, y = @load_crabs;
pipe = @pipeline MyPipe(
    std = Standardizer(),
    clf = @load LinearBinaryClassifier pkg=GLM
)
mach = machine(MyPipe(), X, y) |> fit!
r = report(mach)
r.machines
2-element Array{Any,1}:
 Machine{LinearBinaryClassifier{LogitLink}} @ 1…57
 Machine{Standardizer} @ 7…33

r.report_given_machine[machs[1]]
(deviance = 3.8893386087844543e-7,
 dof_residual = 195.0,
 stderror = [18954.83496713119, ..., 2111.1294584763386],
 vcov = [3.592857686311793e8 ... .442545425533723e6;
         ...
         5.38856837634321e6 ... 2.1799125705781363e7 4.456867590446599e6],)
```

"""
report(mach::AbstractMachine) = mach.report


## SERIALIZATION

# saving:
"""
    MLJ.save(filename, mach::Machine; kwargs...)
    MLJ.save(io, mach::Machine; kwargs...)

    MLJBase.save(filename, mach::Machine; kwargs...)
    MLJBase.save(io, mach::Machine; kwargs...)

Serialize the machine `mach` to a file with path `filename`, or to an
input/output stream `io` (at least `IOBuffer` instances are
supported).

The format is JLSO (a wrapper for julia native or BSON serialization)
unless a custom format has been implemented for the model type of
`mach.model`. The keyword arguments `kwargs` are passed to
the format-specific serializer, which in the JSLO case include these:

keyword        | values                        | default
---------------|-------------------------------|-------------------------
`format`       | `:julia_serialize`, `:BSON`   | `:julia_serialize`
`compression`  | `:gzip`, `:none`              | `:none`

See (see
[https://github.com/invenia/JLSO.jl](https://github.com/invenia/JLSO.jl)
for details.

Machines are de-serialized using the `machine` constructor as shown in
the example below. Data (or nodes) may be optionally passed to the
constructor for retraining on new data using the saved model.


### Example

    using MLJ
    tree = @load DecisionTreeClassifier
    X, y = @load_iris
    mach = fit!(machine(tree, X, y))

    MLJ.save("tree.jlso", mach, compression=:none)
    mach_predict_only = machine("tree.jlso")
    predict(mach_predict_only, X)

    mach2 = machine("tree.jlso", selectrows(X, 1:100), y[1:100])
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
    Maliciously constructed JLSO files, like pickles, and most other
    general purpose serialization formats, can allow for arbitrary code
    execution during loading. This means it is possible for someone
    to use a JLSO file that looks like a serialized MLJ machine as a
    [Trojan
    horse](https://en.wikipedia.org/wiki/Trojan_horse_(computing)).

"""
function MMI.save(file, mach::AbstractMachine; verbosity=1, kwargs...)
    isdefined(mach, :fitresult)  ||
        error("Cannot save an untrained machine. ")
    MMI.save(file, mach.model, mach.fitresult, mach.report; kwargs...)
end

# restoring:
function machine(file::Union{String,IO}, args...; kwargs...)
    model, fitresult, report = MMI.restore(file; kwargs...)
    isempty(args) || check(model, args...)
    mach = NodalMachine(model, args...)
    mach.fitresult = fitresult
    mach.report = report
    return mach
end
