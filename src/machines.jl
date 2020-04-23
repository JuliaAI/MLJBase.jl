abstract type AbstractMachine{M<:Model} <: MLJType end

mutable struct Machine{M<:Model} <: AbstractMachine{M}
    model::M
    previous_model::M
    fitresult
    cache
    args::Tuple
    report
    previous_rows
    # constructor
    function Machine{M}(model, args...) where M
        machine = new{M}(model)
        machine.args = args
        return machine
    end
end


## CONSTRUCTORS

Machine(model::M, args...) where M <: Model = Machine{M}(model, args...)

function check(model::Supervised, args...)
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

    # checks on input type:
    input_scitype(model) <: Unknown ||
        scitype(X) <: input_scitype(model) ||
        @warn "The scitype of `X`, in `machine(model, X, y)` or " *
        "`machine(model, X, y, w)` is incompatible with " *
        "`model`:\nscitype(X) = $(scitype(X))\n" *
        "input_scitype(model) = $(input_scitype(model))."

    # checks on target type:
    target_scitype(model) <: Unknown ||
        scitype(y) <: target_scitype(model) ||
        @warn "The scitype of `y`, in `machine(model, X, y)` " *
        "or `machine(model, X, y, w)` is incompatible with " *
        "`model`:\nscitype(y) = $(scitype(y))\n" *
        "target_scitype(model) = $(target_scitype(model))."

    # checks on dimension matching:
    nrows(X) == nrows(y) ||
        throw(DimensionMismatch("Differing number of observations "*
                                "in input and target. "))
    return nothing
end

function check(model::Unsupervised, args...)
    nargs = length(args)
    nargs <= 1 ||
        throw(ArgumentError("Wrong number of arguments. Use " *
                            "`machine(model, X)` for an unsupervised model "*
                            "(or `Machine(model)` if there are no training "*
                            "arguments (\"static\" tranformers"))
    if nargs == 1
        X = args[1]
        # check input scitype
        input_scitype(model) <: Unknown ||
            scitype(X) <: input_scitype(model) ||
            @warn "The scitype of `X`, in `machine(model, X)` is "*
        "incompatible with `model`:\nscitype(X) = $(scitype(X))\n" *
            "input_scitype(model) = $(input_scitype(model))."
    end
    return nothing
end

function machine(model::Model, args...)
    # isempty(args) &&
    #     error("`machine(model)` is ambiguous. Use `Machine(model)` or "*
    #           "`NodalMachine(model)` (for machines in a learning network). ")
    check(model, args...)
    return Machine(model, args...)
end

# machine(model::Static, args...) = Machine(model, args...)

# Note: The following method is written to fit `NodalMachine`s
# defined in networks.jl, in addition to `Machine`s defined above.

"""
    fit!(mach::Machine; rows=nothing, verbosity=1, force=false)

When called for the first time, call

    fit(mach.model, verbosity, mach.args...)

storing the returned fit-result and report in `mach`. Subsequent calls
do nothing unless: (i) `force=true`, or (ii) the specified `rows` are
different from those used the last time a fit-result was computed, or
(iii) `mach.model` has changed since the last time a fit-result was
computed (the machine is *stale*). In cases (i) or (ii) `MLJBase.fit`
is called again. Otherwise, `MLJBase.update` is called.

    fit!(mach::NodalMachine; rows=nothing, verbosity=1, force=false)

When called for the first time, attempt to call

    fit(mach.model, verbosity, mach.args...)

This will fail if an argument of the machine depends ultimately on
some other untrained machine for successful calling, but this is
resolved by instead calling `fit!` any node `N` for which
`mach in machines(N)` is true, which trains all necessary machines in
an appropriate order. Subsequent `fit!` calls do nothing unless: (i)
`force=true`, or (ii) some machine on which `mach` depends has
computed a new fit-result since `mach` last computed its fit-result,
or (iii) the specified `rows` have changed since the last time a
fit-result was last computed, or (iv) `mach` is stale (see below). In
cases (i), (ii) or (iii), `MLJBase.fit` is called. Otherwise
`MLJBase.update` is called.

A machine `mach` is *stale* if `mach.model` has changed since the last
time a fit-result was computed, or if one of its training arguments
is `stale`. A node `N` is stale if `N.machine` is stale or one of its
arguments is stale. `Source` nodes are never stale.

Note that a nodal machine obtains its training data by *calling* its
node arguments on the specified `rows` (rather than *indexing* its arguments
on those rows) and that this calling is a recursive operation on nodes
upstream of those arguments.

"""
function fit!(mach::AbstractMachine; rows=nothing, verbosity=1, force=false)

    if mach isa NodalMachine && mach.frozen
        verbosity < 0 || @warn "$mach not trained as it is frozen."
        return mach
    end

    # catch machines with no arguments (no data):
    mach isa Machine{<:Supervised} ||
        mach isa Machine{<:Unsupervised} &&
        !(mach isa Machine{<:Static}) &&
        isempty(mach.args) &&
        error("This machine is not bound to any data and so "*
              "cannot be trained. ")

    warning = clean!(mach.model)
    isempty(warning) || verbosity < 0 || @warn warning

    rows === nothing && (rows = (:))

    rows_have_changed = !isdefined(mach, :previous_rows) ||
                          rows != mach.previous_rows

    if mach isa NodalMachine
        # determine if concrete data to be used in training may have changed:
        upstream_state = Tuple([state(arg) for arg in mach.args])
        data_has_changed =
            rows_have_changed || (upstream_state != mach.upstream_state)
        previously_fit = (mach.state > 0)
        args = [arg(rows=rows) for arg in mach.args]
    else
        data_has_changed = rows_have_changed
        previously_fit = isdefined(mach, :fitresult)
        args = [selectrows(arg, rows) for arg in mach.args]
    end

    if !previously_fit || data_has_changed || force
        # fit the model:
        verbosity < 1 || @info "Training $mach."
        mach.fitresult, mach.cache, mach.report =
            fit(mach.model, verbosity, args...)

    elseif !is_stale(mach)
        # don't fit the model
        if verbosity > 0
            @info "Not retraining $mach.\n It appears up-to-date. " *
                  "Use `force=true` to force retraining."
        end
        return mach
    else
        # update the model:
        verbosity < 1 || @info "Updating $mach."
        mach.fitresult, mach.cache, mach.report =
            update(mach.model, verbosity, mach.fitresult, mach.cache, args...)

    end

    if rows_have_changed
        mach.previous_rows = deepcopy(rows)
    end

    mach.previous_model = deepcopy(mach.model)

    if mach isa NodalMachine
        mach.upstream_state = upstream_state
         mach.state = mach.state + 1
    end

    return mach
end

is_stale(mach::Machine) =
    !isdefined(mach, :fitresult) || (mach.model != mach.previous_model)

params(mach::AbstractMachine) = params(mach.model)

"""
    report(mach)

Return the report for a machine `mach` that has been
`fit!`, for example the coefficients in a linear model.

This is a named tuple and human-readable if possible.

If `mach` is a machine for a composite model, then the returned value
has keys `machines` and `report_given_machine`, whose
corresponding values are a vector of (nodal) machines appearing in the
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
r = report(mach)
r.machines
2-element Array{Any,1}:
 NodalMachine{LinearBinaryClassifier{LogitLink}} @ 1…57
 NodalMachine{Standardizer} @ 7…33

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
    MLJ.save(filename, mach::AbstractMachine; kwargs...)
    MLJ.save(io, mach::Machine; kwargs...)

    MLJBase.save(filename, mach::AbstractMachine; kwargs...)
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
    mach = Machine(model, args...)
    mach.fitresult = fitresult
    mach.report = report
    return mach
end
