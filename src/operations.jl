# We wish to extend operations to identically named methods dispatched
# on `Machine`s. For example, we have from the model API
#
# `predict(model::M, fitresult, X) where M<:Supervised`
#
# but want also want to define
#
# 1. `predict(machine::Machine, X)` where `X` is concrete data
#
# and we would like the syntactic sugar (for `X` a node):
#
# 2. `predict(machine::Machine, X::Node) = node(predict, machine, X)`
#
# Finally, for a `model` that is `ProbabilisticComposite`,
# `DetermisiticComposite`, or `UnsupervisedComposite`, we want
#
# 3. `predict(model, fitresult, X) = fitresult.predict(X)`
#
# which makes sense because `fitresult` in those cases is a named
# tuple keyed on supported operations and with nodes as values.

## TODO: need to add checks on the arguments of
## predict(::Machine, ) and transform(::Machine, )

const ERR_ROWS_NOT_ALLOWED = ArgumentError(
    "Calling `transform(mach, rows=...)` or "*
    "`predict(mach, rows=...)` when "*
    "`mach.model isa Static` is not allowed, as no data "*
    "is bound to `mach` in this case. Specify an explicit "*
    "data or node, as in `transform(mach, X)`, or "*
    "`transform(mach, X1, X2, ...)`. "
)

err_serialized(operation) = ArgumentError(
    "Calling $operation on a "*
    "deserialized machine with no data "*
    "bound to it. "
)

const err_untrained(mach) = ErrorException("$mach has not been trained. ")

const WARN_SERIALIZABLE_MACH = "You are attempting to use a "*
    "deserialised machine whose learned parameters "*
    "may be unusable. To be sure they are usable, "*
    "first run restore!(mach)."

_scrub(x::NamedTuple) = isempty(x) ? nothing : x
_scrub(something_else) = something_else
# Given return value `ret` of an operation with symbol `operation` (eg, `:predict`) return
# `ret` in the ordinary case that the operation does not include an "report" component ;
# otherwise update `mach.report` with that component and return the non-report part of
# `ret`:
function get!(ret, operation, mach)
    model = last_model(mach)
    if operation in reporting_operations(model)
        report = _scrub(last(ret))
        # mach.report will always be a dictionary:
        if isempty(mach.report)
            mach.report = LittleDict{Symbol,Any}(operation => report)
        else
            mach.report[operation] = report
        end
        return first(ret)
    end
    return ret
end

# 0. operations on machine, given rows=...:

for operation in OPERATIONS

    quoted_operation = QuoteNode(operation) # eg, :(:predict)

    operation == :inverse_transform && continue

    ex = quote
        function $(operation)(mach::Machine{<:Model,<:Any,false}; rows=:)
            # catch deserialized machine with no data:
            isempty(mach.args) && throw(err_serialized($operation))
            return ($operation)(mach, mach.args[1](rows=rows))
        end
        function $(operation)(mach::Machine{<:Model,<:Any,true}; rows=:)
            # catch deserialized machine with no data:
            isempty(mach.args) && throw(err_serialized($operation))
            model = last_model(mach)
            ret = ($operation)(
                model,
                mach.fitresult,
                selectrows(model, rows, mach.data[1])...,
            )
            return get!(ret, $quoted_operation, mach)
        end

        # special case of Static models (no training arguments):
        $operation(mach::Machine{<:Static,<:Any,true}; rows=:) =
            throw(ERR_ROWS_NOT_ALLOWED)
        $operation(mach::Machine{<:Static,<:Any,false}; rows=:) =
            throw(ERR_ROWS_NOT_ALLOWED)
    end
    eval(ex)

end

inverse_transform(mach::Machine; rows=:) =
            throw(ArgumentError("`inverse_transform(mach)` and "*
                                "`inverse_transform(mach, rows=...)` are "*
                                "not supported. Data or nodes "*
                                "must be explictly specified, "*
                                "as in `inverse_transform(mach, X)`. "))

_symbol(f) = Base.Core.Typeof(f).name.mt.name

# catches improperly deserialized machines and silently fits the machine if it is
# untrained and has no training arguments:
function _check_and_fit_if_warranted!(mach)
    mach.state == -1 && @warn WARN_SERIALIZABLE_MACH
    if mach.state == 0
        if isempty(mach.args)
            fit!(mach, verbosity=0)
        else
            throw(err_untrained(mach))
        end
    end
end

for operation in OPERATIONS

    quoted_operation = QuoteNode(operation) # eg, :(:predict)

    ex = quote
        # 1. operations on machines, given *concrete* data:
        function $operation(mach::Machine, Xraw)
            _check_and_fit_if_warranted!(mach)
            model = last_model(mach)
            ret = $(operation)(
                model,
                mach.fitresult,
                reformat(model, Xraw)[1],
            )
            get!(ret, $quoted_operation, mach)
        end

        function $operation(mach::Machine, Xraw, Xraw_more...)
            _check_and_fit_if_warranted!(mach)
            ret = $(operation)(
                last_model(mach),
                mach.fitresult,
                Xraw,
                Xraw_more...,
            )
            get!(ret, $quoted_operation, mach)
        end

        # 2. operations on machines, given *dynamic* data (nodes):
        $operation(mach::Machine, X::AbstractNode) =
            node($(operation), mach, X)

        $operation(
            mach::Machine,
            X::AbstractNode,
            Xmore::AbstractNode...,
        ) = node($(operation), mach, X, Xmore...)
    end
    eval(ex)
end


const err_unsupported_operation(operation) = ErrorException(
    "The `$operation` operation has been applied to a composite model or learning "*
    "network machine that does not support it. "
)

## NETWORK COMPOSITE MODELS

# In the case of `NetworkComposite` models, the `fitresult` is a learning network
# signature. If we call a node in the signature (eg, do `fitresult.predict()`) then we may
# mutate the underlying learning network (and hence `fitresult`). This is because some
# nodes in the network may be attached to machines whose reports are mutated when an
# operation is called on them (the associated model has a non-empty `reporting_operations`
# trait). For this reason we must first duplicate `fitresult`.

# The function `output_and_report(signature, operation, Xnew)` called below (and defined
# in signatures.jl) duplicates `signature`, applies `operation` with data `Xnew`, and
# returns the output and signature report.

for operation in [:predict,
                  :predict_joint,
                  :transform,
                  :inverse_transform]
    quote
        function $operation(model::NetworkComposite, fitresult, Xnew...)
            if $(QuoteNode(operation)) in MLJBase.operations(fitresult)
                return output_and_report(fitresult, $(QuoteNode(operation)), Xnew...)
            end
            throw(err_unsupported_operation($operation))
        end
    end |> eval
end

for (operation, fallback) in [(:predict_mode, :mode),
                              (:predict_mean, :mean),
                              (:predict_median, :median)]
    quote
        function $(operation)(m::ProbabilisticNetworkComposite,
                              fitresult,
                              Xnew)
            if $(QuoteNode(operation)) in MLJBase.operations(fitresult)
                return output_and_report(fitresult, $(QuoteNode(operation)), Xnew)
            end
            # The following line retuns a `Tuple` since `m` is a `NetworkComposite`
            predictions, report = predict(m, fitresult, Xnew)
            return $(fallback).(predictions), report
        end
    end |> eval
end
