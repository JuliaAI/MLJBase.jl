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
    "is bound to `mach` in this case. Specify a explicit "*
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


# Given return value `ret` of an operation with symbol `operation` (eg, `:predict`) return
# `ret` in the ordinary case that the operation does not include an "report" component ;
# otherwise update `mach.report` with that component and return the non-report part of
# `ret`:
named_tuple(t::Nothing) = NamedTuple()
named_tuple(t) = t
function get!(ret, operation, mach)
    model = recent_model(mach)
    if operation in reporting_operations(model)
        report = named_tuple(last(ret))
        if isnothing(mach.report) || isempty(mach.report)
            mach.report = Dict{Symbol,Any}(operation => report)
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
        function $(operation)(mach::Machine{<:Model,false}; rows=:)
            # catch deserialized machine with no data:
            isempty(mach.args) && throw(err_serialized($operation))
            return ($operation)(mach, mach.args[1](rows=rows))
        end
        function $(operation)(mach::Machine{<:Model,true}; rows=:)
            # catch deserialized machine with no data:
            isempty(mach.args) && _err_serialized($operation)
            model = recent_model(mach)
            ret = ($operation)(
                model,
                mach.fitresult,
                selectrows(model, rows, mach.data[1])...,
            )
            return get!(ret, $quoted_operation, mach)
        end

        # special case of Static models (no training arguments):
        $operation(mach::Machine{<:Static,true}; rows=:) = throw(ERR_ROWS_NOT_ALLOWED)
        $operation(mach::Machine{<:Static,false}; rows=:) = throw(ERR_ROWS_NOT_ALLOWED)
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
            model = recent_model(mach)
            ret = $(operation)(
                model,
                mach.fitresult,
                reformat(model, Xraw)...,
            )
            get!(ret, $quoted_operation, mach)
        end

        function $operation(mach::Machine{<:Static}, Xraw, Xraw_more...)
            _check_and_fit_if_warranted!(mach)
            ret = $(operation)(
                recent_model(mach),
                mach.fitresult,
                Xraw,
                Xraw_more...,
            )
            get!(ret, $quoted_operation, mach)
        end

        # 2. operations on machines, given *dynamic* data (nodes):
        $operation(mach::Machine, X::AbstractNode) =
            node($(operation), mach, X)

        $operation(mach::Machine{<:Static},
                   X::AbstractNode,
                   Xmore::AbstractNode...) =
                       node($(operation), mach, X, Xmore...)
    end
    eval(ex)
end


## SURROGATE AND COMPOSITE MODELS

const err_unsupported_operation(operation) = ErrorException(
    "The `$operation` operation has been applied to a composite model or learning "*
    "network machine that does not support it. "
)

for operation in [:predict,
                  :predict_joint,
                  :transform,
                  :inverse_transform]
    ex = quote
        function $operation(model::Union{Composite,Surrogate}, fitresult,X)
            if hasproperty(fitresult, $(QuoteNode(operation)))
                return fitresult.$operation(X)
            else
                throw(err_unsupported_operation($operation))
            end
        end
    end
    eval(ex)
end

for (operation, fallback) in [(:predict_mode, :mode),
                              (:predict_mean, :mean),
                              (:predict_median, :median)]
    ex = quote
        function $(operation)(m::Union{ProbabilisticComposite,ProbabilisticSurrogate},
                              fitresult,
                              Xnew)
            if hasproperty(fitresult, $(QuoteNode(operation)))
                return fitresult.$(operation)(Xnew)
            end
            return $(fallback).(predict(m, fitresult, Xnew))
        end
    end
    eval(ex)
end
