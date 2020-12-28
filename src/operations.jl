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

const OPERATIONS = (:predict, :predict_mean, :predict_mode, :predict_median,
                    :predict_joint, :transform, :inverse_transform)

# 0. operations on machine, given rows=...:

for operation in OPERATIONS

    if operation != :inverse_transform

        ex = quote
            function $(operation)(mach::Machine; rows=:)
                if isempty(mach.args) # deserialized machine with no data
                    throw(ArgumentError("Calling $($operation) on a "*
                                        "deserialized machine with no data "*
                                        "bound to it. "))
                end
                return ($operation)(mach, mach.args[1](rows=rows))
            end
        end
        eval(ex)

    end
end

const ROWS_NOT_ALLOWED = "Calling `transform(mach, rows=...)` when "*
                        "`mach.model isa Static` is not allowed, as no data "*
                        "is bound to `mach` in this case. Specify a explicit "*
                        "data or node, as in `transform(mach, X)`, or "*
                        "`transform(mach, X1, X2, ...)`. "

# special case of Static models (no training arguments):
transform(mach::Machine{<:Static}; rows=:) =
    throw(ArgumentError(ROWS_NOT_ALLOWED))

inverse_transform(mach::Machine; rows=:) =
            throw(ArgumentError("`inverse_transform()(mach)` and "*
                                "`inverse_transform(mach, rows=...)` are "*
                                "not supported. Data or nodes "*
                                "must be explictly specified, "*
                                "as in `inverse_transform(mach, X)`. "))

_symbol(f) = Base.Core.Typeof(f).name.mt.name

for operation in OPERATIONS

    ex = quote
        # 1. operations on machines, given *concrete* data:
        function $operation(mach::Machine, Xraw)
            if mach.state > 0
                return $(operation)(mach.model, mach.fitresult,
                                    Xraw)
            else
                error("$mach has not been trained.")
            end
        end

        function $operation(mach::Machine{<:Static}, Xraw, Xraw_more...)
            isdefined(mach, :fitresult) || (mach.fitresult = nothing)
            return $(operation)(mach.model, mach.fitresult,
                                    Xraw, Xraw_more...)
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

for operation in [:predict, :predict_joint, :transform, :inverse_transform]
    ex = quote
        $operation(model::Union{Composite,Surrogate}, fitresult,X) =
            fitresult.$operation(X)
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
            if haskey(fitresult, $(QuoteNode(operation)))
                return fitresult.$(operation)(Xnew)
            end
            return $(fallback).(predict(m, fitresult, Xnew))
        end
    end
    eval(ex)
end
