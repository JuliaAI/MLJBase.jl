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
                  :transform, :inverse_transform)

for operation in OPERATIONS

    if operation != :inverse_transform

        ex = quote
            # 0. operations on machs, given empty data:
            function $(operation)(mach::Machine; rows=:)
                Base.depwarn("`$($operation)(mach)` and "*
                             "`$($operation)(mach, rows=...)` are "*
                             "deprecated. Data or nodes "*
                             "must be explictly specified, "*
                             "as in `$($operation)(mach, X)`. ",
                             Base.Core.Typeof($operation).name.mt.name)
                if isempty(mach.args) # deserialized machine with no data
                    throw(ArgumentError("Calling $($operation) on a "*
                                        "deserialized machine with no data "*
                                        "bound to it. "))
                end
                return ($operation)(mach, mach.args[1]())
            end
        end
        eval(ex)

    end
end

for operation in (:inverse_transform,)
    ex = quote
        # 0. operations on machines, given empty data:
        $(operation)(mach::Machine; rows=:) =
            throw(ArgumentError("`$($operation)(mach)` and "*
                                "`$($operation)(mach, rows=...)` is "*
                                "not supported. Data or nodes "*
                                "must be explictly specified, "*
                                "as in `$($operation)(mach, X)`. "))
    end
    eval(ex)
end

for operation in OPERATIONS

    ex = quote
        # 1. operations on machines, given *concrete* data:
        function $(operation)(mach::Machine{M}, Xraw, Xraw_more...) where M
            if mach.state > 0 || M <: Static
                return $(operation)(mach.model, mach.fitresult,
                                    Xraw, Xraw_more...)
            else
                error("$mach has not been trained.")
            end
        end

        # 2. operations on machines, given *dynamic* data (nodes):
        $(operation)(mach::Machine, X::AbstractNode) =
            node($(operation), mach, X)

        # 3. operations on composite and surrogate models:
        $(operation)(model::Union{Composite,Surrogate}, fitresult, X) =
            fitresult.$operation(X)
    end
    eval(ex)
end
