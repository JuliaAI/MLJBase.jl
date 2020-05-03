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
# Finally, for a `model` that is `ProbabilisticNetwork`,
# `DetermisiticNetwork`, or `UnsupervisedNetwork`, we want
#
# 3. `predict(model, fitresult, X) = fitresult.predict(X)`
#
# which makes sense because `fitresult` in those cases is a named
# tuple keyed on supported operations and with nodes as values.

## TODO: need to add checks on the arguments of
## predict(::Machine, ) and transform(::Machine, )

for operation in (:predict, :predict_mean, :predict_mode, :predict_median,
                  :transform, :inverse_transform)
    ex = quote
        # 0. operations on machines, given empty data:
        $(operation)(machine::NodalMachine; rows=:) =
            error("`$($operation)(mach)` and "*
                  "`$($operation)(mach, rows=...)` are "*
                  "no longer supported. Data or nodes "*
                  "must be explictly specified, "*
                  "as in `$($operation)(mach, X)`. ")

        # 1. operations on machines, given *concrete* data:
        function $(operation)(machine::NodalMachine{M}, Xraw) where M
            if state(machine) > 0 || M <: Static
                return $(operation)(machine.model, machine.fitresult, Xraw)
            else
                error("$machine has not been trained.")
            end
        end

        # 2. operations on machines, given *dynamic* data (nodes):
        $(operation)(machine::NodalMachine{M}, X::AbstractNode) =
            node($(operation), machine, X)

        # 3. operations on composite models:
        $(operation)(model::GenericNetwork, fitresult, X) =
            fitresult.$operation(X)
    end
    eval(ex)
end
