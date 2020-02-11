# We wish to extend operations to identically named methods dispatched
# on `Machine`s and `NodalMachine`s. For example, we have from the model API
#
# `predict(model::M, fitresult, X) where M<:Supervised`
#
# but want also want
#
# `predict(machine::Machine, X)` where `X` is data
#
# and "networks.jl" requires us to define
#
# `predict(machine::NodalMachine, X)` where `X` is data
#
# and we would like the syntactic sugar (for `X` a node):
#
# `predict(machine::NodalMachine, X::Node)=node(predict, machine, X)`
#
# (If an operation has zero arguments, we cannot achieve the last
# desire because of ambiguity with the preceding one.)

## TODO: need to add checks on the arguments of
## predict(::AbstractMachine, ) and transform(::AbstractMachine, )

for operation in (:predict, :predict_mean, :predict_mode,
                  :transform, :inverse_transform)
    ex = quote
        function $(operation)(machine::AbstractMachine, args...)
            if isdefined(machine, :fitresult)
                return $(operation)(machine.model, machine.fitresult, args...)
            else
                throw(error("$machine has not been trained."))
            end
        end
        $(operation)(machine::Machine; rows=:) =
            $(operation)(machine, selectrows(machine.args[1], rows))
        $(operation)(machine::NodalMachine, args::AbstractNode...) =
            node($(operation), machine, args...)
    end
    eval(ex)
end

# the zero argument special cases:
function fitted_params(machine::AbstractMachine)
    if isdefined(machine, :fitresult)
        return fitted_params(machine.model, machine.fitresult)
    else
        throw(error("$machine has not been trained."))
    end
end
