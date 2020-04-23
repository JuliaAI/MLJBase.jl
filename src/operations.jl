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

for operation in (:predict, :predict_mean, :predict_mode, :predict_median,
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
"""
    fitted_params(mach)

Return the learned parameters for a machine `mach` that has been
`fit!`, for example the coefficients in a linear model.

This is a named tuple and human-readable if possible.

If `mach` is a machine for a composite model, then the returned value
has keys `machines` and `fitted_params_given_machine`, whose
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
fp = fitted_params(mach)
machs = fp.machines
2-element Array{Any,1}:
 NodalMachine{LinearBinaryClassifier{LogitLink}} @ 1…57
 NodalMachine{Standardizer} @ 7…33

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
