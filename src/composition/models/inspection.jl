## USER FRIENDLY INSPECTION OF COMPOSITE MACHINES

try_scalarize(v) = length(v) == 1 ? v[1] : v

# get the component models of a composite model `mach` (versions at
# last fit!!) and the corresponding field names (filtering out fields
# that are not models):
function models_and_names(mach::Machine{M}) where  M<:Composite
    raw_names = fieldnames(M)
    _values = Tuple(getproperty(mach.old_model, name) for name in raw_names)
    raw_meta = zip(_values, raw_names) |> collect
    meta = filter(raw_meta) do (value, name)
        value isa Model
    end
    return tuple(zip(meta...)...)
end

# given a dictionary keyed on machines, with "items" as values
# (use-cases are reports or fitted_params) return a named-tuple, keyed
# on model names, whose values are corresponding items or vectors of
# items, but restricted to the specified `models`.
function tuple_keyed_on_model_names(item_given_machine, _models, names)
    machs = keys(item_given_machine) |> collect
    named_tuple_values = map(_models) do model
        sub_machs =
            filter(mach -> mach.old_model == model, machs)
        [item_given_machine[mach] for mach in sub_machs] |> try_scalarize
    end
    return NamedTuple{names}(Tuple(named_tuple_values))
end

function report(mach::Machine{<:Composite})
    dict = mach.report.report_given_machine
    _models, names =  models_and_names(mach)
    return merge(tuple_keyed_on_model_names(dict, _models, names),
                 (report_given_machine=dict,))
end

function fitted_params(mach::Machine{<:Composite})
    fp = fitted_params(mach.model, mach.fitresult)
    machines = fp.machines
    dict = fp.fitted_params_given_machine
    _models, names =  models_and_names(mach)
    return merge(tuple_keyed_on_model_names(dict, _models, names),
                 (machines=machines, fitted_params_given_machine=dict,))
end
