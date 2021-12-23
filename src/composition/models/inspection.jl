## USER FRIENDLY INSPECTION OF COMPOSITE MACHINES

try_scalarize(v) = length(v) == 1 ? v[1] : v

function machines_given_model_name(mach::Machine{M}) where  M<:Composite
    network_model_names = mach.cache.network_model_names
    names = unique(filter(name->!(name === nothing), network_model_names))
    network_models = MLJBase.models(glb(mach))
    network_machines = MLJBase.machines(glb(mach))
    ret = LittleDict{Symbol,Any}()
    for name in names
        mask = map(==(name), network_model_names)
        _models = network_models[mask]
        _machines = filter(mach->mach.model in _models, network_machines)
        ret[name] = _machines
    end
    return ret
end

function tuple_keyed_on_model_names(item_given_machine, mach)
    dict = MLJBase.machines_given_model_name(mach)
    names = tuple(keys(dict)...)
    named_tuple_values = map(names) do name
        [item_given_machine[m] for m in dict[name]] |> try_scalarize
    end
    return NamedTuple{names}(named_tuple_values)
end

function report(mach::Machine{<:Composite})
    machines = mach.report.machines
    dict = mach.report.report_given_machine
    return merge(tuple_keyed_on_model_names(dict, mach), mach.report)
end

function fitted_params(mach::Machine{<:Composite})
    fp = fitted_params(mach.model, mach.fitresult)
    _machines = fp.machines
    dict = fp.fitted_params_given_machine
    return merge(MLJBase.tuple_keyed_on_model_names(dict, mach), fp)
end
