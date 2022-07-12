## USER FRIENDLY INSPECTION OF COMPOSITE MACHINES

try_scalarize(v) = length(v) == 1 ? v[1] : v

function machines_given_model_name(mach::Machine{M}) where  M<:Composite
    network_model_names = getfield(mach.fitresult, :network_model_names)
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
    # We have two different ways to get the report. One uses `mach.cache` and is slightly
    # more up-to-date because it includes effects of report-altering operations (such as
    # `transform`) post-`fit!` for special components with non-empty `reporting_operations`
    # trait. However, if a machine is a deserialized one, then it has no `cache` and we
    # fallback to report at last `fit!`.
    if isdefined(mach, :cache)
        glb_report = report(mach.cache.glb)
        dict = glb_report.report_given_machine
        return merge(
            tuple_keyed_on_model_names(dict, mach),
            glb_report,
            mach.cache.report_additions,
        )
    else
        dict = mach.report.report_given_machine
        return merge(tuple_keyed_on_model_names(dict, mach), mach.report)
    end
end

function fitted_params(mach::Machine{<:Composite})
    fp = fitted_params(mach.model, mach.fitresult)
    dict = fp.fitted_params_given_machine
    return merge(MLJBase.tuple_keyed_on_model_names(dict, mach), fp)
end
