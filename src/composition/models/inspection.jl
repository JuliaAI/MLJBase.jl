## USER FRIENDLY INSPECTION OF COMPOSITE MACHINES

try_scalarize(v) = length(v) == 1 ? v[1] : v

function machines_given_model_name(mach::Machine{M}) where  M<:Composite
    network_model_names = getfield(mach.fitresult, :network_model_names)
    names = unique(filter(name->!(name === nothing), network_model_names))
    glb = MLJBase.glb(mach)
    network_models = MLJBase.models(glb)
    network_machines = MLJBase.machines(glb)
    ret = LittleDict{Symbol,Any}()
    for name in names
        mask = map(==(name), network_model_names)
        _models = network_models[mask]
        _machines = filter(mach->mach.model in _models, network_machines)
        ret[name] = _machines
    end
    return ret
end

function tuple_keyed_on_model_names(machines, mach, f)
    dict = MLJBase.machines_given_model_name(mach)
    names = tuple(keys(dict)...)
    named_tuple_values = map(names) do name
        [f(m) for m in dict[name]] |> try_scalarize
    end
    return NamedTuple{names}(named_tuple_values)
end

function report(mach::Machine{<:Union{Composite,Surrogate}})
    report_additions = mach.report.additions
    report_basic = mach.report.basic
    report_components = mach isa Machine{<:Surrogate} ? NamedTuple() :
        MLJBase.tuple_keyed_on_model_names(report_basic.machines, mach, MLJBase.report)
    return merge(report_components, report_basic, report_additions)
end

function fitted_params(mach::Machine{<:Composite})
    fp_basic = fitted_params(mach.model, mach.fitresult)
    machines = fp_basic.machines
    fp_components =
        MLJBase.tuple_keyed_on_model_names(machines, mach, MLJBase.fitted_params)
    return merge(fp_components, fp_basic)
end
