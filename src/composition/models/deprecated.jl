# # IMPLEMENTATION OF MLJ MODEL INTERFACE FOR  <:Composite MODELS


# # save and restore!

# Returns a new `CompositeFitresult` that is a shallow copy of the original one.
# To do so,  we build a copy of the learning network where each machine contained
# in it needs to be called `serializable` upon.
function save(model::Composite, fitresult)
    signature = MLJBase.signature(fitresult)
    operation_nodes = values(MLJBase._operation_part(signature))
    report_nodes = values(MLJBase._report_part(signature))
    W = glb(operation_nodes..., report_nodes...)
    newnode_given_old =
        IdDict{AbstractNode,AbstractNode}([old => source() for old in sources(W)])

    newsignature = copysignature!(signature, newnode_given_old; newmodel_given_old=nothing)

    newfitresult = MLJBase.CompositeFitresult(newsignature)
    setfield!(newfitresult, :network_model_names, getfield(fitresult, :network_model_names))

    return newfitresult
end


# Restores a machine of a composite model by restoring all
# submachines contained in it.
function restore!(mach::Machine{<:Composite})
    glb_node = glb(mach)
    for submach in machines(glb_node)
        restore!(submach)
    end
    mach.state = 1
    return mach
end

function setreport!(copymach::Machine{<:Composite}, mach)
    basic = report(glb(copymach.fitresult))
    additions = mach.report.additions
    copymach.report = (; basic, additions)
end
