abstract type Task <: MLJType end

struct UnsupervisedTask <: Task
    data
    ignore::Vector{Symbol}
    input_scitypes
    input_quantity::Symbol
end

function UnsupervisedTask(
    ; data=nothing
    , ignore=Symbol[])

    data != nothing  || throw(error("You must specify data=..."))
    input_scitypes = scitype(data)
    input_quantity =
        data isa AbstractVector ? :univariate : multivariate
    return SupervisedTask(data, ignore, input_scitypes, input_quantity)
end

struct SupervisedTask <: Task
    data
    targets
    ignore::Vector{Symbol}
    is_probabilistic
    target_scitype
    input_scitypes
    input_quantity
end

function SupervisedTask(
    ; data=nothing
    , targets=nothing
    , ignore=Symbol[]
    , probabilistic=nothing)

    data != nothing    || error("You must specify data=... ")
    targets != nothing || error("You must specify targets=... (use Symbol or Vector{Symbol}) ")
    probabilistic != nothing || error("You must specify is_probabilistic=true or is_proabilistic=false. ")

    if targets isa Vector
        target_scitype = Tuple{[scitype(getproperty(data, t)) for t in targets]...}
    else
        target_scitype = scitype(getproperty(data, targets))
        targets = [targets, ]
    end

    issubset(Set(targets), Set(schema(data).names)) ||
        throw(error("One or more specified targets missing from supplied data. "))
                                      
    input_quantity =
        data isa AbstractVector ? :univariate : :multivariate

        return SupervisedTask(data, targets, ignore, probabilistic, target_scitype, input_scitypes, input_quantity)
end


## RUDIMENTARY TASK OPERATIONS

nrows(task::Task) = schema(task.data).nrows
Base.eachindex(task::Task) = Base.OneTo(nrows(task))

features(task::Task) = filter!(schema(task.data).names |> collect) do ftr
    !(ftr in task.ignore)
end

features(task::SupervisedTask) = filter(schema(task.data).names |> collect) do ftr
    !(ftr in task.targets) && !(ftr in task.ignore)
end

X_and_y(task::SupervisedTask) = (selectcols(task.data, features(task)),
                                 [selectcols(task.data, fld) for fld in task.targets]...)


