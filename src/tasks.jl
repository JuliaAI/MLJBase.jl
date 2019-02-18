# TODO: add evaluation metric:
# TODO: add `input_kinds` and `outputs_are`
# TODO: add multiple targets

abstract type Task <: MLJType end

struct UnsupervisedTask <: Task
    data
    ignore::Vector{Symbol}
    input_kinds
    input_quantity
    output_kind
    output_quantity
end

function UnsupervisedTask(
    ; data=nothing
    , ignore=Symbol[]
    , input_kinds=nothing
    , input_quantity=:multivariate
    , output_kind=nothing
    , output_quantity=:univariate)

    data != nothing         || throw(error("You must specify data=..."))
    input_kinds != nothing  || throw(error("You must specify input_kinds=..."))
    if !(input_kinds isa Vector)
        input_kinds = [input_kinds, ]
    end
    output_kind != nothing  || throw(error("You must specify output_kind=..."))
    
    return SupervisedTask(data, ignore, input_kinds, input_quantity,
                          output_kind, output_quantity)
end


struct SupervisedTask <: Task
    data
    targets
    ignore::Vector{Symbol}
    is_probabilistic
    input_kinds
    input_quantity
    output_kind
    output_quantity
end

function SupervisedTask(
    ; data=nothing
    , targets=nothing
    , ignore=Symbol[]
    , is_probabilistic=nothing
    , input_kinds=nothing
    , input_quantity=:multivariate
    , output_kind=nothing
    , output_quantity=:univariate)


    data != nothing    || throw(error("You must specify data=..."))
    targets != nothing || throw(error("You must specify targets=... (use Symbol or Vector{Symbol})"))
    if !(targets isa Vector)
        targets = [targets, ]
    end
    issubset(Set(targets), Set(schema(data).names)) ||
        throw(error("Supplied data does not have $target as field."))
    is_probabilistic != nothing || throw(error("You must specify is_probabilistic=..."))
    input_kinds != nothing      || throw(error("You must specify input_kinds=..."))
    output_kind != nothing      || throw(error("You must specify output_kind=..."))

    return SupervisedTask(data, targets, ignore, is_probabilistic, input_kinds, input_quantity,
                          output_kind, output_quantity)
end


## RUDIMENTARY TASK OPERATIONS

nrows(task::Task) = schema(task.data).nrows
Base.eachindex(task::Task) = Base.OneTo(nrows(task))

features(task::Task) = filter!(schema(task.data).names |> collect) do ftr
    !(ftr in task.ignore)
end

features(task::SupervisedTask) = filter(schema(task.data).names |> collect) do ftr
    ftr != task.targets && !(ftr in task.ignore)
end

X_and_y(task::SupervisedTask) = (selectcols(task.data, features(task)),
                                 [selectcols(task.data, fld) for fld in task.targets]...)

#GET EXPORT STATEMENTS FROM MLJ
