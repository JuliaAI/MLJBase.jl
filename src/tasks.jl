abstract type Task <: MLJType end

struct UnsupervisedTask <: Task
    data
    ignore::Vector{Symbol}
    input_scitypes
    input_is_multivariate::Bool
end

"""
    task = UnsupervisedTask(; data=nothing, ignore=Symbol[])

Construct a unsupervised learning task using `data`, which is any table
supported by Tables.jl. Note that rows must correspond to patterns and
columns to features.  The input features for the task are all
the columns of `data`, with the exception of those specified
by `ignore`. 

    X = task()

Retrieve the table of inputs for `task` (the table `data` without
features in `ignore`.

"""
function UnsupervisedTask(
    ; data=nothing
    , ignore=Symbol[])

    data != nothing  || throw(error("You must specify data=..."))

    input_scitypes = scitype(data)
    input_is_multivariate = true
#        data isa AbstractVector ? false : multivariate

    if ignore isa Symbol
        ignore = [ignore, ]
    end
    
    return UnsupervisedTask(data, ignore, input_scitypes, input_is_multivariate)
end

struct SupervisedTask{U} <: Task # U is true for single target
    data
    targets
    ignore::Vector{Symbol}
    is_probabilistic
    target_scitype
    input_scitypes
    input_is_multivariate::Bool
end

"""
    task = SupervisedTask(; data=nothing, targets=nothing, ignore=Symbol[], probabilistic=nothing)

Construct a supervised learning task using `data`, which is any table
supported by Tables.jl. Note that rows must correspond to patterns and
columns to features. The name of the target column(s) is specfied by
`target` (a symbol or vector of symbols). The input features are all
the remaining columns of `data`, with the exception of those specified
by `ignore`. All keyword arguments except `ignore` are compulsory.

    X, y = task()

Retrieve the table of input features `X` and target(s) `y`. For single
targets, `y` is a `Vector` or `CategoricalVector`. In the multivariate
case, `y` is a named tuple of vectors (and, in particular, a table
supported by Tables.jl). Additional methods give access to select parts of the data: 

    X_(task), y_(task) == task()   # true

"""
function SupervisedTask(
    ; data=nothing
    , targets=nothing
    , ignore=Symbol[]
    , probabilistic=nothing)

    data != nothing    || error("You must specify data=... ")

    targets != nothing ||
        error("You must specify targets=... (use Symbol or Vector{Symbol}) ")

    probabilistic != nothing ||
        error("You must specify is_probabilistic=true or is_proabilistic=false. ")

    is_univariate = true

    if targets isa Vector
        is_univariate = false
        target_scitype = Tuple{[scitype(getproperty(data, t)) for t in targets]...}
    else
        target_scitype = scitype(getproperty(data, targets))
        targets = [targets, ]
    end

    issubset(Set(targets), Set(schema(data).names)) ||
        throw(error("One or more specified targets missing from supplied data. "))
                                      
    if ignore isa Symbol
        ignore = [ignore, ]
    end
    
    input_is_multivariate = true
#        data isa AbstractVector ? false : true

    return SupervisedTask{is_univariate}(data, targets, ignore,
                                         probabilistic, target_scitype,
                                         input_scitypes, input_is_multivariate)
end


## RUDIMENTARY TASK OPERATIONS

nrows(task::Task) = schema(task.data).nrows
Base.eachindex(task::Task) = Base.OneTo(nrows(task))

"""
    names(task::UnsupervisedTask)

Return the names of all columns of the task data, excluding any in `task.ignore`.

"""
Base.names(task::UnsupervisedTask) = filter!(schema(task.data).names |> collect) do ftr
    !(ftr in task.ignore)
end

"""
    names(task::SupervisedTask)

Return the names of all columns of the task data, excluding target
columns and any names in `task.ignore`.

"""
Base.names(task::SupervisedTask) = filter(schema(task.data).names |> collect) do ftr
    !(ftr in task.targets) && !(ftr in task.ignore)
end

X_(task::Task) = selectcols(task.data, names(task))

y_(task::SupervisedTask{true}) = selectcols(task.data, task.targets[1])

function y_(task::SupervisedTask{false})
    cols = [selectcols(task.data, fld) for fld in task.targets]
    return NamedTuple{tuple(task.targets...)}(tuple(cols...))
end


X_and_y(task::SupervisedTask) = (X_(task), y_(task))

# make tasks callable:
(task::UnsupervisedTask)() = X_(task)
(task::SupervisedTask)() = X_and_y(task)


