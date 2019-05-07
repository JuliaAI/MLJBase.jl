abstract type MLJTask <: MLJType end # `Task` already has meaning in Base

mutable struct UnsupervisedTask <: MLJTask
    X
    input_scitypes::NamedTuple
    input_scitype_union::DataType
    input_is_multivariate::Bool
end

function UnsupervisedTask(; data=nothing, ignore=Symbol[], verbosity=1)

    data != nothing || error("You must specify data=... ")

    if ignore isa Symbol
        ignore = [ignore, ]
    end

    names = schema(data).names |> collect
    
    issubset(ignore, names) || error("ignore=$ignore contains a column name "*
                                     "not encountered in supplied data.")

    features = filter(names) do ftr
        !(ftr in ignore)
    end

    if length(features) == 1
        input_is_multivariate = false
        X = selectcols(data, features[1])
    else
        input_is_multivariate = true
        X = selectcols(data, features)
    end
    
    if input_is_multivariate
        input_scitypes = scitypes(X)
        input_scitype_union = Union{input_scitypes...}
    else
        input_scitype_union = scitype_union(X)
        input_scitypes = (input=input_scitype_union,)
    end
    
    if Unknown <: input_scitype_union
        @warn "An input feature with unknown scitype has been encountered. "*
        "Check the input_scitypes field of your task. "
        
    end

    if verbosity > 0 
        @info "input_scitype_union = $input_scitype_union"
    end

    return UnsupervisedTask(X, input_scitypes, input_scitype_union, 
                            input_is_multivariate)
end

# X and y can be different views of a common object.
mutable struct SupervisedTask <: MLJTask 
    X                               
    y
    is_probabilistic
    input_scitypes
    target
    input_scitype_union
    target_scitype_union
    input_is_multivariate::Bool
end

function SupervisedTask(X, y::AbstractVector;
                        is_probabilistic=false,
                        input_is_multivariate=true,
                        target=:target,
                        verbosity=1)

    # is_probabilistic != nothing ||
    #     error("You must specify is_probabilistic=true or is_probabilistic=false. ")

    target_scitype_union = scitype_union(y)

    if target_scitype_union <: Tuple
        types = target_scitype_union.parameters        
        unknown_found = reduce(|, [Unknown <: t for t in types])
    else
        unknown_found = Unknown <: target_scitype_union
    end

    if input_is_multivariate
        input_scitypes = scitypes(X)
        input_scitype_union = Union{input_scitypes...}
    else
        input_scitype_union = scitype_union(X)
        input_scitypes = (input=input_scitype_union,)
    end
    
    unknown_found = unknown_found || (Unknown <: input_scitype_union)

    if unknown_found
        @warn "An Unknown scitype has been encountered. "*
        "Check the input_scitypes and target_scitype_union fields of "*
        "your task. "
    end

    if verbosity > 0 
        @info "\nis_probabilistic = $is_probabilistic\ninput_scitype_union = "*
        "$input_scitype_union \ntarget_scitype_union = $target_scitype_union"
    end

    return SupervisedTask(X, y,
                          is_probabilistic, 
                          input_scitypes,
                          target,
                          input_scitype_union, target_scitype_union,
                          input_is_multivariate)
end

function SupervisedTask(; data=nothing, is_probabilistic=false, target=nothing, ignore=Symbol[], verbosity=1)

    target_as_input = target

    data != nothing ||
        error("You must specify data=... or use SupervisedTask(X, y, ...).")

    is_probabilistic != nothing ||
        error("You must specify is_probabilistic=true or is_proabilistic=false. ")

    if ignore isa Symbol
        ignore = [ignore, ]
    end
    
    names = schema(data).names |> collect
    
    issubset(ignore, names) ||
        error("ignore=$ignore contains a column name not encountered "*
              "in supplied data.")

    target != nothing ||
        error("You must specify target=... (use Symbol or Vector{Symbol}) ")

    if target isa Vector
        issubset(target, names) ||
            error("One or more specified target columns missing from "*
                  "supplied data. ")
        yrowtable = Tables.rowtable(selectcols(data, target))
        y = values.(yrowtable)
    else
        target in schema(data).names ||
            error("Specificed target not found in data. ")
        y = selectcols(data, target)
        target = [target, ]
    end

    features = filter(names) do ftr
        !(ftr in target) && !(ftr in ignore)
    end

    if length(features) == 1
        X = selectcols(data, features[1])
        input_is_multivariate = false
    else
        X = selectcols(data, features)
        input_is_multivariate = true
    end
    
    return SupervisedTask(X, y;
                          is_probabilistic=is_probabilistic,
                          input_is_multivariate=input_is_multivariate,
                          target=target_as_input,
                          verbosity=verbosity)
end



## RUDIMENTARY TASK OPERATIONS

nrows(task::MLJTask) = nrows(task.X)
Base.eachindex(task::MLJTask) = Base.OneTo(nrows(task))
nfeatures(task::MLJTask) = length(schema(task.X).names)

X_(task::MLJTask) = task.X
y_(task::SupervisedTask) = task.y

X_and_y(task::SupervisedTask) = (X_(task), y_(task))

# make tasks callable:
(task::UnsupervisedTask)() = X_(task)
(task::SupervisedTask)() = X_and_y(task)




