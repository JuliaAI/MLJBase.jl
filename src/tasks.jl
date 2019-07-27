abstract type MLJTask <: MLJType end # `Task` already has meaning in Base

mutable struct UnsupervisedTask <: MLJTask
    X
    input_scitypes::NamedTuple
    input_scitype
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
        X = selectcols(data, features[1]) # an abstract vector
    else
        input_is_multivariate = true
        X = selectcols(data, features) # a table
    end

    input_scitype = scitype(X)
    
    if input_is_multivariate
        input_scitypes = scitypes(X)
    else
        input_scitypes = (input=scitype_union(X), )
    end
    
    if Unknown <: input_scitype
        @warn "An input feature with unknown scitype has been encountered. "*
        "Check the input_scitypes field of your task. "
        
    end

    if verbosity > 0 
        @info "input_scitype = $input_scitype"
    end

    return UnsupervisedTask(X, input_scitypes, input_scitype)
end

# X and y can be different views of a common object.
mutable struct SupervisedTask <: MLJTask 
    X                               
    y
    is_probabilistic
    input_scitypes
    target
    input_scitype
    target_scitype
end

"""
    contains_unknown(X)

In the case that X is a table, determine if one of the columns has
elements of `Unknown` scitype. In the case of an `AbstractVector` of
tuples, determine if one of the tuples has an element of `Unknown`
scitype. In the case of any other `AbstractVector`, determine if an
element has `Unknown` scitype. In all other cases, return `false`.

"""
function contains_unknown(X)
    target_scitype = scitype(X)
    if target_scitype <: Table
        types = values(scitypes(X))
        unknown_detected = !all([!(Unknown <: t) for t in types])
    elseif target_scitype <: AbstractArray{<:Tuple}
        types = scitype_union(X).types 
        unknown_detected = !all([!(Unknown <: t) for t in types])
    elseif target_scitype <: AbstractArray
        unknown_detected = Unknown <: scitype_union(X)
    else
        unknown_detected = false
    end
    return unknown_detected
end

function SupervisedTask(X, y::AbstractVector;
                        is_probabilistic=false,
                        target=:target,
                        verbosity=1)

    # is_probabilistic != nothing ||
    #     error("You must specify is_probabilistic=true or is_probabilistic=false. ")

    target_scitype = scitype(y)
    input_scitype = scitype(X)

    if input_scitype <: Table
        input_scitypes = scitypes(X)
    else
        input_scitypes = (input=scitype_union(X),)
    end
    
    unknown_detected = contains_unknown(input_scitype)||
    contains_unknown(target_scitype)

    if unknown_detected
        @warn "An Unknown scitype has been encountered. "*
        "Check the input_scitypes and target_scitype fields of "*
        "your task. "
    end

    if verbosity > 0 
        @info "\nis_probabilistic = $is_probabilistic\ninput_scitype = "*
        "$input_scitype \ntarget_scitype = $target_scitype"
    end

    return SupervisedTask(X, y,
                          is_probabilistic, 
                          input_scitypes,
                          target,
                          input_scitype, target_scitype)
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




