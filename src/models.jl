# # EXCEPETIONS

function message_expecting_model(X; spelling=false)
    message = "Expected a model instance but got `$X`"
    if X isa Type{<:Model}
        message *= ", which is a model *type*"
    else
        spelling && (message *= ". Perhaps you misspelled a keyword argument")
    end
    message *= ". "
    return message
end

err_expecting_model(X; kwargs...) = ArgumentError(message_expecting_model(X; kwargs...))

"""
    MLJBase.check_ismodel(model; spelling=false)

Return `nothing` if `model` is a model, throwing `err_expecting_model(X; spelling)`
otherwise. Specify `spelling=true` if there is a chance that user mispelled a keyword
argument that is being interpreted as a model.

**Private method.**

"""
check_ismodel(model; kwargs...) = model isa Model ? nothing :
    throw(err_expecting_model(model; kwargs...))
