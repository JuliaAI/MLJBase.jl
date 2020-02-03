# Add fallbacks for predict_* which require mean, mode, median.

const BadModeTypes   = Union{AbstractArray{Continuous},Table(Continuous)}
const BadMeanTypes   = Union{AbstractArray{<:Finite},Table(Finite)}
const BadMedianTypes = Union{AbstractArray{<:Finite},Table(Finite)}

# mode:
MMI.predict_mode(m::Probabilistic, fitres, Xnew) =
    predict_mode(m, fitres, Xnew, Val(target_scitype(model)))
MMI.predict_mode(model, fitresult, Xnew, ::Any) =
    mode.(predict(model, fitresult, Xnew))
MMI.predict_mode(model, fitresult, Xnew, ::Val{<:BadModeTypes}) =
    throw(ArgumentError("Attempting to compute mode of predictions made "*
                        "by a model expecting `Continuous` targets. "))

# mean:
MMI.predict_mean(model::Probabilistic, fitresult, Xnew) =
    predict_mean(model, fitresult, Xnew, Val(target_scitype(model)))
MMI.predict_mean(model, fitresult, Xnew, ::Any) =
    mean.(predict(model, fitresult, Xnew))
MMI.predict_mean(model, fitresult, Xnew, ::Val{<:BadMeanTypes}) =
    throw(ArgumentError("Attempting to compute mean of predictions made "*
                        "by a model expecting `Finite` targets. "))

# median:
MMI.predict_median(model::Probabilistic, fitresult, Xnew) =
    predict_median(model, fitresult, Xnew, Val(target_scitype(model)))
MMI.predict_median(model, fitresult, Xnew, ::Any) =
    median.(predict(model, fitresult, Xnew))
MMI.predict_median(model, fitresult, Xnew, ::Val{<:BadMedianTypes}) =
    throw(ArgumentError("Attempting to compute median of predictions made "*
                        "by a model expecting `Finite` targets. "))

# # operations implemented by some meta-models:
# function evaluate end

# not in MLJModelInterface as methodswith requires InteractiveUtils
MMI.implemented_methods(::FI, M::Type{<:MLJType}) =
    getfield.(methodswith(M), :name)
