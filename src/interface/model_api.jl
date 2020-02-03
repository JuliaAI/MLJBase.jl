# Add fallbacks for predict_* which require mean, mode, median.

const BadModeTypes   = Union{AbstractArray{Continuous},Table(Continuous)}
const BadMeanTypes   = Union{AbstractArray{<:Finite},Table(Finite)}
const BadMedianTypes = Union{AbstractArray{<:Finite},Table(Finite)}

# mode:
MMI.predict_mode(m::Probabilistic, fitres, Xnew) =
    predict_mode(m, fitres, Xnew, Val(target_scitype(m)))
MMI.predict_mode(m, fitresult, Xnew, ::Any) =
    mode.(predict(m, fitresult, Xnew))
MMI.predict_mode(m, fitresult, Xnew, ::Val{<:BadModeTypes}) =
    throw(ArgumentError("Attempting to compute mode of predictions made "*
                        "by a model expecting `Continuous` targets. "))

# mean:
MMI.predict_mean(m::Probabilistic, fitresult, Xnew) =
    predict_mean(m, fitresult, Xnew, Val(target_scitype(m)))
MMI.predict_mean(m, fitresult, Xnew, ::Any) =
    mean.(predict(m, fitresult, Xnew))
MMI.predict_mean(m, fitresult, Xnew, ::Val{<:BadMeanTypes}) =
    throw(ArgumentError("Attempting to compute mean of predictions made "*
                        "by a model expecting `Finite` targets. "))

# median:
MMI.predict_median(m::Probabilistic, fitresult, Xnew) =
    predict_median(m, fitresult, Xnew, Val(target_scitype(m)))
MMI.predict_median(m, fitresult, Xnew, ::Any) =
    median.(predict(m, fitresult, Xnew))
MMI.predict_median(m, fitresult, Xnew, ::Val{<:BadMedianTypes}) =
    throw(ArgumentError("Attempting to compute median of predictions made "*
                        "by a model expecting `Finite` targets. "))

# # operations implemented by some meta-models:
# function evaluate end

# not in MLJModelInterface as methodswith requires InteractiveUtils
MMI.implemented_methods(::FI, M::Type{<:MLJType}) =
    getfield.(methodswith(M), :name)
