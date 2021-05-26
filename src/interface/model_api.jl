# Add fallbacks for predict_* which require mean, mode, median.

const BadModeTypes   = Union{AbstractArray{Continuous},Table(Continuous)}
const BadMeanTypes   = Union{AbstractArray{<:Finite},Table(Finite)}
const BadMedianTypes = Union{AbstractArray{<:Finite},Table(Finite)}

# mode:
predict_mode(m::Probabilistic, fitresult, Xnew) =
    predict_mode(m, fitresult, Xnew, Val(target_scitype(m)))
predict_mode(m, fitresult, Xnew, ::Any) =
    mode.(predict(m, fitresult, Xnew))
predict_mode(m, fitresult, Xnew, ::Val{<:BadModeTypes}) =
    throw(ArgumentError("Attempting to compute mode of predictions made "*
                        "by a model expecting `Continuous` targets. "))

# mean:
predict_mean(m::Probabilistic, fitresult, Xnew) =
    predict_mean(m, fitresult, Xnew, Val(target_scitype(m)))
predict_mean(m, fitresult, Xnew, ::Any) =
    mean.(predict(m, fitresult, Xnew))
predict_mean(m, fitresult, Xnew, ::Val{<:BadMeanTypes}) =
    throw(ArgumentError("Attempting to compute mean of predictions made "*
                        "by a model expecting `Finite` targets. "))

# median:
predict_median(m::Probabilistic, fitresult, Xnew) =
    predict_median(m, fitresult, Xnew, Val(target_scitype(m)))
predict_median(m, fitresult, Xnew, ::Any) =
    median.(predict(m, fitresult, Xnew))
predict_median(m, fitresult, Xnew, ::Val{<:BadMedianTypes}) =
    throw(ArgumentError("Attempting to compute median of predictions made "*
                        "by a model expecting `Finite` targets. "))

# not in MLJModelInterface as methodswith requires InteractiveUtils
MLJModelInterface.implemented_methods(::FI, M::Type{<:MLJType}) =
    getfield.(methodswith(M), :name) |> unique

# serialization fallbacks:
MLJModelInterface.save(filename, model, fitresult; kwargs...) = fitresult
MLJModelInterface.restore(filename, model, serializable_fitresult) =
                          serializable_fitresult

# to suppress inclusion of abstract types in the model registry.
for T in (:Supervised, :Unsupervised,
          :Interval, :Static, :Deterministic, :Probabilistic)
    ex = quote
        MLJModelInterface.is_wrapper(::Type{$T}) = true
    end
    eval(ex)
end

