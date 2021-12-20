# Add fallbacks for predict_* which require mean, mode, median.

const BadMeanTypes   = Union{AbstractArray{<:Finite},Table(Finite)}
const BadMedianTypes = Union{AbstractArray{<:Finite},Table(Finite)}

const err_wrong_target_scitype(actual_scitype) = ArgumentError(
    "Attempting to compute mode of predictions made "*
    "by a model with `$actual_scitype` targets. ")

# mode:
predict_mode(m, fitresult, Xnew) =
    mode.(predict(m, fitresult, Xnew))

# mean:
predict_mean(m, fitresult, Xnew) =
    predict_mean(m, fitresult, Xnew, target_scitype(m))
predict_mean(m, fitresult, Xnew, ::Any) =
    mean.(predict(m, fitresult, Xnew))
predict_mean(m, fitresult, Xnew, ::Type{<:BadMeanTypes}) =
    throw(err_wrong_target_scitype(Finite))

# median:
predict_median(m, fitresult, Xnew) =
    predict_median(m, fitresult, Xnew, target_scitype(m))
predict_median(m, fitresult, Xnew, ::Any) =
    median.(predict(m, fitresult, Xnew))
predict_median(m, fitresult, Xnew, ::Type{<:BadMedianTypes}) =
    throw(err_wrong_target_scitype(Finite))

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
