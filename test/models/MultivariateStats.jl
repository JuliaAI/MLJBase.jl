export RidgeRegressor, PCA

import MLJBase
import MLJBase: @mlj_model, metadata_model, metadata_pkg
# using Distances
using LinearAlgebra
using Tables, ScientificTypes

import MultivariateStats

const MS = MultivariateStats

struct LinearFitresult{F} <: MLJBase.MLJType
    coefficients::Vector{F}
    intercept::F
end

const RIDGE_DESCR = "Ridge regressor with regularization parameter lambda. Learns a linear regression with a penalty on the l2 norm of the coefficients."
const PCA_DESCR = "Principal component analysis. Learns a linear transformation to project the data  on a lower dimensional space while preserving most of the initial variance."

####
#### RIDGE
####

"""
RidgeRegressor(; lambda=1.0)

$RIDGE_DESCR

## Parameters

* `lambda=1.0`: non-negative parameter for the regularization strength.
"""
@mlj_model mutable struct RidgeRegressor <: MLJBase.Deterministic
    lambda::Real = 1.0::(_ ≥ 0)
end

function MLJBase.fit(model::RidgeRegressor, verbosity::Int, X, y)
    Xmatrix   = MLJBase.matrix(X)
    features  = Tables.schema(X).names
    θ         = MS.ridge(Xmatrix, y, model.lambda)
    coefs     = θ[1:end-1]
    intercept = θ[end]

    fitresult = LinearFitresult(coefs, intercept)
    report    = NamedTuple()
    cache     = nothing

    return fitresult, cache, report
end

MLJBase.fitted_params(::RidgeRegressor, fr) =
    (coefficients=fr.coefficients, intercept=fr.intercept)

function MLJBase.predict(::RidgeRegressor, fr, Xnew)
    Xmatrix = MLJBase.matrix(Xnew)
    return Xmatrix * fr.coefficients .+ fr.intercept
end

####
#### PCA
####

const PCAFitResultType = MS.PCA

"""
PCA(; maxoutdim=nothing, method=:auto, pratio=0.99, mean=nothing)

$PCA_DESCR

## Parameters

* `maxoutdim=nothing`: maximum number of output dimensions, unconstrained if nothing.
* `method=:auto`: method to use to solve the problem, one of `:auto`, `:cov` or `:svd`
* `pratio=0.99`: ratio of variance preserved
* `mean=nothing`: if set to nothing centering will be computed and applied, if set to `0` no centering (assumed pre-centered), if a vector is passed, the centering is done with that vector.
"""
@mlj_model mutable struct PCA <: MLJBase.Unsupervised
    maxoutdim::Union{Nothing,Int} = nothing::(_ === nothing || _ ≥ 1)
    method::Symbol  = :auto::(_ in (:auto, :cov, :svd))
    pratio::Float64 = 0.99::(0.0 < _ ≤ 1.0)
    mean::Union{Nothing, Real, Vector{Float64}} = nothing::(_ === nothing || (_ isa Real && iszero(_)) || true)
end

function MLJBase.fit(model::PCA, verbosity::Int, X)
    Xarray = MLJBase.matrix(X)
    mindim = minimum(size(Xarray))

    maxoutdim = model.maxoutdim === nothing ? mindim : model.maxoutdim

    # NOTE: copy/transpose
    fitresult = MS.fit(MS.PCA, permutedims(Xarray);
                       method=model.method,
                       pratio=model.pratio,
                       maxoutdim=maxoutdim,
                       mean=model.mean)

    cache = nothing
    report = (indim=MS.indim(fitresult),
              outdim=MS.outdim(fitresult),
              mean=MS.mean(fitresult),
              principalvars=MS.principalvars(fitresult),
              tprincipalvar=MS.tprincipalvar(fitresult),
              tresidualvar=MS.tresidualvar(fitresult),
              tvar=MS.tvar(fitresult))

    return fitresult, cache, report
end

MLJBase.fitted_params(::PCA, fr) = (projection=fr,)


function MLJBase.transform(::PCA, fr::PCAFitResultType, X)
    # X is n x d, need to transpose and copy twice...
    Xarray = MLJBase.matrix(X)
    Xnew   = permutedims(MS.transform(fr, permutedims(Xarray)))
    return MLJBase.table(Xnew, prototype=X)
end


####
#### METADATA
####

metadata_pkg.((RidgeRegressor, PCA),
              name="MultivariateStats",
              uuid="6f286f6a-111f-5878-ab1e-185364afe411",
              url="https://github.com/JuliaStats/MultivariateStats.jl",
              license="MIT",
              julia=true,
              is_wrapper=false)

metadata_model(RidgeRegressor,
               input=MLJBase.Table(MLJBase.Continuous),
               target=AbstractVector{MLJBase.Continuous},
               weights=false,
               descr=RIDGE_DESCR)

metadata_model(PCA,
               input=MLJBase.Table(MLJBase.Continuous),
               target=MLJBase.Table(MLJBase.Continuous),
               weights=false,
               descr=PCA_DESCR)

