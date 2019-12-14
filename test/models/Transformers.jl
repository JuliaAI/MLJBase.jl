## CONSTANTS

export FeatureSelector,
    UnivariateStandardizer, Standardizer,
    UnivariateBoxCoxTransformer,
    OneHotEncoder, UnivariateDiscretizer

using Statistics

const N_VALUES_THRESH = 16 # for BoxCoxTransformation
const CategoricalElement = MLJBase.CategoricalElement

## DESCRIPTIONS (see also metadata at the bottom)

const FEATURE_SELECTOR_DESCR = "Filter features (columns) of a table by name."
const UNIVARIATE_STD_DESCR = "Standardize (whiten) univariate data."
const UNIVARIATE_DISCR_DESCR = "Discretize continuous variables via quantiles."
const STANDARDIZER_DESCR = "Standardize (whiten) data."
const UNIVARIATE_BOX_COX_DESCR = "Box-Cox transformation of univariate data."
const ONE_HOT_DESCR = "One-Hot-Encoding of the categorical data."

##
## FOR FEATURE (COLUMN) SELECTION
##

"""
FeatureSelector(features=Symbol[])

An unsupervised model for filtering features (columns) of a table.
Only those features encountered during fitting will appear in
transformed tables if `features` is empty (the default).
Alternatively, if a non-empty `features` is specified, then only the
specified features are used. Throws an error if a recorded or
specified feature is not present in the transformation input.
"""
mutable struct FeatureSelector <: MLJBase.Unsupervised
    features::Vector{Symbol}
end

FeatureSelector(; features=Symbol[]) = FeatureSelector(features)

function MLJBase.fit(transformer::FeatureSelector, verbosity::Int, X)
    namesX = collect(Tables.schema(X).names)
    if isempty(transformer.features)
        fitresult = namesX
    else
        all(e -> e in namesX, transformer.features) ||
            throw(error("Attempting to select non-existent feature(s)."))
        fitresult = transformer.features
    end
    report = NamedTuple()
    return fitresult, nothing, report
end

MLJBase.fitted_params(::FeatureSelector, fitresult) = (features_to_keep=fitresult,)

function MLJBase.transform(transformer::FeatureSelector, features, X)
    all(e -> e in Tables.schema(X).names, features) ||
        throw(error("Supplied frame does not admit previously selected features."))
    return MLJBase.selectcols(X, features)
end

##
## UNIVARIATE Discretizer
##

# helper functions
# TODO: move these to MLJBase/src/data.jl

const message1 = "Attempting to transform a level not in pool of specified "*
   "categorical element. "

# Transform a raw level `x` in the pool of some categorical element,
# `element`, into a categorical element (with the same pool):
function MLJBase.transform(element::C, x) where C<:CategoricalElement
    pool = element.pool
    x in levels(pool) || error(message1)
    ref = pool.invindex[x]
    return C(ref, pool)
end

# Transform ordinary array `X` into a categorical array with the same
# pool as the categorical element `element`:
function MLJBase.transform(element::CategoricalElement,
                           X::AbstractArray{T,N}) where {T,N}
    pool = element.pool

    levels_presented = unique(X)
    issubset(levels_presented, levels(pool)) || error(message1)
    Missing <: T &&
        error("Missing values not supported. ")

    refs = broadcast(x -> pool.invindex[x], X)

    return CategoricalArray{T,N}(refs, pool)
end

reftype(::CategoricalArray{<:Any,<:Any,R}) where R = R

"""
UnivariateDiscretizer(n_classes=512)

Returns an `MLJModel` for for discretizing any continuous vector `v`
 (`scitype(v) <: AbstractVector{Continuous}`), where `n_classes`
 describes the resolution of the discretization.

Transformed output `w` is a vector of ordered factors (`scitype(w) <:
 AbstractVector{<:OrderedFactor}`). Specifically, `w` is a
 `CategoricalVector`, with element type
 `CategoricalValue{R,R}`, where `R<Unsigned` is optimized.

The transformation is chosen so that the vector on which the
 transformer is fit has, in transformed form, an approximately uniform
 distribution of values.

### Example

    using MLJ
    t = UnivariateDiscretizer(n_classes=10)
    discretizer = machine(t, randn(1000))
    fit!(discretizer)
    v = rand(10)
    w = transform(discretizer, v)
    v_approx = inverse_transform(discretizer, w) # reconstruction of v from w

"""
mutable struct UnivariateDiscretizer <:MLJBase.Unsupervised
    n_classes::Int
end

UnivariateDiscretizer(; n_classes=512) = UnivariateDiscretizer(n_classes)

struct UnivariateDiscretizerResult{C}
    odd_quantiles::Vector{Float64}
    even_quantiles::Vector{Float64}
    element::C
end

function MLJBase.fit(transformer::UnivariateDiscretizer, verbosity::Int,X)
    n_classes = transformer.n_classes
    quantiles = quantile(X, Array(range(0, stop=1, length=2*n_classes+1)))
    clipped_quantiles = quantiles[2:2*n_classes] # drop 0% and 100% quantiles

    # odd_quantiles for transforming, even_quantiles used for
    # inverse_transforming:
    odd_quantiles = clipped_quantiles[2:2:(2*n_classes-2)]
    even_quantiles = clipped_quantiles[1:2:(2*n_classes-1)]

    # determine optimal reference type for encoding as categorical:
    R = reftype(categorical(1:n_classes, true))
    output_prototype = categorical(R(1):R(n_classes), true, ordered=true)
    element = output_prototype[1]

    cache  = nothing
    report = NamedTuple()

    res = UnivariateDiscretizerResult(odd_quantiles, even_quantiles, element)
    return res, cache, report
end

# acts on scalars:
function transform_to_int(
    result::UnivariateDiscretizerResult{<:CategoricalElement{R}},
    r::Real) where R

    k = R(1)
    for q in result.odd_quantiles
        if r > q
            k += R(1)
        end
    end
    return k
end

# transforming scalars:
MLJBase.transform(::UnivariateDiscretizer, result, r::Real) =
    transform(result.element, transform_to_int(result, r))

# transforming vectors:
function MLJBase.transform(::UnivariateDiscretizer, result, v)
   w = [transform_to_int(result, r) for r in v]
   return transform(result.element, w)
end

# inverse_transforming raw scalars:
function MLJBase.inverse_transform(
    transformer::UnivariateDiscretizer, result , k::Integer)
    k <= transformer.n_classes && k > 0 ||
        error("Cannot transform an integer outside the range "*
              "`[1, n_classes]`, where `n_classes = $(transformer.n_classes)`")
    return result.even_quantiles[k]
end

# inverse transforming a categorical value:
function MLJBase.inverse_transform(
    transformer::UnivariateDiscretizer, result, e::CategoricalElement)
    k = get(e)
    return inverse_transform(transformer, result, k)
end

# inverse transforming raw vectors:
MLJBase.inverse_transform(transformer::UnivariateDiscretizer, result,
                          w::AbstractVector{<:Integer}) =
      [inverse_transform(transformer, result, k) for k in w]

# inverse transforming vectors of categorical elements:
function MLJBase.inverse_transform(transformer::UnivariateDiscretizer, result,
                          wcat::AbstractVector{<:CategoricalElement})
    w = MLJBase.int(wcat)
    return [inverse_transform(transformer, result, k) for k in w]
end


## UNIVARIATE STANDARDIZATION

"""
UnivariateStandardizer()

Unsupervised model for standardizing (whitening) univariate data.
"""
mutable struct UnivariateStandardizer <: MLJBase.Unsupervised end

function MLJBase.fit(transformer::UnivariateStandardizer, verbosity::Int,
             v::AbstractVector{T}) where T<:Real
    std(v) > eps(Float64) ||
        @warn "Extremely small standard deviation encountered in standardization."
    fitresult = (mean(v), std(v))
    cache = nothing
    report = NamedTuple()
    return fitresult, cache, report
end

# for transforming single value:
function MLJBase.transform(transformer::UnivariateStandardizer, fitresult, x::Real)
    mu, sigma = fitresult
    return (x - mu)/sigma
end

# for transforming vector:
MLJBase.transform(transformer::UnivariateStandardizer, fitresult, v) =
              [transform(transformer, fitresult, x) for x in v]

# for single values:
function MLJBase.inverse_transform(transformer::UnivariateStandardizer, fitresult, y::Real)
    mu, sigma = fitresult
    return mu + y*sigma
end

# for vectors:
MLJBase.inverse_transform(transformer::UnivariateStandardizer, fitresult, w) =
    [inverse_transform(transformer, fitresult, y) for y in w]

## STANDARDIZATION OF ORDINAL FEATURES OF TABULAR DATA

"""
Standardizer(; features=Symbol[])

Unsupervised model for standardizing (whitening) the columns of
tabular data. If `features` is empty then all columns `v` for which
all elements have `Continuous` scitypes are standardized. For
different behaviour (e.g. standardizing counts as well), specify the
names of features to be standardized.

    using DataFrames
    X = DataFrame(x1=[0.2, 0.3, 1.0], x2=[4, 2, 3])
    stand_model = Standardizer()
    transform(fit!(machine(stand_model, X)), X)

    3×2 DataFrame
    │ Row │ x1        │ x2    │
    │     │ Float64   │ Int64 │
    ├─────┼───────────┼───────┤
    │ 1   │ -0.688247 │ 4     │
    │ 2   │ -0.458831 │ 2     │
    │ 3   │ 1.14708   │ 3     │

"""
mutable struct Standardizer <: MLJBase.Unsupervised
    features::Vector{Symbol} 
end

Standardizer(; features=Symbol[]) = Standardizer(features)

function MLJBase.fit(transformer::Standardizer, verbosity::Int, X::Any)
    all_features = Tables.schema(X).names
    mach_types   = collect(eltype(selectcols(X, c)) for c in all_features)

    # determine indices of all_features to be transformed
    if isempty(transformer.features)
        cols_to_fit = filter!(eachindex(all_features) |> collect) do j
            mach_types[j] <: AbstractFloat
        end
    else
        issubset(transformer.features, all_features) ||
            @warn "Some specified features not present in table to be fit. "
        cols_to_fit = filter!(eachindex(all_features) |> collect) do j
            all_features[j] in transformer.features && mach_types[j] <: Real
        end
    end

    fitresult_given_feature = Dict{Symbol,Tuple{Float64,Float64}}()

    # fit each feature
    verbosity < 2 || @info "Features standarized: "
    for j in cols_to_fit
        col_fitresult, cache, report =
            fit(UnivariateStandardizer(), verbosity - 1, selectcols(X, j))
        fitresult_given_feature[all_features[j]] = col_fitresult
        verbosity < 2 ||
            @info "  :$(all_features[j])    mu=$(col_fitresult[1])  sigma=$(col_fitresult[2])"
    end

    fitresult = fitresult_given_feature
    cache = nothing
    report = (features_fit=keys(fitresult_given_feature),)

    return fitresult, cache, report
end

MLJBase.fitted_params(::Standardizer, fitresult) = (mean_and_std_given_feature=fitresult,)

function MLJBase.transform(transformer::Standardizer, fitresult, X)
    # `fitresult` is dict of column fitresults, keyed on feature names
    features_to_be_transformed = keys(fitresult)

    all_features = Tables.schema(X).names

    all(e -> e in all_features, features_to_be_transformed) ||
        error("Attempting to transform data with incompatible feature labels.")

    col_transformer = UnivariateStandardizer()

    cols = map(all_features) do ftr
        if ftr in features_to_be_transformed
            transform(col_transformer, fitresult[ftr], selectcols(X, ftr))
        else
            selectcols(X, ftr)
        end
    end

    named_cols = NamedTuple{all_features}(tuple(cols...))

    return MLJBase.table(named_cols, prototype=X)
end

##
## UNIVARIATE BOX-COX TRANSFORMATIONS
##

function standardize(v)
    map(v) do x
        (x - mean(v))/std(v)
    end
end

function midpoints(v::AbstractVector{T}) where T <: Real
    return [0.5*(v[i] + v[i + 1]) for i in 1:(length(v) -1)]
end

function normality(v)
    n  = length(v)
    v  = standardize(convert(Vector{Float64}, v))
    # sort and replace with midpoints
    v = midpoints(sort!(v))
    # find the (approximate) expected value of the size (n-1)-ordered statistics for
    # standard normal:
    d = Distributions.Normal(0,1)
    w = map(collect(1:(n-1))/n) do x
        quantile(d, x)
    end
    return cor(v, w)
end

function boxcox(lambda, c, x::Real)
    c + x >= 0 || throw(DomainError)
    if lambda == 0.0
        c + x > 0 || throw(DomainError)
        return log(c + x)
    end
    return ((c + x)^lambda - 1)/lambda
end

boxcox(lambda, c, v::AbstractVector{T}) where T <: Real =
    [boxcox(lambda, c, x) for x in v]


"""
UnivariateBoxCoxTransformer(; n=171, shift=false)

Unsupervised model specifying a univariate Box-Cox
transformation of a single variable taking non-negative values, with a
possible preliminary shift. Such a transformation is of the form

    x -> ((x + c)^λ - 1)/λ for λ not 0
    x -> log(x + c) for λ = 0

On fitting to data `n` different values of the Box-Cox
exponent λ (between `-0.4` and `3`) are searched to fix the value
maximizing normality. If `shift=true` and zero values are encountered
in the data then the transformation sought includes a preliminary
positive shift `c` of `0.2` times the data mean. If there are no zero
values, then no shift is applied.

"""
mutable struct UnivariateBoxCoxTransformer <: MLJBase.Unsupervised
    n::Int
    shift::Bool
end

UnivariateBoxCoxTransformer(; n=171, shift=false) =
    UnivariateBoxCoxTransformer(n, shift)

function MLJBase.fit(transformer::UnivariateBoxCoxTransformer, verbosity::Int,
             v::AbstractVector{T}) where T <: Real

    m = minimum(v)
    m >= 0 || error("Cannot perform a Box-Cox transformation on negative data.")

    c = 0.0 # default
    if transformer.shift
        if m == 0
            c = 0.2*mean(v)
        end
    else
        m != 0 || error("Zero value encountered in data being Box-Cox transformed.\n"*
                        "Consider calling `fit!` with `shift=true`.")
    end

    lambdas = range(-0.4, stop=3, length=transformer.n)
    scores = Float64[normality(boxcox(l, c, v)) for l in lambdas]
    lambda = lambdas[argmax(scores)]

    return  (lambda, c), nothing, NamedTuple()
end

MLJBase.fitted_params(::UnivariateBoxCoxTransformer, fitresult) =
    (λ=fitresult[1], c=fitresult[2])

# for X scalar or vector:
MLJBase.transform(transformer::UnivariateBoxCoxTransformer, fitresult, X) =
    boxcox(fitresult..., X)

# scalar case:
function MLJBase.inverse_transform(transformer::UnivariateBoxCoxTransformer,
                           fitresult, x::Real)
    lambda, c = fitresult
    if lambda == 0
        return exp(x) - c
    else
        return (lambda*x + 1)^(1/lambda) - c
    end
end

# vector case:
function MLJBase.inverse_transform(transformer::UnivariateBoxCoxTransformer,
                           fitresult, w::AbstractVector{T}) where T <: Real
    return [inverse_transform(transformer, fitresult, y) for y in w]
end


## ONE HOT ENCODING

"""
OneHotEncoder(; features=Symbol[], drop_last=false, ordered_factor=true)

Unsupervised model for one-hot encoding all features of `Finite`
scitype, within some table. If `ordered_factor=false` then
only `Multiclass` features are considered. The features encoded are
further restricted to those in `features`, when specified and
non-empty.

If `drop_last` is true, the column for the last level of each
categorical feature is dropped. New data to be transformed may lack
features present in the fit data, but no new features can be present.

*Warning:* This transformer assumes that the elements of a categorical
 feature in new data to be transformed point to the same
 CategoricalPool object encountered during the fit.

"""
mutable struct OneHotEncoder <: MLJBase.Unsupervised
    features::Vector{Symbol}
    drop_last::Bool
    ordered_factor::Bool
end

OneHotEncoder(; features=Symbol[], drop_last=false, ordered_factor=true) =
    OneHotEncoder(features, drop_last, ordered_factor)

# we store the categorical refs for each feature to be encoded and the
# corresponing feature labels generated (called
# "names"). `all_features` is stored to ensure no new features appear
# in new input data, causing potential name clashes.
struct OneHotEncoderResult <: MLJBase.MLJType
    all_features::Vector{Symbol} # all feature labels
    ref_name_pairs_given_feature::Dict{Symbol,Vector{Pair{<:Unsigned,Symbol}}}
end

# join feature and level into new label without clashing with anything
# in all_features:
function compound_label(all_features, feature, level)
    label = Symbol(string(feature, "__", level))
    # in the (rare) case subft is not a new feature label:
    while label in all_features
        label = Symbol(string(label,"_"))
    end
    return label
end

function MLJBase.fit(transformer::OneHotEncoder, verbosity::Int, X)

    all_features = Tables.schema(X).names # a tuple not vector
    specified_features =
        isempty(transformer.features) ? collect(all_features) : transformer.features
    #
    ref_name_pairs_given_feature = Dict{Symbol,Vector{Pair{<:Unsigned,Symbol}}}()
    allowed_scitypes = ifelse(transformer.ordered_factor, Finite, Multiclass)
    col_scitypes = schema(X).scitypes
    # apply on each feature
    for j in eachindex(all_features)
        ftr = all_features[j]
        col = MLJBase.selectcols(X,j)
        T = col_scitypes[j]
        if T <: allowed_scitypes && ftr in specified_features
            ref_name_pairs_given_feature[ftr] = Pair{<:Unsigned,Symbol}[]
            shift = transformer.drop_last ? 1 : 0
            levels = MLJBase.classes(first(col))
            if verbosity > 0
                @info "Spawning $(length(levels)-shift) sub-features "*
                "to one-hot encode feature :$ftr."
            end
            for level in levels[1:end-shift]
                ref = MLJBase.int(level)
                name = compound_label(all_features, ftr, level)
                push!(ref_name_pairs_given_feature[ftr], ref => name)
            end
        end
    end

    fitresult = OneHotEncoderResult(collect(all_features),
                                    ref_name_pairs_given_feature)

    # get new feature names
    d = ref_name_pairs_given_feature
    new_features = Symbol[]
    features_to_be_transformed = keys(d)
    for ftr in all_features
        if ftr in features_to_be_transformed
            append!(new_features, last.(d[ftr]))
        else
            push!(new_features, ftr)
        end
    end

    report = (features_to_be_encoded=
              collect(keys(ref_name_pairs_given_feature)),
              new_features=new_features)
    cache = nothing

    return fitresult, cache, report
end

# If v=categorical('a', 'a', 'b', 'a', 'c') and MLJBase.int(v[1]) = ref
# then `_hot(v, ref) = [true, true, false, true, false]`
_hot(v::AbstractVector{<:CategoricalElement}, ref) = map(v) do c
    MLJBase.int(c) == ref
end

function MLJBase.transform(transformer::OneHotEncoder, fitresult, X)
    features = Tables.schema(X).names # tuple not vector
    d = fitresult.ref_name_pairs_given_feature
    # check the features match the fit result
    all(e -> e in fitresult.all_features, features) ||
        error("Attempting to transform table with feature "*
              "labels not seen in fit. ")
    new_features = Symbol[]
    new_cols = Vector[]
    features_to_be_transformed = keys(d)
    for ftr in features
        col = MLJBase.selectcols(X, ftr)
        if ftr in features_to_be_transformed
            append!(new_features, last.(d[ftr]))
            pairs = d[ftr]
            refs = first.(pairs)
            names = last.(pairs)
            cols_to_add = map(refs) do ref
                float.(_hot(col, ref))
            end
            append!(new_cols, cols_to_add)
        else
            push!(new_features, ftr)
            push!(new_cols, col)
        end
    end
    named_cols = NamedTuple{tuple(new_features...)}(tuple(new_cols)...)
    return MLJBase.table(named_cols, prototype=X)
end

##
## Metadata for all built-in transformers
##

metadata_pkg.((FeatureSelector, UnivariateStandardizer,
               UnivariateDiscretizer, Standardizer,
               UnivariateBoxCoxTransformer, OneHotEncoder),
              name="MLJBase",
              julia=true,
              license="MIT")

metadata_model(FeatureSelector,
               input=MLJBase.Table(MLJBase.Scientific),
               output=MLJBase.Table(MLJBase.Scientific),
               weights=false,
               descr=FEATURE_SELECTOR_DESCR,
               path="MLJBase.FeatureSelector")

metadata_model(UnivariateDiscretizer,
               input=AbstractVector{<:MLJBase.Continuous},
               output=AbstractVector{<:MLJBase.OrderedFactor},
               weights=false,
               descr=UNIVARIATE_DISCR_DESCR,
               path="MLJBase.UnivariateDiscretizer")

metadata_model(UnivariateStandardizer,
               input=AbstractVector{<:MLJBase.Infinite},
               output=AbstractVector{MLJBase.Continuous},
               weights=false,
               descr=UNIVARIATE_STD_DESCR,
               path="MLJBase.UnivariateStandardizer")

metadata_model(Standardizer,
               input=MLJBase.Table(MLJBase.Scientific),
               output=MLJBase.Table(MLJBase.Scientific),
               weights=false,
               descr=STANDARDIZER_DESCR,
               path="MLJBase.Standardizer")

metadata_model(UnivariateBoxCoxTransformer,
               input=AbstractVector{MLJBase.Continuous},
               output=AbstractVector{MLJBase.Continuous},
               weights=false,
               descr=UNIVARIATE_BOX_COX_DESCR,
               path="MLJBase.UnivariateBoxCoxTransformer")

metadata_model(OneHotEncoder,
               input=MLJBase.Table(MLJBase.Scientific),
               output=MLJBase.Table(MLJBase.Scientific),
               weights=false,
               descr=ONE_HOT_DESCR,
               path="MLJBase.OneHotEncoder")
