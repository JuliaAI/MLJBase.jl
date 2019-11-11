# ---------------------------------------------------
## CLASSIFICATION METRICS (PROBABILISTIC PREDICTIONS)
# >> CrossEntropy
# >> BriersScore
# ---------------------------------------------------

struct CrossEntropy <: Measure end

"""
    cross_entropy(ŷ, y::AbstractVector{<:Finite})

Given an abstract vector of `UnivariateFinite` distributions `ŷ` (ie,
probabilistic predictions) and an abstract vector of true observations
`y`, return the negative log-probability that each observation would
occur, according to the corresponding probabilistic prediction.

For more information, run `info(cross_entropy)`.

"""
cross_entropy = CrossEntropy()
name(::Type{<:CrossEntropy}) = "cross_entropy"
target_scitype(::Type{<:CrossEntropy}) = AbstractVector{<:Finite}
prediction_type(::Type{<:CrossEntropy}) = :probabilistic
orientation(::Type{<:CrossEntropy}) = :loss
reports_each_observation(::Type{<:CrossEntropy}) = true
is_feature_dependent(::Type{<:CrossEntropy}) = false
supports_weights(::Type{<:CrossEntropy}) = false

# for single observation:
_cross_entropy(d, y) = -log(pdf(d, y))

function (::CrossEntropy)(ŷ::AbstractVector{<:UnivariateFinite},
                          y::AbstractVector{<:CategoricalElement})
    check_dimensions(ŷ, y)
    check_pools(ŷ, y)
    return broadcast(_cross_entropy, Any[ŷ...], y)
end

# TODO: support many distributions/samplers D below:
struct BrierScore{D} <: Measure end

"""
    brier = BrierScore(; distribution=UnivariateFinite)
    brier(ŷ, y)

Given an abstract vector of distributions `ŷ` and an abstract vector
of true observations `y`, return the corresponding Brier (aka
quadratic) scores.

Currently only `distribution=UnivariateFinite` is supported, which is
applicable to superivised models with `Finite` target scitype. In this
case, if `p(y)` is the predicted probability for a *single*
observation `y`, and `C` all possible classes, then the corresponding
Brier score for that observation is given by

``2p(y) - \\left(\\sum_{η ∈ C} p(η)^2\\right) - 1``

For more information, run `info(brier_score)`.

"""
function BrierScore(; distribution=UnivariateFinite)
    distribution == UnivariateFinite ||
        error("Only `UnivariateFinite` Brier scores currently supported. ")
    return BrierScore{distribution}()
end

name(::Type{<:BrierScore{D}}) where D = "BrierScore{$(string(D))}"
target_scitype(::Type{<:BrierScore{D}}) where D = AbstractVector{<:Finite}
prediction_type(::Type{<:BrierScore}) = :probabilistic
orientation(::Type{<:BrierScore}) = :score
reports_each_observation(::Type{<:BrierScore}) = true
is_feature_dependent(::Type{<:BrierScore}) = false
supports_weights(::Type{<:BrierScore}) = true

# For single observations (no checks):

# UnivariateFinite:
function brier_score(d::UnivariateFinite, y)
    levels = classes(d)
    pvec = broadcast(pdf, d, levels)
    offset = 1 + sum(pvec.^2)
    return 2*pdf(d, y) - offset
end

# For multiple observations:

# UnivariateFinite:
function (::BrierScore{<:UnivariateFinite})(
    ŷ::AbstractVector{<:UnivariateFinite},
    y::AbstractVector{<:CategoricalElement})
    check_dimensions(ŷ, y)
    check_pools(ŷ, y)
    return broadcast(brier_score, Any[ŷ...], y)
end

function (score::BrierScore{<:UnivariateFinite})(
    ŷ, y, w::AbstractVector{<:Real})
    check_dimensions(y, w)
    return w .* score(ŷ, y) ./ (sum(w)/length(y))
end


# ---------------------------------------------------
## CLASSIFICATION METRICS (DETERMINISTIC PREDICTIONS)
# ==> MisclassificationRate / MCR
# ==> Accuracy
# Binary only
# = order independent
# ==> AUC
# = order dependent
# ==> Recall
# ==> Precision
# ---------------------------------------------------

struct MisclassificationRate <: Measure end

"""
misclassification_rate(ŷ, y)
misclassification_rate(ŷ, y, w)
misclassification_rate(conf_mat)

Returns the rate of misclassification of the (point) predictions `ŷ`,
given true observations `y`, optionally weighted by the weights
`w`. All three arguments must be abstract vectors of the same length.
A confusion matrix can also be passed as argument.

For more information, run `info(misclassification_rate)`.
You can also equivalently use `mcr`.
"""
const misclassification_rate = MisclassificationRate()
const mcr = misclassification_rate

name(::Type{<:MisclassificationRate}) = "misclassification_rate"
target_scitype(::Type{<:MisclassificationRate}) = AbstractVector{<:Finite}
prediction_type(::Type{<:MisclassificationRate}) = :deterministic
orientation(::Type{<:MisclassificationRate}) = :loss
reports_each_observation(::Type{<:MisclassificationRate}) = false
is_feature_dependent(::Type{<:MisclassificationRate}) = false
supports_weights(::Type{<:MisclassificationRate}) = true

const MCR = MisclassificationRate

(::MCR)(ŷ::AbstractVector{<:CategoricalElement},
        y::AbstractVector{<:CategoricalElement}) = mean(y .!= ŷ)

(::MCR)(ŷ::AbstractVector{<:CategoricalElement},
        y::AbstractVector{<:CategoricalElement},
        w::AbstractVector{<:Real}) = sum((y .!= ŷ) .* w) / sum(w)

(::MCR)(cm::ConfusionMatrix) = 1.0 - sum(diag(cm.mat)) / sum(cm.mat)


struct Accuracy <: Measure end

"""
accuracy(ŷ, y)
accuracy(ŷ, y, w)
accuracy(conf_mat)

Returns the accuracy of the (point) predictions `ŷ`,
given true observations `y`, optionally weighted by the weights
`w`. All three arguments must be abstract vectors of the same length.

For more information, run `info(accuracy)`.
"""
const accuracy = Accuracy()

(::Accuracy)(args...) = 1.0 - misclassification_rate(args...)
(::Accuracy)(m::ConfusionMatrix) = sum(diag(m.mat)) / sum(m.mat)

metadata_measure(Accuracy;
    name="accuracy",
    target_scitype=AbstractVector{<:Finite},
    prediction_type=:deterministic,
    orientation=:score,
    reports_each_observation=false,
    is_feature_dependent=false,
    supports_weights=true)

## Binary but order independent

struct AUC <: Measure end
const auc = AUC()

function (::AUC)(ŷ::AbstractVector{<:UnivariateFinite},
                 y::AbstractVector{<:CategoricalElement})
    # implementation drawn from the ranked comparison in
    # ht_tps://blog.revolutionanalytics.com/2016/11/calculating-auc.html
    label_1 = levels(y)[1]
    scores  = pdf.(ŷ, label_1)
    ranking = sortperm(scores, rev=true)

    # sorted scores
    scores = scores[ranking]
    mask_1 = (y[ranking] .== label_1)

    scores_1 = scores[mask_1]
    scores_2 = scores[.!mask_1]

    n_1 = length(scores_1)
    n_2 = length(scores_2)

    M = 0.0
    for i in n_1:-1:1, j in 1:n_2
        M += (1 + sign(scores_1[i] - scores_2[j]))/2
    end
    auc = 1 - M / (n_1 * n_2)
    return auc
end

metadata_measure(AUC;
    name="auc",
    target_scitype=AbstractVector{<:Finite},
    prediction_type=:probabilistic,
    orientation=:score,
    reports_each_observation=false,
    is_feature_dependent=false,
    supports_weights=false)

## Binary and order dependent

const CM2 = ConfusionMatrix{2}

struct Recall <: Measure;      rev::Union{Nothing,Bool}; end
struct Precision <: Measure;   rev::Union{Nothing,Bool}; end
struct Specificity <: Measure; rev::Union{Nothing,Bool}; end
struct FScore{β} <: Measure;   rev::Union{Nothing,Bool}; end

Recall(;     rev=nothing)         = Recall(rev)
Precision(;  rev=nothing)         = Precision(rev)
Specificity(;rev=nothing)         = Specificity(rev)
FScore{β}(;  rev=nothing) where β = FScore{β}(rev)

const recall      = Recall()
const sensitivity = recall

const specificity = Specificity()
const selectivity = specificity

# nothing for precision as there is Base.precision

const f1score = FScore{1}()

metadata_measure.((Recall, Precision, Specificity, FScore);
    target_scitype=AbstractVector{<:Finite},
    prediction_type=:deterministic,
    orientation=:score,
    reports_each_observation=false,
    is_feature_dependent=false,
    supports_weights=false)

# adjustments
name(::Type{<:Recall})            = "recall"
name(::Type{<:Precision})         = "precision"
name(::Type{<:Specificity})       = "specificity"
name(::Type{<:FScore{β}}) where β = "F$β-score"

### auxilliary functions
# they are exported for convenience but ONLY apply to confusion matrix
# to avoi ambiguities with the label ordering

tp(m::CM2) = m[2,2]
tn(m::CM2) = m[1,1]
fp(m::CM2) = m[2,1]
fn(m::CM2) = m[1,2]

tpr(m::CM2) = tp(m) / (tp(m) + fn(m))
tnr(m::CM2) = tn(m) / (tn(m) + fp(m))

fdr(m::CM2) = fp(m) / (tp(m) + fp(m))

(r::Recall)(m::CM2)      = tpr(m)
(s::Specificity)(m::CM2) = tnr(m)
(p::Precision)(m::CM2)   = 1.0 - fdr(m)
Base.precision(m::CM2)   = m |> Precision()

function (f::FScore{β})(m::CM2) where β
    β2   = β^2
    prec = precision(m)
    rec  = recall(m)
    return (1 + β2) * (prec * rec) / (β2 * prec + rec)
end

for M in (Recall, Specificity, Precision, FScore)
    (m::M)(ŷ, y) = confmat(ŷ, y; rev=m.rev) |> m
end

Base.precision(ŷ, y) = confmat(ŷ, y) |> Precision()
