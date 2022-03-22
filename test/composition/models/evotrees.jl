import CategoricalArrays
import EvoTrees
import MLJ
import ScientificTypes

const EvoTreeRegressor = MLJ.@load EvoTreeRegressor pkg=EvoTrees

function get_classes(; levels, pool)
    num_levels  = length(levels)
    pool_length = length(pool)
    if num_levels != pool_length
        msg = "`num_levels` is `$(num_levels)` but `pool_length` is `$(pool_length)`"
        throw(ArgumentError(msg))
    end
    if num_levels > 2
        string(
            "There are $(num_levels) levels, but the ",
            "`RegressorToBinaryClassifier` wrapper only supports binary ",
            "classification",
        )
        throw(ErrorException(msg))
    end
    if num_levels < 2
        string(
            "Target `y` must have two classes in its pool, even if only one ",
            "class is manifest",
        )
        throw(ErrorException(msg))
    end
    negative_class = levels[1]
    positive_class = levels[2]
    return (; negative_class, positive_class)
end

function get_levels_and_pool(y)
    levels = MLJ.levels(y)
    pool = CategoricalArrays.pool(y)
    classes = get_classes(; levels, pool)
    negative_class = classes.negative_class
    positive_class = classes.positive_class
    scitype = MLJ.scitype(y)
    if scitype <: AbstractVector{MLJ.OrderedFactor{2}}
    elseif scitype <: AbstractVector{<:MLJ.Multiclass{2}}
        warning = string(
            "Taking positive class as $(positive_class)",
            " and negative class as $(negative_class)",
            ". Coerce target to `OrderedFactor{2}` to suppress this warning, ",
            "ensuring that positive class > negative class",
        )
        @warn warning
    else
        msg = "Unsupported scitype: $(scitype)"
        throw(ArgumentError(msg))
    end
    return (; levels, pool)
end

function transform(::Type{T}, y_categorical::AbstractVector; levels, pool) where {T}
    classes = get_classes(; levels, pool)
    negative_class = classes.negative_class
    positive_class = classes.positive_class
    n = length(y_categorical)
    y_numerical = Vector{T}(undef, n)
    for i in 1:n
        cat = y_categorical[i]
        if cat == positive_class
            num = T(1)
        elseif cat == negative_class
            num = T(0)
        else
            msg = "Invalid value: $(cat)"
            throw(ErrorException(msg))
        end
        y_numerical[i] = num
    end
    return y_numerical
end

function inverse_transform(y_single_probability::AbstractVector; levels, pool)
    classes = get_classes(; levels, pool)
    negative_class = classes.negative_class
    positive_class = classes.positive_class
    n = length(y_single_probability)
    y_probability_distribution = Vector{MLJ.UnivariateFinite}(undef, n)
    for i in 1:n
        single_prob = y_single_probability[i]
        prob_dist = MLJ.UnivariateFinite(
            levels,
            single_prob;
            pool,
            augment = true,
        )
        y_probability_distribution[i] = prob_dist
    end
    return y_probability_distribution
end

X, y = MLJ.@load_crabs

levels_and_pool = get_levels_and_pool(y)
levels = levels_and_pool.levels
pool = levels_and_pool.pool

device = "cpu"
# device = "gpu"
loss = :logistic
metric = :none
atom = EvoTreeRegressor(; device, loss, metric)

traits = Dict{Symbol, Any}()
traits[:target_scitype]  = AbstractVector{<:MLJ.Finite}
traits[:predict_scitype] = AbstractVector{<:ScientificTypes.Density{<:MLJ.Finite}}
traits[:prediction_type] = :probabilistic

model = MLJ.TransformedTargetModel(
    atom;
    traits,
    target  = y -> transform(Float64, y; levels, pool),
    inverse = y -> inverse_transform( y; levels, pool),
)

nfolds = 5
resampling = MLJ.CV(; nfolds, shuffle = true)
measure = MLJ.auc # AUC = AUROC = area under the ROC curve
mach = MLJ.machine(
    model,
    X,
    y,
)
performance_evaluation = MLJ.evaluate!(
    mach;
    resampling,
    measure,
)
display(performance_evaluation)
@test only(performance_evaluation.measurement) > 0.85
