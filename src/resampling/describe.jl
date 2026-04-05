"""
    individualize(iterator, delim="_")

Given an `iterator` of abstract strings, return a vector of the same length, with numeric
suffixes that make the new elements unique, as in the following example:

```julia-repl
julia> individualize(["cat", "dog", "cat", "mouse", "cat", "mouse"])
6-element Vector{String}:
 "cat_1"
 "dog"
 "cat_2"
 "mouse_1"
 "cat_3"
 "mouse_2"
```

"""
function individualize(iterator, delim="_")
    occurences_given_string = StatsBase.countmap(iterator)
    d = deepcopy(occurences_given_string)
    map(iterator) do s
        occurences_given_string[s] == 1 && return s
        digit = occurences_given_string[s] - d[s] + 1
        d[s] = d[s] - 1
        return "$s$delim$digit"
    end
end

"""
    describe(evaluation::MLJBase.AbstractPerformanceEvaluation)

Return a named tuple summarizing an MLJ performance evaluation, as returned by the methods
[`evaluate`](@ref) and [`evaluate!`](@ref). The summary includes all aggregated
measurements and their 95% radii of uncertainty. See also [`PerformanceEvaluation`](@ref).

This is particularly useful for tabulating multiple evaluations, as shown in the following
example, which assumes you have MLJ, NearestNeighborModels, and DecisionTree in your
package environment.

```julia-repl
using MLJ
X, y = @load_iris # a vector and a table

# instantiate two models:
knn = (@load KNNClassifier pkg=NearestNeighborModels)()
tree = (@load DecisionTreeClassifier pkg=DecisionTree)()

named_models = [
    "Dummy" => ConstantClassifier(),  # a built-in model
    "K-nearest neighbors" => knn,
    "Decision Tree" => tree,
]
performance_evaluations = evaluate(named_models, X, y; measures=[brier_score, accuracy])
julia> describe(performance_evaluations[1])
(tag = "Dummy", BrierScore = -0.573 ± 0.1, Accuracy = 0.33 ± 0.23)

table = describe.(performance_evaluations)
julia> pretty(table)
┌─────────────────────┬──────────────────────┬──────────────────────┐
│ tag                 │ BrierScore           │ Accuracy             │
│ String              │ Measurement{Float64} │ Measurement{Float64} │
│ Textual             │ Continuous           │ Continuous           │
├─────────────────────┼──────────────────────┼──────────────────────┤
│ Dummy               │ -0.573±0.1           │ 0.33±0.23            │
│ K-nearest neighbors │ -0.21±0.21           │ 0.92±0.18            │
│ Decision Tree       │ -0.00118977±0.0      │ 1.0±0.0              │
└─────────────────────┴──────────────────────┴──────────────────────┘
```


"""
function DataAPI.describe(e::AbstractPerformanceEvaluation)
    key_value_pairs = Any[:tag=>e.tag]
    measure_names = individualize(
        map(e.measure) do measure
            split(_repr_(measure), "(") |> first
        end,
        "",
    )
    for (i, name) in enumerate(measure_names)
        value = e.measurement[i]
        δ =  e.uncertainty_radius_95[i]
        if !isnothing(δ) && δ isa Real && !isinf(δ)
            # decorate with uncertainty radius:
            value = Measurements.measurement(value, δ)
        end
        push!(key_value_pairs, Symbol(name) => value)
    end
    NamedTuple(key_value_pairs)
end
