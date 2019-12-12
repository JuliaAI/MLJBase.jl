export DecisionTreeClassifier, DecisionTreeRegressor

import MLJBase
import MLJBase: @mlj_model, metadata_pkg, metadata_model

using ScientificTypes

using CategoricalArrays

import DecisionTree 

## DESCRIPTIONS

const DTC_DESCR = "Decision Tree Classifier."
const DTR_DESCR = "Decision Tree Regressor."

## CLASSIFIER

struct TreePrinter{T}
    tree::T
end
(c::TreePrinter)(depth) = DecisionTree.print_tree(c.tree, depth)
(c::TreePrinter)() = DecisionTree.print_tree(c.tree, 5)

Base.show(stream::IO, c::TreePrinter) =
    print(stream, "TreePrinter object (call with display depth)")


"""
DecisionTreeClassifer(; kwargs...)

A variation on the CART decision tree classifier from [https://github.com/bensadeghi/DecisionTree.jl/blob/master/README.md](https://github.com/bensadeghi/DecisionTree.jl/blob/master/README.md).

Inputs are tables with ordinal columns. That is, the element scitype
of each column can be `Continuous`, `Count` or `OrderedFactor`.

Instead of predicting the mode class at each leaf, a UnivariateFinite
distribution is fit to the leaf training classes, with smoothing
controlled by an additional hyperparameter `pdf_smoothing`: If `n` is
the number of observed classes, then each class probability is
replaced by `pdf_smoothing/n`, if it falls below that ratio, and the
resulting vector of probabilities is renormalized. Smoothing is only
applied to classes actually observed in training. Unseen classes
retain zero-probability predictions.

To visualize the fitted tree in the REPL, set `verbosity=2` when
fitting, or call `report(mach).print_tree(display_depth)` where `mach`
is the fitted machine, and `display_depth` the desired
depth. Interpretting the results will require a knowledge of the
internal integer encodings of classes, which are given in
`fitted_params(mach)` (which also stores the raw learned tree object
from the DecisionTree.jl algorithm).

For post-fit pruning, set `post-prune=true` and set
`min_purity_threshold` appropriately. Other hyperparameters as per
package documentation cited above. 


"""
@mlj_model mutable struct DecisionTreeClassifier <: MLJBase.Probabilistic
    pruning_purity::Float64         = 1.0::(_ ≤ 1)
    max_depth::Int                  = (-)(1)::(_ ≥ -1)
    min_samples_leaf::Int           = 1::(_ ≥ 0)
    min_samples_split::Int          = 2::(_ ≥ 2)
    min_purity_increase::Float64    = 0.0::(_ ≥ 0)
    n_subfeatures::Int              = 0::(_ ≥ 0)
    display_depth::Int              = 5::(_ ≥ 1)
    post_prune::Bool                = false
    merge_purity_threshold::Float64 = 0.9::(0 ≤ _ ≤ 1)
    pdf_smoothing::Float64          = 0.05::(0 ≤ _ ≤ 1)
end

#> A required `fit` method returns `fitresult, cache, report`. (Return
#> `cache=nothing` unless you are overloading `update`)
function MLJBase.fit(model::DecisionTreeClassifier, verbosity::Int, X, y)
    Xmatrix = MLJBase.matrix(X)
    yplain  = MLJBase.int(y)
    classes_seen = filter(in(unique(y)), MLJBase.classes(y[1]))
    integers_seen = MLJBase.int(classes_seen) #unique(yplain)

    tree = DecisionTree.build_tree(yplain, Xmatrix,
                                   model.n_subfeatures,
                                   model.max_depth,
                                   model.min_samples_leaf,
                                   model.min_samples_split,
                                   model.min_purity_increase)
    if model.post_prune
        tree = DecisionTree.prune_tree(tree, model.merge_purity_threshold)
    end

    verbosity < 2 || DecisionTree.print_tree(tree, model.display_depth)

    fitresult = (tree, classes_seen, integers_seen)

    #> return package-specific statistics (eg, feature rankings,
    #> internal estimates of generalization error) in `report`, which
    #> should be a named tuple with the same type every call (can have
    #> empty values):

    cache = nothing
    report = (classes_seen=classes_seen,
              print_tree=TreePrinter(tree))

    return fitresult, cache, report
end

function get_encoding(classes_seen)
    a_cat_element = classes_seen[1]
    return Dict(c => MLJBase.int(c) for c in MLJBase.classes(a_cat_element))
end

MLJBase.fitted_params(::DecisionTreeClassifier, fitresult) =
    (tree_or_leaf=fitresult[1], encoding=get_encoding(fitresult[2]))

function smooth(prob_vector, smoothing)
    threshold = smoothing/length(prob_vector)
    smoothed_vector = map(prob_vector) do p
        p < threshold ? threshold : p
    end
    smoothed_vector = smoothed_vector/sum(smoothed_vector)
    return smoothed_vector
end

function MLJBase.predict(model::DecisionTreeClassifier
                     , fitresult
                     , Xnew)
    Xmatrix = MLJBase.matrix(Xnew)

    tree, classes_seen, integers_seen = fitresult

    y_probabilities =
        DecisionTree.apply_tree_proba(tree, Xmatrix, integers_seen)

    return [MLJBase.UnivariateFinite(classes_seen,
                                     smooth(y_probabilities[i,:],
                                            model.pdf_smoothing))
            for i in 1:size(y_probabilities, 1)]
end


## REGRESSOR

"""
DecisionTreeRegressor(; kwargs...)

CART decision tree classifier from
[https://github.com/bensadeghi/DecisionTree.jl/blob/master/README.md](https://github.com/bensadeghi/DecisionTree.jl/blob/master/README.md). Predictions
are Deterministic.

Inputs are tables with ordinal columns. That is, the element scitype
of each column can be `Continuous`, `Count` or `OrderedFactor`.

For post-fit pruning, set `post-prune=true` and set
`pruning_purity_threshold` appropriately. Other hyperparameters as per
package documentation cited above.

"""
@mlj_model mutable struct DecisionTreeRegressor <: MLJBase.Deterministic
    pruning_purity_threshold::Float64 = 0.0::(0 ≤ _ ≤ 1)
    max_depth::Int					  = (-)(1)::(_ ≥ -1)
    min_samples_leaf::Int			  = 5::(_ ≥ 0)
    min_samples_split::Int			  = 2::(_ ≥ 2)
    min_purity_increase::Float64	  = 0.0::(_ ≥ 0)
    n_subfeatures::Int				  = 0::(_ ≥ 0)
    post_prune::Bool				  = false
end

function MLJBase.fit(model::DecisionTreeRegressor, verbosity::Int, X, y)
    Xmatrix   = MLJBase.matrix(X)
    fitresult = DecisionTree.build_tree(float.(y), Xmatrix
       , model.n_subfeatures
       , model.max_depth
       , model.min_samples_leaf
       , model.min_samples_split
       , model.min_purity_increase)

    if model.post_prune
        fitresult = DecisionTree.prune_tree(fitresult,
                                            model.pruning_purity_threshold)
    end
    cache = nothing
    report = nothing

    return fitresult, cache, report
end

MLJBase.fitted_params(::DecisionTreeRegressor, fitresult) =
    (tree_or_leaf = fitresult,)

function MLJBase.predict(model::DecisionTreeRegressor
                     , fitresult
                     , Xnew)
    Xmatrix = MLJBase.matrix(Xnew)
    return DecisionTree.apply_tree(fitresult,Xmatrix)
end

##
## METADATA
##

metadata_pkg.((DecisionTreeClassifier, DecisionTreeRegressor),
              name="DecisionTree",
              uuid="7806a523-6efd-50cb-b5f6-3fa6f1930dbb",
              url="https://github.com/bensadeghi/DecisionTree.jl",
              julia=true,
              license="MIT",
              is_wrapper=false)

metadata_model(DecisionTreeClassifier,
               input=MLJBase.Table(Continuous, Count, OrderedFactor),
               target=AbstractVector{<:MLJBase.Finite},
               weights=false,
               descr=DTC_DESCR)

metadata_model(DecisionTreeRegressor,
               input=MLJBase.Table(Continuous, Count, OrderedFactor),
               target=AbstractVector{MLJBase.Continuous},
               weights=false,
               descr=DTR_DESCR)

