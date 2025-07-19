module TestLearningCompositesInspection

using Test
using MLJBase
using ..Models
import MLJModelInterface as MMI

"""
    children(N::AbstractNode, y::AbstractNode)

List all (immediate) children of node `N` in the ancestor graph of `y`
(training edges included).

"""
children(N::AbstractNode, y::AbstractNode) = filter(nodes(y)) do Z
    t = N in MLJBase.args(Z) ||
        N in MLJBase.train_args(Z)
end |> unique

@constant X = source()
@constant y = source()

hot = OneHotEncoder()
hotM = machine(hot, X)
@constant W = transform(hotM, X)
knn = KNNRegressor()
knnM = machine(knn, W, y)
@constant yhat = predict(knnM, W)
@constant K =  2*X
@constant all = glb(yhat, K)

@test MLJBase.tree(yhat) == (operation = predict,
                             model = knn,
                             arg1 = (operation = transform,
                                     model = hot,
                                     arg1 = (source = X, ),
                                     train_arg1 = (source = X, )),
                             train_arg1 = (operation = transform,
                                           model = hot,
                                           arg1 = (source = X, ),
                                           train_arg1 = (source = X, )),
                             train_arg2 = (source = y,))

@test Set(MLJBase.models(yhat)) == Set([hot, knn])
@test Set(sources(yhat)) == Set([X, y])
@test Set(origins(yhat)) == Set([X,])
@test Set(machines(yhat)) == Set([knnM, hotM])
@test Set(MLJBase.args(yhat)) == Set([W, ])
@test Set(MLJBase.train_args(yhat)) == Set([W, y])
@test Set(children(X, all)) == Set([W, K])

@constant Q = 2X
@constant R = 3X
@constant S = glb(X, Q, R)
@test Set(children(X, S)) == Set([Q, R, S])
@test MLJBase.lower_bound([Int, Float64]) == Union{}
@test MLJBase.lower_bound([Int, Integer]) == Int
@test MLJBase.lower_bound([Int, Integer]) == Int
@test MLJBase.lower_bound([]) == Any

@test input_scitype(2X) == Unknown
@test input_scitype(yhat) == input_scitype(KNNRegressor())
W2 = transform(machine(UnivariateStandardizer(), X), X)
# @test input_scitype(X, glb(W, W2)) == Union{}
# @test input_scitype(X, glb(Q, W)) == Unknown

y1 = predict(machine(DecisionTreeRegressor(), X, y), X)
@test input_scitype(y1) == Table(Continuous, OrderedFactor, Count)
y2 = predict(machine(KNNRegressor(), X, y), X)
@test input_scitype(y2) == Table(Continuous)
# @test input_scitype(X, glb(y1, y2)) == Table(Continuous)
# @test input_scitype(X, glb(y1, y2, Q)) == Unknown

end

true
