module TestStacking

using Test
using MLJBase
using ..Models
using Random


rng = Random.seed!(1234)


function model_evaluation(models::NamedTuple, X, y; measure=rmse)
    results = []
    for model in models
        mach = machine(model, X, y)
        ev = evaluate!(mach; resampling=CV(;nfolds=3), measure=measure)
        push!(results, ev.measurement[1])
    end
    results
end


@testset "Stack on regression Continuous target" begin
    models = (constant=DeterministicConstantRegressor(),
                decisiontree=DecisionTreeRegressor(), 
                ridge_lambda=FooBarRegressor(;lambda=0.1), 
                ridge=FooBarRegressor(;lambda=0))

    stack = Stack(FooBarRegressor();
                    cv_strategy=CV(;nfolds=3),
                    models...)

    # Testing attribute access of the stack
    @test propertynames(stack) == (:cv_strategy, :metalearner, :models, :constant, 
                                    :decisiontree, :ridge_lambda, :ridge)

    @test stack.decisiontree isa DecisionTreeRegressor

    # Testing performance

    # The dataset is a simple regression model with intercept
    # No model in the stack can recover the true model on its own 
    # Indeed, FooBarRegressor has no intercept 
    # By combining models, the stack can generalize better than any submodel
    X, y = make_regression(500, 5)
    mach = machine(stack, X, y)
    results = model_evaluation((stack=stack, models...), X, y)
    @test argmin(results) == 1

    # Testing fitted_params results are easily accessible for each
    # submodel. They are in order of the cross validation procedure.
    # Here 3-folds then 3 machines + the final fit
    fit!(mach)
    fp = fitted_params(mach)
    @test nrows(getfield(fp, :constant)) == 4
    @test nrows(getfield(fp, :decisiontree)) == 4
    @test nrows(getfield(fp, :ridge)) == 4
    @test nrows(getfield(fp, :ridge_lambda)) == 4


end


# @testset "Stack on Binary target" begin
#     models = (constant=DeterministicConstantClassifier(),
#                 decisiontree=DecisionTreeClassifier(), 
#                 # foobar_lambda=FooBarRegressor(), 
#                 foobar=KNNClassifier())

#     stack = Stack(DecisionTreeClassifier();
#                     cv_strategy=CV(;nfolds=3),
#                     models...)

#     X, y = make_circles(500)

#     model = DecisionTreeClassifier()
#     mach = machine(model, X, y)
#     fit!(mach)
#     ypred = predict(mach)
#     # Check the Stack is performing better than any of its submodels
#     results = model_evaluation((stack=stack, models...), measure=log_loss)
#     @test argmin(results) == 1
    
# end

end