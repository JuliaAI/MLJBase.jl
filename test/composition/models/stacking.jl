module TestStacking

using Test
using MLJBase
include("test/_models/models.jl")
using ..Models


rng = Random.seed!(1234)

function model_evaluation(models::NamedTuple; measure=rmse)
    results = []
    for model in models
        mach = machine(model, X, y)
        ev = evaluate!(mach;resampling=CV(;nfolds=3), measure=measure, check_measure=false)
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

    X, y = make_regression(500, 5)

    mach = machine(stack, X, y)

    fit!(mach)
    # The dataset is a simple regression model with intercept
    # No model in the stack can recover the true model on its own 
    # Indeed, FooBarRegressor has no intercept 
    # By combining models, the stack can generalize better than any submodel
    results = model_evaluation((stack=stack, models...))
    @test argmin(results) == 1
    
end


@testset "Stack on Binary target" begin
    models = (constant=DeterministicConstantClassifier(),
                decisiontree=DecisionTreeClassifier(), 
                # foobar_lambda=FooBarRegressor(), 
                foobar=KNNClassifier())

    stack = Stack(DecisionTreeClassifier();
                    cv_strategy=CV(;nfolds=3),
                    models...)

    X, y = make_circles(500)

    # Check the Stack is performing better than any of its submodels
    results = model_evaluation((stack=stack, models...), measure=log_loss)
    @test argmin(results) == 1
    
end

end