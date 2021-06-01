module TestStacking

using Test
using MLJBase
using MLJModelInterface
using ..Models
using Random

import Distributions


rng = Random.seed!(1234)


function model_evaluation(models::NamedTuple, X, y; measure=rmse)
    results = []
    for model in models
        mach = machine(model, X, y)
        ev = evaluate!(mach; resampling=CV(;nfolds=3), measure=measure, check_measure=false)
        push!(results, ev.measurement[1])
    end
    results
end


@testset "Testing Stack on Continuous target" begin
    X, y = make_regression(500, 5)

    @testset "Testing Deterministic Stack" begin
    # Testing performance

    # The dataset is a simple regression model with intercept
    # No model in the stack can recover the true model on its own 
    # Indeed, FooBarRegressor has no intercept 
    # By combining models, the stack can generalize better than any submodel

    models = (constant=DeterministicConstantRegressor(),
                decisiontree=DecisionTreeRegressor(), 
                ridge_lambda=FooBarRegressor(;lambda=0.1), 
                ridge=FooBarRegressor(;lambda=0))

    mystack = stack(FooBarRegressor();
        cv_strategy=CV(;nfolds=3),
        models...)
    
    results = model_evaluation((stack=mystack, models...), X, y)
    @test argmin(results) == 1

    # Mixing ProbabilisticModels amd Deterministic models as members of the stack
    models = (constant=ConstantRegressor(),
                decisiontree=DecisionTreeRegressor(), 
                ridge_lambda=FooBarRegressor(;lambda=0.1), 
                ridge=FooBarRegressor(;lambda=0))

    mystack = stack(FooBarRegressor();
                    cv_strategy=CV(;nfolds=3),
                    models...)
    # Testing attribute access of the stack
    @test propertynames(mystack) == (:cv_strategy, :metalearner, :models, :constant, 
                                    :decisiontree, :ridge_lambda, :ridge)

    @test mystack.decisiontree isa DecisionTreeRegressor

    @test target_scitype(mystack) == target_scitype(FooBarRegressor())

    # Testing fitted_params results are easily accessible for each
    # submodel. They are in order of the cross validation procedure.
    # Here 3-folds then 3 machines + the final fit
    mach = machine(mystack, X, y)
    fit!(mach)
    fp = fitted_params(mach)
    @test nrows(getfield(fp, :constant)) == 4
    @test nrows(getfield(fp, :decisiontree)) == 4
    @test nrows(getfield(fp, :ridge)) == 4
    @test nrows(getfield(fp, :ridge_lambda)) == 4

    # Testing prediction is Deterministic
    @test predict(mach) isa Vector{Float64}
    end

    @testset "Testing ProbabilisticStack" begin
        models = (constant=ConstantRegressor(),
                    decisiontree=DecisionTreeRegressor(), 
                    ridge_lambda=FooBarRegressor(;lambda=0.1), 
                    ridge=FooBarRegressor(;lambda=0))

        # The type of the stack is determined by the type of the metalearner
        metalearner = ConstantRegressor(;distribution_type=Distributions.Cauchy)
        mystack = stack(metalearner;
                    cv_strategy=CV(;nfolds=3),
                    models...)

        @test target_scitype(mystack) == target_scitype(metalearner)

        mach = machine(mystack, X, y)
        fit!(mach)
        @test predict(mach) isa Vector{Distributions.Cauchy{Float64}}

    end

end

@testset "Testing ProbabilisticStack on Finite target" begin
    X, y = make_blobs()

    models = (constant=ConstantClassifier(),
                decisiontree=DecisionTreeClassifier(), 
                knn=KNNClassifier())

    mystack = stack(DecisionTreeClassifier();
                    cv_strategy=CV(;nfolds=3),
                    models...)
    
    @test target_scitype(mystack) == target_scitype(DecisionTreeClassifier())
    mach = machine(mystack, X, y)
    fit!(mach)
    @test predict(mach) isa Vector{<:MLJBase.UnivariateFinite}

end

@testset "Stack constructor valid argument checks" begin
    # metalearner should have target_scitype:
    # Union{AbstractArray{<:Continuous}, AbstractArray{<:Finite}}
    @test_throws ArgumentError stack(Standardizer(); 
                        constant=ConstantClassifier())

    # models should have the same target scitype as the metalearner
    @test_throws ArgumentError stack(ConstantClassifier(); 
                        constant=KNNRegressor())
end

end

true