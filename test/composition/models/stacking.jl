module TestStacking

using Test
using MLJBase
using MLJModelInterface
using ..Models
using Random
using StableRNGs

import Distributions

rng = StableRNGs.StableRNG(1234)


function model_evaluation(models::NamedTuple, X, y; measure=rmse)
    cv = CV(;nfolds=3)
    results = []
    for model in models
        mach = machine(model, X, y)
        ev = evaluate!(mach; resampling=cv, verbosity=0, measure=measure, check_measure=false)
        push!(results, ev.measurement[1])
    end
    results
end


@testset "Testing Stack on Continuous target" begin
    X, y = make_regression(500, 5; rng=rng)

    @testset "Testing Deterministic Stack" begin
    # Testing performance

    # The dataset is a simple regression model with intercept
    # No model in the stack can recover the true model on its own 
    # Indeed, FooBarRegressor has no intercept 
    # By combining models, the stack can generalize better than any submodel
    # And optimize the rmse

    models = (constant=DeterministicConstantRegressor(),
                decisiontree=DecisionTreeRegressor(), 
                ridge_lambda=FooBarRegressor(;lambda=0.1), 
                ridge=FooBarRegressor(;lambda=0))

    mystack = Stack(;metalearner=FooBarRegressor(),
                    resampling=CV(;nfolds=3),
                    models...)
    
    results = model_evaluation((stack=mystack, models...), X, y)
    @test argmin(results) == 1

    # Mixing ProbabilisticModels amd Deterministic models as members of the stack
    models = (constant=ConstantRegressor(),
                decisiontree=DecisionTreeRegressor(), 
                ridge_lambda=FooBarRegressor(;lambda=0.1), 
                ridge=FooBarRegressor(;lambda=0))

    mystack = Stack(;metalearner=FooBarRegressor(),
                    resampling=CV(;nfolds=3),
                    models...)
    # Testing attribute access of the stack
    @test propertynames(mystack) == (:resampling, :metalearner, :constant, 
                                    :decisiontree, :ridge_lambda, :ridge)

    @test mystack.decisiontree isa DecisionTreeRegressor
    
    @test target_scitype(mystack) == target_scitype(FooBarRegressor())
    @test input_scitype(mystack) == input_scitype(FooBarRegressor())

    # Testing fitted_params results are easily accessible for each
    # submodel. They are in order of the cross validation procedure.
    # Here 3-folds then 3 machines + the final fit
    mach = machine(mystack, X, y)
    fit!(mach, verbosity=0)
    fp = fitted_params(mach)
    @test nrows(getfield(fp, :constant)) == 4
    @test nrows(getfield(fp, :decisiontree)) == 4
    @test nrows(getfield(fp, :ridge)) == 4
    @test nrows(getfield(fp, :ridge_lambda)) == 4

    # The metalearner has been fit and has one coefficient 
    # for each model in the library (No intercept)
    @test fp.metalearner isa Vector{Float64}
    @test nrows(fp.metalearner) == 4

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
        mystack = Stack(;metalearner=metalearner,
                    resampling=CV(;nfolds=3),
                    models...)

        @test target_scitype(mystack) == target_scitype(metalearner)
        @test input_scitype(mystack) == input_scitype(FooBarRegressor())

        mach = machine(mystack, X, y)
        fit!(mach, verbosity=0)
        @test predict(mach) isa Vector{Distributions.Cauchy{Float64}}

    end

end

@testset "Testing ProbabilisticStack on Finite target" begin
    X, y = make_blobs(;rng=rng, shuffle=false)

    models = (constant=ConstantClassifier(),
                decisiontree=DecisionTreeClassifier(), 
                knn=KNNClassifier())

    mystack = Stack(;metalearner=DecisionTreeClassifier(),
                    resampling=StratifiedCV(;nfolds=3),
                    models...)
    
    
    # Check input and target scitypes
    @test target_scitype(mystack) == target_scitype(DecisionTreeClassifier())
    # Here the greatest lower bound is the scitype of the knn
    @test input_scitype(mystack) == input_scitype(KNNClassifier())

    mach = machine(mystack, X, y)
    fit!(mach, verbosity=0)
    @test predict(mach) isa Vector{<:MLJBase.UnivariateFinite}

end

@testset "Stack constructor valid argument checks" begin
    # metalearner should have target_scitype:
    # Union{AbstractArray{<:Continuous}, AbstractArray{<:Finite}}
    @test_throws ArgumentError Stack(;metalearner=Standardizer(),
                        constant=ConstantClassifier())

    @test_throws ArgumentError Stack(;constant=KNNRegressor())
end


@testset "Misc" begin 
    # Test setproperty! behaviour
    models = (constant=DeterministicConstantRegressor(),
                decisiontree=DecisionTreeRegressor(), 
                ridge_lambda=FooBarRegressor(;lambda=0.1))

    mystack = Stack(;metalearner=FooBarRegressor(),
                    resampling=CV(;nfolds=3),
                    models...)
    @test mystack.ridge_lambda.lambda == 0.1
    @test mystack.metalearner isa FooBarRegressor
    @test mystack.resampling isa CV
    
    mystack.ridge_lambda = FooBarRegressor(;lambda=0.2)
    @test mystack.ridge_lambda.lambda == 0.2

    mystack.metalearner = DecisionTreeRegressor()
    @test mystack.metalearner isa DecisionTreeRegressor

    mystack.resampling = StratifiedCV()
    @test mystack.resampling isa StratifiedCV

    # using inner constructor accepts :resampling and :metalearner
    # as modelnames
    modelnames = (:resampling, :metalearner)
    models = [DeterministicConstantRegressor(), FooBarRegressor(;lambda=0)]
    metalearner = DeterministicConstantRegressor()
    resampling = CV()

    MLJBase.DeterministicStack(modelnames, models, metalearner, resampling)

    # Test input_target_scitypes with non matching target_scitypes
    models = [KNNRegressor()]
    metalearner = KNNClassifier()
    inp_scitype, tg_scitype = MLJBase.input_target_scitypes(models, metalearner)
    @test tg_scitype == Unknown
    @test inp_scitype == Table{<:AbstractVector{<:Continuous}}

    # Test input_target_scitypes with non matching target_scitypes
    models = [ConstantClassifier(), DecisionTreeClassifier()]
    metalearner = KNNClassifier()
    inp_scitype, tg_scitype = MLJBase.input_target_scitypes(models, metalearner)
    @test tg_scitype == AbstractVector{<:Finite}
    @test inp_scitype == Table{<:Union{AbstractVector{<:Continuous}, 
                                       AbstractVector{<:Count},
                                       AbstractVector{<:OrderedFactor}}}

end

@testset  "function oos_set" begin
    X = (x = Float64[1, 1, 2, 2, 3, 3],)
    y = coerce(['a', 'b', 'b', 'c', 'a', 'a'], Multiclass)
    n = nrows(y)

    model1 = KNNClassifier(K=2)
    model2 = ConstantClassifier()
    judge = KNNClassifier(K=3)

    stack = Stack(metalearner=judge,
                model1=model1,
                model2=model2,
                resampling=CV(;nfolds=3, shuffle=true, rng=rng))

    Xs = source(X)
    ys = source(y)
    folds = MLJBase.getfolds(ys, stack.resampling, n)

    Zval, yval = MLJBase.oos_set(stack, folds, Xs, ys)
    # To be accessed, the machines need to be trained
    fit!(Zval, verbosity=0)
    # Each model in the library should output a 3-dim vector to be concatenated
    # resulting in the table of shape (nrows, 6) here nrows=6 
    # for future use by the metalearner
    sc = schema(Zval())
    @test sc.nrows == 6
    @test sc.names == (:x1, :x2, :x3, :x4, :x5, :x6)

    # The lines of yval should match the reordering indexes
    # of the original y (reordering given by the folds node)
    reordering = vcat([x[2] for x in folds()]...)
    @test yval() == y[reordering]
    # And the same is true for Zval, let's check this for model1's output
    # on the first fold, ie (2 first rows, 3 first columns)
    # First we need to train the model
    trainingrows = folds()[1][1]
    Xtrain = selectrows(X, trainingrows)
    ytrain = selectrows(y, trainingrows)
    mach = machine(model1, Xtrain, ytrain)
    fit!(mach, verbosity=0)

    # Then predict on the validation rows
    Xpred = selectrows(X, folds()[1][2])
    Zval_expected_dist = predict(mach, Xpred)
    # This is a distribution, we need to apply the appropriate transformation
    Zval_expected = pdf(Zval_expected_dist, levels(first(Zval_expected_dist)))
    @test matrix(Zval())[1:2, 1:3] == Zval_expected
    
end

@testset "An integration test for stacked classification" begin

    # We train a stack by hand and compare with the canned version
    # `Stack(...)`. There are two base learners, with 3-fold
    # cross-validation used to construct the out-of-sample base model
    # predictions. 

    probs(y) = pdf(y, levels(first(y)))

    # data:
    N = 2000
    X = (x = rand(rng, 3N), )
    y = coerce(rand("abc", 3N), Multiclass)

    # row splits:
    test1 = 1:N
    test2 = (N + 1):2N
    test3 = (2N + 1):3N
    train1 = (N + 1):3N
    train2 = vcat(1:N, (2N + 1):3N)
    train3 = 1:2N

    # base `model1`:
    model1 = KNNClassifier(K=2)
    mach1 = machine(model1, X, y)
    fit!(mach1, rows=train1, verbosity=0)
    y11 = predict(mach1, rows=test1) |> probs
    mach1 = machine(model1, X, y)
    fit!(mach1, rows=train2, verbosity=0)
    y12 = predict(mach1, rows=test2) |> probs
    mach1 = machine(model1, X, y)
    fit!(mach1, rows=train3, verbosity=0)
    y13 = predict(mach1, rows=test3) |> probs
    y1_oos = vcat(y11, y12, y13)
    mach1_full = machine(model1, X, y)
    fit!(mach1_full, verbosity=0)
    y1 = predict(mach1_full, X) |> probs

    # base `model2`:
    model2 = DecisionTreeClassifier()
    mach2 = machine(model2, X, y)
    fit!(mach2, rows=train1, verbosity=0)
    y21 = predict(mach2, rows=test1) |> probs
    mach2 = machine(model2, X, y)
    fit!(mach2, rows=train2, verbosity=0)
    y22 = predict(mach2, rows=test2) |> probs
    mach2 = machine(model2, X, y)
    fit!(mach2, rows=train3, verbosity=0)
    y23 = predict(mach2, rows=test3) |> probs
    y2_oos = vcat(y21, y22, y23)
    mach2_full = machine(model2, X, y)
    fit!(mach2_full, verbosity=0)
    y2 = predict(mach2_full, X) |> probs

    # metalearner (`judge`):
    X_oos = MLJBase.table(hcat(y1_oos, y2_oos))
    judge = KNNClassifier(K=3)
    m_judge = machine(judge, X_oos, y)
    fit!(m_judge, verbosity=0)
    X_judge = MLJBase.table(hcat(y1, y2))
    yhat_matrix = predict(m_judge, X_judge) |> probs

    # alternatively, use stack:
    stack = Stack(metalearner=judge,
                model1=model1,
                model2=model2,
                resampling=CV(nfolds=3))
    mach = machine(stack, X, y)
    fit!(mach, verbosity=0)
    yhat_matrix_stack = predict(mach, X) |> probs

    # compare:
    @test yhat_matrix_stack ≈ yhat_matrix
    
end

end

true