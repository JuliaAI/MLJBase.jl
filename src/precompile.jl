using PrecompileTools

@compile_workload begin
    X = [1 2 3; 4 5 6]
    y = [1, 2]

    mutable struct A <: Probabilistic
        k::Float64
    end
    A(; k=0.0) = A(k)

    I = MLJBase.Distributions.I

    function MLJBase.fit(model::A, verbosity::Int, X, y)
        X = MLJBase.matrix(X)
        y = convert(Vector{Float64}, y)
        fitresult = (X'X + model.k * I) \ (X'y)
        cache = nothing
        report = nothing
        return fitresult, cache, report
    end

    function MLJBase.predict(model::A, fitresult, Xnew)
        MLJBase.matrix(Xnew) * fitresult
    end

    regressor = machine(A(), X, y; scitype_check_level=0)
end
