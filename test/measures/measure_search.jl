ms = map(measures()) do m
    m.name
end
@test "LogLoss" in ms
@test "RootMeanSquaredError"  in ms

# test `M()` makes sense for all measure types `M` extracted from `name`,
@test all(Symbol.(ms)) do ex
    try
        eval(:($ex()))
        true
    catch
        false
    end
end

task(m) = AbstractVector{<:Finite} <: m.target_scitype

ms = map(measures(task)) do m
    m.name
end

@test "LogLoss" in ms
@test !("RootMeanSquaredError"  in ms)

task(m) = AbstractVector{Continuous} <: m.target_scitype

ms = map(measures(task)) do m
    m.name
end

@test !("LogLoss" in ms)
@test "RootMeanSquaredError"  in ms

ms = map(measures("Brier")) do  m
    m.name
end

@test Set(ms) == Set(["BrierLoss", "BrierScore",
                      "InfiniteBrierScore"])
true
