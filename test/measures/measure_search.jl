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

ms = map(measures(matching(categorical(1:3)))) do m
         m.name
end
@test "LogLoss" in ms
@test !("RootMeanSquaredError"  in ms)

ms = map(measures(matching(rand(3)))) do m
         m.name
end
@test !("LogLoss" in ms)
@test "RootMeanSquaredError"  in ms

true
