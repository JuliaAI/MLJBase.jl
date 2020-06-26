ms = map(measures()) do m
    m.name
end
@test "cross_entropy" in ms
@test "rms"  in ms

ms = map(measures(matching(categorical(1:3)))) do m
         m.name
end
@test "cross_entropy" in ms
@test !("rms"  in ms)

ms = map(measures(matching(rand(3)))) do m
         m.name
end
@test !("cross_entropy" in ms)
@test "rms"  in ms

true
