module TestUtilities

# using Revise
using Test
using MLJBase
import Random

train, test = partition(1:100, 0.9)
@test collect(train) == collect(1:90)
@test collect(test) == collect(91:100)
train, test = partition(1:100, 0.9, shuffle=true)
@test length(train) == 90

train, test = partition(1:100, 0.9, shuffle=true, rng=1)
@test length(train) == 90

train, test = partition(1:100, 0.9, shuffle=true, rng=Random.MersenneTwister(3))
@test length(train) == 90

end # module
true

