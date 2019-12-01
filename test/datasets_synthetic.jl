module TestDatasetsSynthetic

# using Revise
using Test
using MLJBase

# test make_bloobs
X, y = make_blobs(110; p=2, centers=3)
@test (110,2) == size(X)
@test 110 == length(y)

X, y = make_blobs(110; p=3, centers=4)
@test (110,3) == size(X)
@test 110 == length(y)

# test make_circles
X, y = make_circles(55)
@test (55,2) == size(X)
@test 55 == length(y)

# test make_moons
X, y = make_moons(75)
@test (75,2) == size(X)
@test 75 == length(y)


end # module
true
