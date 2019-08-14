module TestDistributions

# using Revise
using Test
using MLJBase
using CategoricalArrays
import Distributions
import Random.seed!
seed!(1234)


## UNIVARIATE NOMINAL

v = collect("asdfghjklzxc")
d = UnivariateFinite(v, [0.09, 0.02, 0.1, 0.1,
                          0.1, 0.1, 0.1, 0.11,
                          0.01, 0.1, 0.07, 0.1])
@test pdf(d, 's') ≈ 0.02
@test mode(d) == 'k'
@test rand(d, 5) == ['a', 'z', 'a', 'k', 'z']
@test Set(levels(d)) == Set(v)

v = collect("abcd")
d = UnivariateFinite(v, [0.2, 0.3, 0.1, 0.4])
sample = rand(d, 10^4)
freq_given_level = Distributions.countmap(sample)
pairs  = collect(freq_given_level)
sort!(pairs, by=pair->pair[2], alg=QuickSort)
sorted_levels = first.(pairs)
@test sorted_levels == ['c', 'a', 'b', 'd']

# test unseen values in pool get zero probability:
v = levels!(categorical(collect("abcd")), collect("abcdf"))
d = UnivariateFinite(v, [0.2, 0.3, 0.1, 0.4])
@test d.prob_given_level['f'] == 0
vp = broadcast(identity, v)
d = UnivariateFinite(vp, [0.2, 0.3, 0.1, 0.4])
@test d.prob_given_level['f'] == 0
vpp = levels!(categorical(["x", "y"]), ["x", "y", "z"])
d = UnivariateFinite(vpp, [0.2, 0.8])
@test d.prob_given_level["z"] == 0

v = categorical(['a', 'b', 'a', 'b', 'c', 'b', 'a', 'a'])
d = Distributions.fit(UnivariateFinite, v) 
@test pdf(d, 'a') ≈ 0.5
@test pdf(d, 'b') ≈ 0.375 
@test pdf(d, 'c') ≈ 0.125

# to check fitting to categorical returns zero prob for missing
# levels, we add and drop new level to a categorical version of v:
w = levels!(v, ['a', 'b', 'c', 'f'])
e = Distributions.fit(UnivariateFinite, w) 
@test e.prob_given_level['f'] == 0

# arithmetic
d1 = UnivariateFinite(['a', 'b'], [0.2, 0.8])
d2 = UnivariateFinite(['b', 'c'], [0.3, 0.7])
d = average([d1, d2])
@test d.prob_given_level['a'] ≈ 0.1 && d.prob_given_level['b'] ≈ 0.55 &&  d.prob_given_level['c'] ≈ 0.35


end # module

true
