module TestDistributions

# using Revise
using Test
using MLJBase
using CategoricalArrays
import Distributions:pdf, support
import Distributions
import Random.seed!
seed!(1234)


## UNIVARIATE NOMINAL

v = categorical(collect("asqfasqffqsaaaa"), ordered=true)
a, s, q, f = v[1], v[2], v[3], v[4]
p = a.pool
@test MLJBase.class(a.level, p) == a
levels!(v, reverse(levels(v)))
@test MLJBase.class(a.level, p) == a
levels!(v, reverse(levels(v)))

dic=Dict(s=>0.1, q=> 0.2, f=> 0.7)
d = UnivariateFinite(dic)
@test classes(d) == [a, f, q, s]
@test support(d) == [f, q, s]
levels!(v, reverse(levels(v)))
@test classes(d) == [s, q, f, a]
@test support(d) == [s, q, f]

@test pdf(d, s) ≈ 0.1
@test mode(d) == f
@test rand(d, 5) == [f, q, f, f, q]

y = categorical(["yes", "no", "yes", "yes", "maybe"])
yes = y[1]
no = y[2]
maybe = y[end]
prob_given_class = Dict(yes=>0.7, no=>0.3)
d =UnivariateFinite(prob_given_class)
@test pdf(d, yes) ≈ 0.7
@test pdf(d, no) ≈ 0.3
@test pdf(d, maybe) ≈ 0

v = categorical(collect("abcd"))
d = UnivariateFinite(v, [0.2, 0.3, 0.1, 0.4])
sample = rand(d, 10^4)
freq_given_class = Distributions.countmap(sample)
pairs  = collect(freq_given_class)
sort!(pairs, by=pair->pair[2], alg=QuickSort)
sorted_classes = first.(pairs)
@test sorted_classes == ['c', 'a', 'b', 'd']

junk = categorical(['j',])
j = junk[1]
v = categorical(['a', 'b', 'a', 'b', 'c', 'b', 'a', 'a', 'f'])
a = v[1]
f = v[end]
# remove f from sample:
v = v[1 : end - 1]
d = Distributions.fit(UnivariateFinite, v) 
@test pdf(d, a) ≈ 0.5
@test pdf(d, 'a') ≈ 0.5
@test pdf(d, 'b') ≈ 0.375 
@test pdf(d, 'c') ≈ 0.125
@test pdf(d, 'f') == 0
@test pdf(d, f) == 0
@test_throws ArgumentError pdf(d, 'j')
@test_throws ArgumentError pdf(d, j)

# arithmetic
v = categorical(collect("abc"))
a, b, c = v[1], v[2], v[3]
d1 = UnivariateFinite([a, b], [0.2, 0.8])
d2 = UnivariateFinite([b, c], [0.3, 0.7])
dvec = [d1, d2]
d = average(dvec)
@test pdf(d, 'a') ≈ 0.1
@test pdf(d, 'b') ≈ 0.55
@test pdf(d, 'c') ≈ 0.35
w = [4, 6]
d = average(dvec, weights=w)
@test pdf(d, 'a') ≈ 0.4*0.2
@test pdf(d, 'b') ≈ 0.4*0.8 + 0.6*0.3
@test pdf(d, 'c') ≈ 0.6*0.7


end # module

true
