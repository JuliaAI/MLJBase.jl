module TestDistributions

# using Revise
using Test
using MLJBase
using CategoricalArrays
import Distributions


## UNIVARIATE NOMINAL

v = collect("asdfghjklzxc")
d = UnivariateNominal(v, [0.09, 0.02, 0.1, 0.1, 0.1, 0.1, 0.1, 0.11, 0.01, 0.1, 0.07, 0.1])
@test pdf(d, 's') ≈ 0.02
@test mode(d) == 'k'
rand(d, 5)

v = collect("abcd")
d = UnivariateNominal(v, [0.2, 0.3, 0.1, 0.4])
sample = rand(d, 10^4)
freq_given_label = Distributions.countmap(sample)
pairs  = collect(freq_given_label)
sort!(pairs, by=pair->pair[2], alg=QuickSort)
sorted_labels = first.(pairs)
# if this fails it is bug or an exceedingly rare event or a bug:
@test sorted_labels == ['c', 'a', 'b', 'd']

v=['a', 'b', 'a', 'b', 'c', 'b', 'a', 'a']
d = Distributions.fit(UnivariateNominal, v) 
@test pdf(d, 'a') ≈ 0.5
@test pdf(d, 'b') ≈ 0.375 
@test pdf(d, 'c') ≈ 0.125

# to check fitting to categorical returns zero prob for missing
# levels, we add and drop new level to a categorical version of v:
w = categorical(append!(v, 'f'))
w = w[1:end-1]
e = Distributions.fit(UnivariateNominal, w) 
@test e.prob_given_label['f'] == 0

# arithmetic
d1 = UnivariateNominal(['a', 'b'], [0.2, 0.8])
d2 = UnivariateNominal(['b', 'c'], [0.3, 0.7])
d = average([d1, d2])
@test d.prob_given_label['a'] ≈ 0.1 && d.prob_given_label['b'] ≈ 0.55 &&  d.prob_given_label['c'] ≈ 0.35


end # module

true
