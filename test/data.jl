module TestData

# using Revise
using Test
using MLJInterface
using DataFrames

import CategoricalArrays

A = broadcast(x->Char(65+mod(x,5)), rand(Int, 10,5))
X = CategoricalArrays.categorical(A)
Xsmall = X[2:5,3:4]

decoder = MLJInterface.CategoricalDecoder(X, eltype=Float16)
@test inverse_transform(decoder, transform(decoder, Xsmall)) == Xsmall

decoder = MLJInterface.CategoricalDecoder(X)
@test inverse_transform(decoder, transform(decoder, Xsmall)) == Xsmall

@test MLJInterface.matrix(DataFrame(A)) == A

end # module

true
