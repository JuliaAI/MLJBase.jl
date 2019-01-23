module TestData

# using Revise
using Test
using MLJBase
using DataFrames
using TypedTables

import CategoricalArrays

A = broadcast(x->Char(65+mod(x,5)), rand(Int, 10,5))
X = CategoricalArrays.categorical(A)
Xsmall = X[2:5,3:4]

decoder = MLJBase.CategoricalDecoder(X, eltype=Float16)
@test inverse_transform(decoder, transform(decoder, Xsmall)) == Xsmall

decoder = MLJBase.CategoricalDecoder(X)
@test inverse_transform(decoder, transform(decoder, Xsmall)) == Xsmall

@test MLJBase.matrix(DataFrame(A)) == A

df = DataFrame(A)
df.z  =1:10
@test retrieve(df, Cols, 4:6) == df[4:6]
@test retrieve(df, Cols, [:x1, :z]) == df[[:x1, :z]]
@test retrieve(df, Cols, :x2) == df.x2
@test retrieve(df, Cols, 2) == df.x2
@test retrieve(df, Rows, 4:6) == df[4:6, :]
s = retrieve(df, Schema)
@test (s.nrows, s.ncols) == size(df)
@test s.names == names(df)
@test s.eltypes == [Char, Char, Char, Char, Char, Int64]

tt = Table(df)
@test retrieve(tt, Cols, 4:6) == df[4:6]
@test retrieve(tt, Cols, [:x1, :z]) == df[[:x1, :z]]
@test retrieve(tt, Rows, 3:4) == df[3:4,:]
s = retrieve(tt, Schema)
@test (s.nrows, s.ncols) == size(df)
@test s.names == names(df)
@test s.eltypes == [Char, Char, Char, Char, Char, Int64]

A = rand(Int, 3, 4)
@test retrieve(A, Cols, 3:4) == A[:,3:4]
@test retrieve(A, Rows, 2:3) == A[2:3,:]
s = retrieve(A, Schema)
@test (s.nrows, s.ncols) == size(A)
@test s.names == [:x1, :x2, :x3, :x4]
@test s.eltypes == fill(Int, 4)

v = rand(Int, 4)
@test retrieve(v, Rows, 2:3) == v[2:3]
s = retrieve(v, Schema)
@test s.nrows == 4
@test s.ncols == 1
@test first(s.eltypes) == Int
@test s.names == [:x]

v = categorical(collect("asdfasdf"))
@test retrieve(v, Rows, 2:3) == v[2:3]
s = retrieve(v, Schema)
@test s.nrows == 8
@test s.ncols == 1
@test first(s.eltypes) == Char
@test s.names == [:x]
df = DataFrame(v=v, w=v)
@test retrieve(df, Cols, :w) == v
tt = TypedTables.Table(df)
@test retrieve(tt, Cols, :w) == v

end # module

true
