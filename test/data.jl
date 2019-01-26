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

@test retrieve(df, Cols, 4:6) == retrieve(df[4:6], Cols, :)
@test retrieve(df, Cols, [:x1, :z]) == retrieve(df[[:x1, :z]], Cols, :)
@test retrieve(df, Cols, :x2) == df.x2
@test retrieve(df, Cols, 2) == df.x2
@test retrieve(df, Rows, 4:6) == retrieve(df[4:6, :], Rows, :)
s = retrieve(df, Schema)
@test (s.nrows, s.ncols) == size(df)
@test s.names == tuple(names(df)...)
@test s.eltypes == (Char, Char, Char, Char, Char, Int64)

tt = Table(df)
@test retrieve(tt, Cols, 4:6) == retrieve(Table(x4=tt.x4, x5=tt.x5, z=tt.z), Cols, :)
@test retrieve(tt, Cols, [:x1, :z]) == retrieve(Table(x1=tt.x1, z=tt.z), Cols, :)
@test retrieve(tt, Cols, :x2) == tt.x2
@test retrieve(tt, Cols, 2) == tt.x2
@test retrieve(tt, Rows, 4:6) == retrieve(tt[4:6], Rows, :)
s = retrieve(tt, Schema)
@test (s.nrows, s.ncols) == size(df)
@test s.names == tuple(names(df)...)
@test s.eltypes == (Char, Char, Char, Char, Char, Int64)


A = rand(Int, 3, 4)
@test retrieve(A, Cols, 3:4) == A[:,3:4]
@test retrieve(A, Rows, 2:3) == A[2:3,:]
s = retrieve(A, Schema)
@test (s.nrows, s.ncols) == size(A)
@test s.names == (:x1, :x2, :x3, :x4)
@test s.eltypes == (Int, Int, Int, Int)

v = rand(Int, 4)
@test retrieve(v, Rows, 2:3) == v[2:3]
s = retrieve(v, Schema)
@test s.nrows == 4
@test s.ncols == 1
@test first(s.eltypes) == Int
@test s.names == (:x,)

v = categorical(collect("asdfasdf"))
@test retrieve(v, Rows, 2:3) == v[2:3]
s = retrieve(v, Schema)
@test s.nrows == 8
@test s.ncols == 1
@test first(s.eltypes) == Char
@test s.names == (:x,)
df = DataFrame(v=v, w=v)
@test retrieve(df, Cols, :w) == v
tt = TypedTables.Table(df)
@test retrieve(tt, Cols, :w) == v

end # module

true
