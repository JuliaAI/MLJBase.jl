@testset "categorical" begin
    x = 1:5
    @test MLJModelInterface.categorical(x) == categorical(x)
end
@testset "int, classes, decoder" begin
    N = 10
    mix = shuffle(0:N - 1)

    Xraw = broadcast(x->mod(x,N), rand(Int, 2N, 3N))
    Yraw = string.(Xraw)

    # to turn a categ matrix into a ordinary array with categorical
    # elements. Needed because broacasting the identity gives a
    # categorical array in CategoricalArrays >0.5.2
    function matrix_(X)
        ret = Array{Any}(undef, size(X))
        for i in eachindex(X)
            ret[i] = X[i]
        end
        return ret
    end

    X = categorical(Xraw)
    x = X[1]
    Y = categorical(Yraw)
    y = Y[1]
    V = matrix_(X)
    W = matrix_(Y)

    # raw(x::MLJBase.CategoricalValue) = x.pool.index[x.level]

    # @test raw.(classes(xo)) == xo.pool.levels
    # @test raw.(classes(yo)) == yo.pool.levels

    # # getting all possible elements from one:
    # @test raw.(X) == Xraw
    # @test raw.(Y) == Yraw
    # @test raw.(classes(xo)) == levels(Xo)
    # @test raw.(classes(yo)) == levels(Yo)

    # broadcasted encoding:
    @test int(X) == int(V)
    @test int(Y) == int(W)

    @test int(X; type=Int8) isa AbstractArray{Int8}

    # encoding is right-inverse to decoding:
    d = decoder(x)
    @test d(int(V)) == V # ie have the same elements
    e = decoder(y)
    @test e(int(W)) == W

    @test int(classes(y)) == 1:length(classes(x))

    # int is based on ordering not index
    v = categorical(['a', 'b', 'c'], ordered=true)
    @test int(v) == 1:3
    levels!(v, ['c', 'a', 'b'])
    @test int(v) == [2, 3, 1]

    # Errors
    @test_throws DomainError int("g")
end

@testset "matrix, table" begin
    B = rand(UInt8, (4, 5))
    @test matrix(DataFrame(B)) == B
    @test matrix(table(B)) == B
    @test matrix(table(B), transpose=true) == B'

    X  = (x1=rand(5), x2=rand(5))

    @test table(X, prototype=DataFrame()) == DataFrame(X)
    T = table((x1=(1,2,3), x2=(:x, :y, :z)))
    @test selectcols(T, :x1) == [1, 2, 3]

    v = categorical(11:20)
    A = hcat(v, v)
    tab = table(A)
    @test selectcols(tab, 1) == v

    @test matrix(B) == B
    @test matrix(B, transpose=true) == permutedims(B)
end

@testset "select etc" begin
    N = 10
    A = broadcast(x->Char(65+mod(x,5)), rand(Int, N, 5))
    X = CategoricalArrays.categorical(A)
    df = DataFrame(A)
    df.z  = 1:N
    tt = TypedTables.Table(df)
    rt = Tables.rowtable(tt)
    ct = Tables.columntable(tt)

    @test selectcols(nothing, 4:6)   == nothing
    @test selectcols(df, 4:6)        == df[:,4:6]
    @test selectcols(df, [:x1, :z])  == df[:,[:x1, :z]]
    @test selectcols(df, :x2)        == df.x2
    @test selectcols(nothing, 4:6)   == nothing
    @test selectcols(df, 2)          == df.x2
    @test selectrows(df, 4:6)        == selectrows(df[4:6, :], :)
    @test selectrows(df, 1)          == selectrows(df[1:1, :], :)
    @test selectrows(nothing, 4:6)   == nothing
    @test MLJBase.select(df, 2, :x2) == df[2, :x2]
    @test MLJBase.select(nothing, 2, :x) == nothing
    s = schema(df)
    @test nrows(df) == N

    @test selectcols(tt, 4:6) ==
        selectcols(TypedTables.Table(x4=tt.x4, x5=tt.x5, z=tt.z), :)
    @test selectcols(tt, [:x1, :z]) ==
        selectcols(TypedTables.Table(x1=tt.x1, z=tt.z), :)
    @test selectcols(tt, :x2) == tt.x2
    @test selectcols(tt, 2)   == tt.x2
    @test selectrows(tt, 4:6) == selectrows(tt[4:6], :)
    @test nrows(tt) == N
    @test MLJBase.select(tt, 2, :x2) == tt.x2[2]

    @test selectrows(rt, 4:6) == rt[4:6]
    @test selectrows(rt, :)   == rt

    @test selectrows(rt, 5)   == rt[5,:]
    @test nrows(rt) == N

    @test Tables.rowtable(selectrows(ct, 4:6)) == rt[4:6]
    @test selectrows(ct, :) == ct
    @test Tables.rowtable(selectrows(ct, 5))[1] == rt[5,1]

    # vector accessors
    v = rand(Int, 4)
    @test selectrows(v, 2:3) == v[2:3]
    @test selectrows(v, 2)   == [v[2]]
    @test nrows(v)           == 4

    v = categorical(collect("asdfasdf"))
    @test selectrows(v, 2:3) == v[2:3]
    @test selectrows(v, 2)   == [v[2]]
    @test nrows(v) == 8

    # matrix accessors
    A = rand(5, 10)
    @test selectrows(A, 2:4) == A[2:4,:]
    @test selectrows(A, 2:4) == A[2:4,:]
    @test selectrows(A, 2)   == A[2:2,:]

    A = rand(5, 10) |> categorical
    @test selectrows(A, 2:4) == A[2:4,:]
    @test selectrows(A, 2:4) == A[2:4,:]
    @test selectrows(A, 2)   == A[2:2,:]

    @test nrows(A) == 5

    # TypedTables
    v = categorical(collect("asdfasdf"))
    df = DataFrame(v=v, w=v)
    @test selectcols(df, :w) == v
    tt = TypedTables.Table(df)
    @test selectcols(tt, :w) == v
end
