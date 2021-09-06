module TestData

using Test
#using DataFrames
import TypedTables
using CategoricalArrays
import Tables
using ScientificTypes

using Random
using StableRNGs

rng = StableRNG(55511)

import MLJBase
import MLJBase: decoder, int, classes, partition, unpack, selectcols, matrix,
    CategoricalValue, selectrows, select, table, nrows, restrict,
    corestrict, complement, transform

@testset "partition" begin
    train, test = partition(1:100, 0.9)
    @test collect(train) == collect(1:90)
    @test collect(test) == collect(91:100)
    rng = StableRNG(666)
    train, test = partition(1:100, 0.9, shuffle=true, rng=rng)
    @test length(train) == 90
    @test length(test) == 10
    @test train[1:8] == [49, 75, 98, 99, 47, 59, 65, 12]
    rng = StableRNG(888)
    train, test = partition(1:100, 0.9, rng=rng)
    rng = StableRNG(888)
    train2, test2 = partition(1:100, 0.9, shuffle=true, rng=rng)
    @test train2 == train
    @test test2 == test

    train, test = partition(1:100, 0.9, shuffle=false, rng=1)
    @test collect(train) == collect(1:90)
    @test collect(test) == collect(91:100)

    # Matrix
    X = collect(reshape(1:10, 5, 2))
    @test partition(X, 0.2, 0.4) == ([1 6], [2 7; 3 8], [4 9; 5 10])
    rng = StableRNG(42)
    @test partition(X, 0.2, 0.4; shuffle=true, rng=rng) == ([5 10], [3 8; 4 9], [1 6; 2 7])

    # Table
    rows = Tables.rows((a=collect(1:5), b=collect(6:10)))
    @test partition(rows, 0.6, 0.2) ==
        ((a = [1, 2, 3], b = [6, 7, 8]), (a = [4], b = [9]), (a = [5], b = [10]))
    rng = StableRNG(123)
    @test partition(rows, 0.6, 0.2; shuffle=true, rng=rng) ==
        ((a = [3, 1, 5], b = [8, 6, 10]), (a = [2], b = [7]), (a = [4], b = [9]))

    # Not a vector/matrix/table
    @test_throws ArgumentError partition(1)

    # with stratification
    y = ones(Int, 1000)
    y[end-100:end] .= 0; # 90%

    train1, test1 =
        partition(eachindex(y), 0.8, stratify=categorical(y), rng=34)
    train, test = partition(eachindex(y), 0.8, stratify=y, rng=34)
    @test train == train1
    @test test == test1
    @test isapprox(sum(y[train])/length(train), 0.9, rtol=1e-2)
    @test isapprox(sum(y[test])/length(test), 0.9, rtol=1e-2)

    s1, s2, s3 = partition(eachindex(y), 0.3, 0.6, stratify=y, rng=345)
    @test isapprox(sum(y[s1])/length(s1), 0.9, rtol=1e-2)
    @test isapprox(sum(y[s2])/length(s2), 0.9, rtol=1e-2)
    @test isapprox(sum(y[s3])/length(s3), 0.9, rtol=1e-2)

    y = ones(Int, 1000)
    y[end-500:end-200] .= 2
    y[end-200+1:end] .= 3
    p1 = sum(y .== 1) / length(y) # 0.5
    p2 = sum(y .== 2) / length(y) # 0.3
    p3 = sum(y .== 3) / length(y) # 0.2

    s1, s2, s3 = partition(eachindex(y), 0.3, 0.6, stratify=y, rng=111)
    # overkill test...
    for s in (s1, s2, s3)
        for (i, p) in enumerate((p1, p2, p3))
            @test isapprox(sum(y[s] .== i)/length(s), p, rtol=1e-2)
        end
    end

    # it should work with missing values though maybe not recommended...
    y = ones(Union{Missing,Int}, 1000)
    y[end-600:end-550] .= missing
    y[end-500:end-200] .= 2
    y[end-200+1:end] .= 3
    p1 = sum(skipmissing(y) .== 1) / length(y) # 0.45
    p2 = sum(skipmissing(y) .== 2) / length(y) # 0.3
    p3 = sum(skipmissing(y) .== 3) / length(y) # 0.2
    pm = sum(ismissing.(y)) / length(y)        # 0.05

    s1, s2 = partition(eachindex(y), 0.7, stratify=y, rng=11)
    for s in (s1, s2)
        for (i, p) in enumerate((p1, p2, p3))
            @test isapprox(sum(y[s] .=== i)/length(s), p, rtol=1e-2)
        end
        @test isapprox(sum(ismissing.(y[s]))/length(s), pm, rtol=1e-1)
    end

    # test ordering is preserved if no shuffle
    s1, s2 = partition(eachindex(y), 0.7, stratify=y)
    @test issorted(s1)
    @test issorted(s2)

    s1, s2 = partition(eachindex(y), 0.7, stratify=y, shuffle=true)
    @test !issorted(s1)
    @test !issorted(s2)
end

@testset "unpack" begin
    channing = TypedTables.Table(
        Sex = categorical(["Female", "Male", "Female"]),
        Entry = Int32[965, 988, 850],
        Exit = Int32[1088, 1045, 940],
        Time = Int32[123, 57, 90],
        Cens = Int32[0, 0, 1],
        weights = [1,2,5])

    w, y, X =  unpack(channing,
                  ==(:weights),
                  ==(:Exit),
                  x -> x != :Time;
                  :Exit=>Continuous,
                  :Entry=>Continuous,
                  :Cens=>Multiclass)

    @test w == selectcols(channing, :weights)
    @test y == selectcols(channing, :Exit)
    @test X == selectcols(channing, [:Sex, :Entry, :Cens])
    @test scitype_union(y) <: Continuous
    @test scitype_union(selectcols(X, :Cens)) <: Multiclass


    w, y, X =  unpack(channing,
                      ==(:weights),
                      ==(:Exit),
                      x -> x != :Time;
                      wrap_singles=true,
                      :Exit=>Continuous,
                      :Entry=>Continuous,
                      :Cens=>Multiclass)

    @test selectcols(w, 1)  == selectcols(channing, :weights)
    @test selectcols(y, 1)  == selectcols(channing, :Exit)
    @test X == selectcols(channing, [:Sex, :Entry, :Cens])

    @test_throws(Exception, unpack(channing,
                                   ==(:weights),
                                   ==(:Exit),
                                   ==(:weights),
                                   x -> x != :Time;
                                   :Exit=>Continuous,
                                   :Entry=>Continuous,
                                   :Cens=>Multiclass))

    # shuffling:
    small = (x=collect(1:5), y = collect("abcde"))
    x, y = unpack(small, ==(:x), ==(:y); shuffle=true, rng=1)
    @test x == [5, 1, 4, 2, 3]
    @test y == ['e', 'a', 'd', 'b', 'c']
    x, y = unpack(small, ==(:x), ==(:y); rng=1)
    @test x == [5, 1, 4, 2, 3]
    @test y == ['e', 'a', 'd', 'b', 'c']
    @test unpack(small, ==(:x), ==(:y); shuffle=true, rng=StableRNG(66)) ==
        unpack(small, ==(:x), ==(:y); rng=StableRNG(66))
end

@testset "restrict and corestrict" begin
    f = ([1], [2, 3], [4, 5, 6, 7], [8, 9, 10])
    @test complement(f, 1) == [2, 3, 4, 5, 6, 7, 8, 9, 10]
    @test complement(f, 2) == [1, 4, 5, 6, 7, 8, 9, 10]
    @test complement(f, 3) == [1, 2, 3, 8, 9, 10]
    @test complement(f, 4) == [1, 2, 3, 4, 5, 6, 7]

    X = 10:10:100
    @test restrict(X, f, 3) == 40:10:70
    @test corestrict(X, f, 3) == [10, 20, 30, 80, 90, 100]
end

@testset "coverage" begin
    @test_throws DomainError partition(1:10, 1.5)

    @test_throws MethodError selectrows(Val(:other), (1,), (1,))
    @test_throws MethodError selectcols(Val(:other), (1,), (1,))
    @test_throws MethodError select(Val(:other), (1,), (1,), (1,))
    @test_throws MethodError nrows(Val(:other), (1,))

    nt = (a=5, b=7)
    @test MLJBase.project(nt, :) == (a=5, b=7)
    @test MLJBase.project(nt, :a) == (a=5, )
    @test MLJBase.project(nt, 1) == (a=5, )

    X = MLJBase.table((x=[1,2,3], y=[4,5,6]))
    @test select(X, 1, :y) == 4
end

@testset "transforming from raw values and categorical values" begin
    values = vcat([missing, ], collect("asdfjklqwerpoi"))
    Xraw = rand(rng,values, 15, 10)
    X = categorical(Xraw)
    element = skipmissing(X) |> first

    @test transform(element, missing) |> ismissing

    raw = first(skipmissing(Xraw))
    c = transform(element, raw)
    @test Set(classes(c)) == Set(classes(X))
    @test c == first(skipmissing(X))

    RAW = Xraw[2:end-1,2:end-1]
    C = transform(element, RAW)
    @test Set(classes(C)) == Set(classes(X))
    @test identity.(skipmissing(C)) ==
        identity.(skipmissing(X[2:end-1,2:end-1]))

    raw = first(skipmissing(Xraw))
    c = transform(X, raw)
    @test Set(classes(c)) == Set(classes(X))
    @test c == first(skipmissing(X))

    RAW = Xraw[2:end-1,2:end-1]
    C = transform(X, RAW)
    @test Set(classes(C)) == Set(classes(X))
    @test identity.(skipmissing(C)) ==
        identity.(skipmissing(X[2:end-1,2:end-1]))
end

@testset "skipinvalid" begin
    w = rand(5)
    @test MLJBase.skipinvalid([1, 2, missing, 3, NaN], [missing, 5, 6, 7, 8]) ==
        ([2, 3], [5, 7])
    @test(
        MLJBase._skipinvalid([1, 2, missing, 3, NaN],
                            [missing, 5, 6, 7, 8],
                            w) ==
        ([2, 3], [5, 7], w[[2,4]]))
    @test(
        MLJBase._skipinvalid([1, 2, missing, 3, NaN],
                            [missing, 5, 6, 7, 8],
                            nothing) ==
        ([2, 3], [5, 7], nothing))
end

end # module

true
