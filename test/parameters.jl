# using Revise
using Test
using MLJBase

@testset "params method" begin
    struct Opaque
        a::Int
    end
    
    struct Transparent
        A::Int
        B::Opaque
    end
    
    MLJBase.istransparent(::Transparent) = true
    
    struct Dummy <:MLJType
        t::Transparent
        o::Opaque
        n::Integer
    end
    
    t= Transparent(6, Opaque(5))
    m = Dummy(t, Opaque(7), 42)
    
    @test params(m) == (t = (A = 6,
                             B = Opaque(5)),
                        o = Opaque(7),
                        n = 42)
end

@testset "getproperty, setproperty! extensions" begin
    
    m = (a1 = (a11 = 10, a12 = 20), a2 = (a21 = (a211 = 30, a212 = 40),)) 

    @test getproperty(m, :(a1.a12)) == 20
    @test getproperty(m, :a1) == (a11 = 10, a12 = 20)
    @test getproperty(m, :(a2.a21.a212)) == 40

    mutable struct M
        a1
        a2
    end
    mutable struct A1
        a11
        a12
    end
    mutable struct A2
        a21
    end
    mutable struct A21
        a211
        a212
    end

    m = M(A1(10, 20), A2(A21(30, 40)))
    setproperty!(m, :(a2.a21.a212), 42)
    @test getproperty(m, :(a1.a11)) == 10
    @test getproperty(m, :(a1.a12)) == 20
    @test getproperty(m, :(a2.a21.a211)) == 30
    @test getproperty(m, :(a2.a21.a212)) == 42
    @test getproperty(getproperty(m, :(a2.a21)), :a212) == 42

end

true
