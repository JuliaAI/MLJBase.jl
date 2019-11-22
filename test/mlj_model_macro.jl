module TestMacroMLJ

using MLJBase, Test, Distances

# No type, no default
@mlj_model mutable struct A1
    a
end
a = A1()
@test ismissing(a.a)
a.a = 5
@test a.a == 5

# No type, with default
@mlj_model mutable struct A1b
    a = 5
end
a = A1b()
@test a.a == 5
a.a = "hello"
@test a.a == "hello"

# If a type is given but no default value is given, then the macro tries to fill
# a default value; either 0 if it's a Number type, or an empty string and otherwise fails.
@mlj_model mutable struct A1c
    a::Int
end
a = A1c()
@test a.a == 0
a = A1c(a=7)
@test a.a == 7
@test_throws InexactError A1c(a=5.3)
@test_throws MethodError A1c(a="hello")

# Type is given and default is given
@mlj_model mutable struct A1d
    a::Int = 5
end
a = A1d()
@test a.a == 5
a = A1d(a=7)
@test a.a == 7

# No type is given but a default and constraint
@mlj_model mutable struct A1e
    a = 5::(_ > 0)
end
a = A1e()
@test a.a == 5
a = A1e(a=7)
@test a.a == 7
@test @test_logs (:warn, "Constraint `model.a > 0` failed; using default: a=5.") A1e(a=-1).a==5
a = A1e(a=7.5)
@test a.a == 7.5

# Type is given with default and constraint
@mlj_model mutable struct A1f
    a::Int = 5::(_ > 0)
end
a = A1f()
@test a.a == 5
a = A1f(a=7)
@test a.a == 7
@test_throws InexactError A1f(a=7.5)
@test @test_logs (:warn, "Constraint `model.a > 0` failed; using default: a=5.") A1f(a=-1).a==5

abstract type FooBar end
@mlj_model mutable struct B1a <: FooBar
    a::Symbol = :auto::(_ in (:auto, :semi))
end
b = B1a()
@test b.a == :auto
b = B1a(a=:semi)
@test b.a == :semi
@test @test_logs (:warn, "Constraint `model.a in (:auto, :semi)` failed; using default: a=:auto.") B1a(a=:autos).a == :auto
@test_throws MethodError B1a(b="blah")

# == dependence on other types
@mlj_model mutable struct B1b
    a::SemiMetric = Euclidean()::(_ isa Metric)
end
@test B1b().a isa Euclidean
@test @test_logs (:warn, "Constraint `model.a isa Metric` failed; using default: a=Euclidean().") B1b(a=BhattacharyyaDist()).a isa Euclidean

@mlj_model mutable struct B1c
    a::SemiMetric = Euclidean()
end
@test B1c().a isa Euclidean

# Implicit defaults
@mlj_model mutable struct Ca
    a::String
end
@test Ca().a == ""
@mlj_model mutable struct Cb
    a::Any
end
@test Cb().a === missing
@mlj_model mutable struct Cc
    a::Union{Nothing,Int}
end
@test Cc().a === nothing
@mlj_model mutable struct Cd
    a::Union{Missing,Int}
end
@test Cd().a === missing

end
true
