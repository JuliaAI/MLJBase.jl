module TestUtilities

using Test
using MLJBase

@test MLJBase.finaltypes(Union{Missing,Int}) == [Union{Missing,Int64}]
@test MLJBase.finaltypes(Float64) == [Float64]

abstract type Foo end
struct Bar <: Foo end
struct Baz <: Foo end

@test MLJBase.finaltypes(Foo) == [Bar, Baz]

end # module
true
