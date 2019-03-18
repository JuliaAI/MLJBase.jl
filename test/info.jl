module TestIntrospection

# using Revise
using MLJBase
import MLJBase
using Test

mutable struct Dummy <: Probabilistic{Int} end

MLJBase.load_path(::Type{Dummy}) = "GreatPackage.MLJ.Dummy"
MLJBase.target_scitype(::Type{Dummy}) = Continuous
MLJBase.input_scitypes(::Type{Dummy}) = Union{Continuous, Discrete, Missing}
MLJBase.input_is_multivariate(::Type{Dummy}) = false
MLJBase.is_pure_julia(::Type{Dummy}) = true
MLJBase.package_name(::Type{Dummy}) = "GreatPackage"
MLJBase.package_uuid(::Type{Dummy}) = "6f286f6a-111f-5878-ab1e-185364afe411"
MLJBase.package_url(::Type{Dummy}) = "https://mickey.mouse.org"

d = Dict(:name => "Dummy",
         :load_path => "GreatPackage.MLJ.Dummy",
         :is_pure_julia => true,
         :package_uuid  => "6f286f6a-111f-5878-ab1e-185364afe411",
         :package_name  => "GreatPackage",
         :target_scitype => Continuous,
         :input_scitypes => Union{Continuous, Discrete, Missing},
         :input_is_multivariate => false,
         :is_probabilistic => true,
         :package_url   => "https://mickey.mouse.org",
         :is_supervised => true,
         :is_wrapper => false)

info(Dummy)[:name]
@test info(Dummy) == d
@test info(Dummy()) == d

end
true
