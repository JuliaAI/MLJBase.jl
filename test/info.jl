module TestIntrospection

# using Revise
using MLJBase
import MLJBase
using Test

mutable struct Dummy <: Probabilistic end

MLJBase.load_path(::Type{Dummy}) = "GreatPackage.MLJ.Dummy"
MLJBase.scitype_X(::Type{Dummy}) = MLJBase.TableScitype(Set([Finite]))
MLJBase.scitype_y(::Type{Dummy}) = MLJBase.VectorScitype(Continuous)
MLJBase.is_pure_julia(::Type{Dummy}) = true
MLJBase.package_name(::Type{Dummy}) = "GreatPackage"
MLJBase.package_uuid(::Type{Dummy}) = "6f286f6a-111f-5878-ab1e-185364afe411"
MLJBase.package_url(::Type{Dummy}) = "https://mickey.mouse.org"

d = Dict(:name => "Dummy",
         :load_path => "GreatPackage.MLJ.Dummy",
         :is_pure_julia => true,
         :package_uuid  => "6f286f6a-111f-5878-ab1e-185364afe411",
         :package_name  => "GreatPackage",
         :scitype_X => MLJBase.TableScitype(Set([Finite])),
         :scitype_y => MLJBase.VectorScitype(Continuous),
         :is_probabilistic => true,
         :package_url   => "https://mickey.mouse.org",
         :is_supervised => true,
         :is_wrapper => false)

info(Dummy)[:name]
@test info(Dummy) == d
@test info(Dummy()) == d

end
true
