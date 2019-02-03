module TestIntrospection

# using Revise
using MLJBase
import MLJBase
using Test

mutable struct Dummy <: Deterministic{Int} end

MLJBase.load_path(::Type{Dummy}) = "GreatPackage.MLJ.Dummy"
MLJBase.output_kind(::Type{Dummy}) = :continuous
MLJBase.output_quantity(::Type{Dummy}) = :univariate
MLJBase.input_kinds(::Type{Dummy}) = [:continuous, ]
MLJBase.input_quantity(::Type{Dummy}) = :univariate
MLJBase.is_pure_julia(::Type{Dummy}) = :yes
MLJBase.package_name(::Type{Dummy}) = "GreatPackage"
MLJBase.package_uuid(::Type{Dummy}) = "6f286f6a-111f-5878-ab1e-185364afe411"
MLJBase.package_url(::Type{Dummy}) = "https://mickey.mouse.org"

d = Dict(:name => "Dummy",
         :load_path => "GreatPackage.MLJ.Dummy",
         :is_pure_julia => :yes,
         :package_uuid  => "6f286f6a-111f-5878-ab1e-185364afe411",
         :package_name  => "GreatPackage",
         :output_kind     => :continuous,
         :output_is_probabilistic => :no,
         :output_quantity  => :univariate,
         :input_kinds => Symbol[:continuous],
         :input_quantity => :univariate,
         :package_url   => "https://mickey.mouse.org",
         :learning_type => :supervised)

info(Dummy)[:name]
@test info(Dummy) == d
@test info(Dummy()) == d

end
true
