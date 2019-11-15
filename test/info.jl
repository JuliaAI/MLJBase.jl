module TestInfo

# using Revise
using MLJBase
import MLJBase.info_dict
using Test
using OrderedCollections
using LossFunctions

mutable struct DummyProb <: Probabilistic
    an_int::Int
    a_float::Float64
    a_vector::Vector{Float64}
    untyped
end
MLJBase.load_path(::Type{DummyProb}) = "GreatPackage.MLJ.DummyProb"
MLJBase.input_scitype(::Type{DummyProb}) = MLJBase.Table(Finite)
MLJBase.target_scitype(::Type{DummyProb}) = AbstractVector{<:Continuous}
MLJBase.is_pure_julia(::Type{DummyProb}) = true
MLJBase.supports_weights(::Type{DummyProb}) = true
MLJBase.package_name(::Type{DummyProb}) = "GreatPackage"
MLJBase.package_uuid(::Type{DummyProb}) = "6f286f6a-111f-5878-ab1e-185364afe411"
MLJBase.package_url(::Type{DummyProb}) = "https://mickey.mouse.org"
MLJBase.package_license(::Type{DummyProb}) = "MIT"
MLJBase.predict(::DummyProb, fr, X) = nothing

mutable struct DummyDeterm <: Deterministic end
MLJBase.load_path(::Type{DummyDeterm}) = "GreatPackage.MLJ.DummyDeterm"
MLJBase.input_scitype(::Type{DummyDeterm}) = MLJBase.Table(Finite)
MLJBase.target_scitype(::Type{DummyDeterm}) = AbstractVector{<:Continuous}
MLJBase.is_pure_julia(::Type{DummyDeterm}) = true
MLJBase.supports_weights(::Type{DummyDeterm}) = true
MLJBase.package_name(::Type{DummyDeterm}) = "GreatPackage"
MLJBase.package_uuid(::Type{DummyDeterm}) = "6f286f6a-111f-5878-ab1e-185364afe411"
MLJBase.package_url(::Type{DummyDeterm}) = "https://mickey.mouse.org"
MLJBase.package_license(::Type{DummyDeterm}) = "MIT"
MLJBase.predict(::DummyDeterm, fr, X) = nothing

mutable struct DummyInt <: Interval end
MLJBase.load_path(::Type{DummyInt}) = "GreatPackage.MLJ.DummyInt"
MLJBase.input_scitype(::Type{DummyInt}) = MLJBase.Table(Finite)
MLJBase.target_scitype(::Type{DummyInt}) = AbstractVector{<:Continuous}
MLJBase.is_pure_julia(::Type{DummyInt}) = true
MLJBase.supports_weights(::Type{DummyInt}) = true
MLJBase.package_name(::Type{DummyInt}) = "GreatPackage"
MLJBase.package_uuid(::Type{DummyInt}) = "6f286f6a-111f-5878-ab1e-185364afe411"
MLJBase.package_url(::Type{DummyInt}) = "https://mickey.mouse.org"
MLJBase.package_license(::Type{DummyInt}) = "MIT"
MLJBase.predict(::DummyInt, fr, X) = nothing

mutable struct DummyUnsup <: Unsupervised end
MLJBase.load_path(::Type{DummyUnsup}) = "GreatPackage.MLJ.DummyUnsup"
MLJBase.input_scitype(::Type{DummyUnsup}) = MLJBase.Table(Finite)
MLJBase.output_scitype(::Type{DummyUnsup}) = AbstractVector{<:Continuous}
MLJBase.is_pure_julia(::Type{DummyUnsup}) = true
MLJBase.supports_weights(::Type{DummyUnsup}) = true
MLJBase.package_name(::Type{DummyUnsup}) = "GreatPackage"
MLJBase.package_uuid(::Type{DummyUnsup}) = "6f286f6a-111f-5878-ab1e-185364afe411"
MLJBase.package_url(::Type{DummyUnsup}) = "https://mickey.mouse.org"
MLJBase.package_license(::Type{DummyUnsup}) = "MIT"
MLJBase.transform(::DummyUnsup, fr, X) = nothing

@testset "info on probabilistic models" begin

    d = LittleDict{Symbol,Any}(:name => "DummyProb",
                               :load_path => "GreatPackage.MLJ.DummyProb",
                               :is_pure_julia => true,
                               :package_uuid  => "6f286f6a-111f-5878-ab1e-185364afe411",
                               :package_name  => "GreatPackage",
                               :package_license => "MIT",
                               :input_scitype => MLJBase.Table(Finite),
                               :supports_weights => true,
                               :target_scitype => MLJBase.AbstractVector{<:Continuous},
                               :prediction_type => :probabilistic,
                               :package_url   => "https://mickey.mouse.org",
                               :is_supervised => true,
                               :is_wrapper => false,
                               :docstring => "DummyProb from GreatPackage.jl.\n[Documentation](https://mickey.mouse.org).",
                               :implemented_methods => [:predict, ],
                               :hyperparameter_types  => ["Int64", "Float64",
                                                     "Array{Float64,1}", "Any"],
                               :hyperparameters => [:an_int, :a_float,
                                                    :a_vector, :untyped])
    
    
    @test info_dict(DummyProb) == d
    @test info_dict(DummyProb(42, 3.14, [1.0, 2.0], :cow)) == d
    # for k in keys(d)
    #       println(string(k, " ",  info_dict(DummyProb)[k] == d[k]))
    # end
end



@testset "info on deterministic models" begin

    d = LittleDict{Symbol,Any}(:name => "DummyDeterm",
                               :load_path => "GreatPackage.MLJ.DummyDeterm",
                               :is_pure_julia => true,
                               :package_uuid  => "6f286f6a-111f-5878-ab1e-185364afe411",
                               :package_name  => "GreatPackage",
                               :package_license => "MIT",
                               :input_scitype => MLJBase.Table(Finite),
                               :supports_weights => true,
                               :target_scitype => MLJBase.AbstractVector{<:Continuous},
                               :prediction_type => :deterministic,
                               :package_url   => "https://mickey.mouse.org",
                               :is_supervised => true,
                               :is_wrapper => false,
                               :docstring => "DummyDeterm from GreatPackage.jl.\n[Documentation](https://mickey.mouse.org).",
                               :implemented_methods => [:predict, ],
                               :hyperparameter_types  => [],
                               :hyperparameters => [])

    
    @test info_dict(DummyDeterm) == d
    @test info_dict(DummyDeterm()) == d
    # for k in keys(d)
    #      println(string(k, " ",  info_dict(DummyDeterm)[k] == d[k]))
    # end
    
end

@testset "info on interval models" begin
    
    d = LittleDict{Symbol,Any}(:name => "DummyInt",
                               :load_path => "GreatPackage.MLJ.DummyInt",
                               :is_pure_julia => true,
                               :package_uuid  => "6f286f6a-111f-5878-ab1e-185364afe411",
                               :package_name  => "GreatPackage",
                               :package_license => "MIT",
                               :input_scitype => MLJBase.Table(Finite),
                               :supports_weights => true,
                               :target_scitype => MLJBase.AbstractVector{<:Continuous},
                               :prediction_type => :interval,
                               :package_url   => "https://mickey.mouse.org",
                               :is_supervised => true,
                               :is_wrapper => false,
                               :docstring => "DummyInt from GreatPackage.jl.\n[Documentation](https://mickey.mouse.org).",
                               :implemented_methods => [:predict, ],
                               :hyperparameter_types  => [],
                               :hyperparameters => [])
    
    @test info_dict(DummyInt) == d
    @test info_dict(DummyInt()) == d
    # for k in keys(d)
    #      println(string(k, " ",  info_dict(DummyInt)[k] == d[k]))
    # end
    
end

@testset "info on unsupervised models" begin

    d = LittleDict{Symbol,Any}(:name => "DummyUnsup",
                               :load_path => "GreatPackage.MLJ.DummyUnsup",
                               :is_pure_julia => true,
                               :package_uuid  => "6f286f6a-111f-5878-ab1e-185364afe411",
                               :package_name  => "GreatPackage",
                               :package_license => "MIT",
                               :input_scitype => MLJBase.Table(Finite),
                               :output_scitype => MLJBase.AbstractVector{<:Continuous},
                               :package_url   => "https://mickey.mouse.org",
                               :is_supervised => false,
                               :is_wrapper => false,
                               :docstring => "DummyUnsup from GreatPackage.jl.\n[Documentation](https://mickey.mouse.org).",
                               :implemented_methods => [:transform, ],
                               :hyperparameter_types  => [],
                               :hyperparameters => [])
    
    @test info_dict(DummyUnsup) == d
    @test info_dict(DummyUnsup()) == d
    # for k in keys(d)
    #     println(string(k, " ",  info_dict(DummyUnsup)[k] == d[k]))
    # end

end

@testset "info for measures" begin
    @test info(rms).name == "rms"
    info(L2DistLoss()).name == "LPDistLoss{2}"
end

end
true
