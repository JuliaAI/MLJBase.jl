module TestMetadataUtils

using MLJBase, Test

@mlj_model mutable struct FooRegressor <: Deterministic
    a::Int = 0::(_ ≥ 0)
    b
end
metadata_pkg(FooRegressor,
    name="FooRegressor",
    uuid="10745b16-79ce-11e8-11f9-7d13ad32a3b2",
    url="http://existentialcomics.com/",
    julia=true,
    license="MIT",
    is_wrapper=false
    )
metadata_model(FooRegressor,
    input=MLJBase.Table(Continuous),
    target=AbstractVector{Continuous},
    descr="La di da")

infos = info_dict(FooRegressor)

@test infos[:input_scitype] == MLJBase.Table(Continuous)
@test infos[:target_scitype] == AbstractVector{Continuous}
@test infos[:is_pure_julia]
@test !infos[:is_wrapper]
@test infos[:docstring] == raw"""La di da
    → based on [FooRegressor](http://existentialcomics.com/)
    → do `@load FooRegressor` to use the model
    → do `?FooRegressor` for documentation."""
@test infos[:name] == "FooRegressor"

@test infos[:is_supervised]
@test infos[:prediction_type] == :deterministic

end
true
