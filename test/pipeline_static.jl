module TestPipelineStatic

using Test
using MLJBase

t  = MLJBase.StaticTransformer(f=log)
f, = fit(t, 1, nothing)
@test transform(t, f, 5) ≈ log(5)

infos = info_dict(t)
@test infos[:input_scitype]  == MLJBase.Table(Scientific)
@test infos[:output_scitype] == MLJBase.Table(Scientific)

end
true
