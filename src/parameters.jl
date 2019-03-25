"""
    params(m)

Recursively convert any object of subtype `MLJType` into a named
tuple, keyed on the fields of `m`. The named tuple is possibly nested
because `params` is recursively applied to the field values, which
themselves might be `MLJType` objects. 

Used, in particluar, in the case that `m` is a model, to inspect its
nested hyperparameters:

    julia> params(EnsembleModel(atom=ConstantClassifier()))
    (atom = (target_type = Bool,),
     weights = Float64[],
     bagging_fraction = 0.8,
     rng_seed = 0,
     n = 100,
     parallel = true,)

"""
params(field) = field

function params(m::M) where M<:MLJType
    fields = fieldnames(M)
    NamedTuple{fields}(Tuple([params(getfield(m, field)) for field in fields]))
end
