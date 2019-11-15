## MODEL TRAITS

# model trait names:
const MODEL_TRAITS = [:input_scitype, :output_scitype, :target_scitype,
                    :is_pure_julia, :package_name, :package_license,
                    :load_path, :package_uuid, 
                    :package_url, :is_wrapper, :supports_weights, :docstring,
                    :name, :is_supervised, :prediction_type,
                    :implemented_methods, :hyperparameters,
                    :hyperparameter_types]
const SUPERVISED_TRAITS = filter(MODEL_TRAITS) do trait
    !(trait in [:output_scitype,])
end
const UNSUPERVISED_TRAITS = filter(MODEL_TRAITS) do trait
    !(trait in [:target_scitype, :prediction_type, :supports_weights])
end

# fallback trait declarations:
input_scitype(::Type) = Unknown
output_scitype(::Type) = Unknown
target_scitype(::Type) = Unknown  # used for measures too
is_pure_julia(::Type) = missing
package_name(::Type) = "unknown"
package_license(::Type) = "unknown"
load_path(::Type) = "unknown"
package_uuid(::Type) = "unknown"
package_url(::Type) = "unknown"
is_wrapper(::Type) = false
supports_weights(::Type) = false  # used for measures too
docstring(object::Type{<:MLJType}) =
    "$(name(object)) from $(package_name(object)).jl.\n"*
"[Documentation]($(package_url(object)))."

# "derived" traits:
name(M::Type) = string(M)
name(M::Type{<:MLJType}) = split(string(coretype(M)), '.')[end] |> String
is_supervised(::Type{<:Type}) = false
is_supervised(::Type{<:Supervised}) = true
prediction_type(::Type) = :unknown # used for measures too
prediction_type(::Type{<:Deterministic}) = :deterministic
prediction_type(::Type{<:Probabilistic}) = :probabilistic
prediction_type(::Type{<:Interval}) = :interval
implemented_methods(M::Type{<:MLJType}) = map(f->f.name, methodswith(M))
hyperparameters(M::Type) = collect(fieldnames(M))
_fieldtypes(M) = [fieldtype(M, fld) for fld in fieldnames(M)]
hyperparameter_types(M::Type) = string.(_fieldtypes(M))
# function hyperparmeter_defaults(M::Type)
#     try
#         model = M()
#         return [Meta.parse(string(getproperty(model, fld)))
#                 for fld in fieldnames(M)]
#     catch
#         return []
#     end
# end
                    

# following 5 lines commented out because they dissallow precompilation:
# for trait in MODEL_TRAITS
#     eval(quote
#         $trait(object) = $trait(typeof(object))
#     end)
# end

input_scitype(object) = input_scitype(typeof(object))
output_scitype(object) = output_scitype(typeof(object))
target_scitype(object) = target_scitype(typeof(object))
is_pure_julia(object) = is_pure_julia(typeof(object))
package_name(object) = package_name(typeof(object))
package_license(object) = package_license(typeof(object))
load_path(object) = load_path(typeof(object))
package_uuid(object) = package_uuid(typeof(object))
package_url(object) = package_url(typeof(object))
is_wrapper(object) = is_wrapper(typeof(object))
supports_weights(object) = supports_weights(typeof(object))
docstring(object) = docstring(typeof(object))
name(object) = name(typeof(object))
is_supervised(object) = is_supervised(typeof(object))
prediction_type(object) = prediction_type(typeof(object))
implemented_methods(object) = implemented_methods(typeof(object))
hyperparameters(object) = hyperparameters(typeof(object))
hyperparameter_types(object) = hyperparameter_types(typeof(object))
