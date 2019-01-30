_response(::Type{<:Supervised}) = :unknown
_response(::Type{<:Deterministic}) = :deterministic
_response(::Type{<:Probabilistic}) = :probabilistic

target_is(modeltype::Type{<:Supervised}) =
    [_response(modeltype), target_kind(modeltype), target_quantity(modeltype)]

name(M::Type{<:Model}) = split(string(M), '.')[end] |> String

if VERSION < v"1.0.0"
    import Base.info
end

function info(M::Type{<:Model})

    message = "$M has a bad trait declaration. "

    load_path != "unknown" || error(message*"`MLJBase.load_path($M) should be defined so that "* 
                                    "`using MLJ; import MLJ.load_path($M)` loads `$M` into current namespace.")

    target_kind(M) in [:continuous, :binary, :multiclass, :ordered_factor_finite,
                       :ordered_factor_infinite, :same_as_inputs, :unknown] ||
                           error(message*"`target_kind($M)` must return :numeric, :binary, :multiclass (or :unknown).")

    target_quantity(M) in [:univariate, :multivariate] ||
        error(message*"`target_quantity($M)` must return :univariate or :multivariate")

    issubset(Set(inputs_can_be(M)), Set([:continuous, :multiclass, :ordered_factor_finite,
                                         :ordered_factor_infinite, :missing])) ||
                                             error(message*"`inputs_can_be($M)` must return a vector with entries from [:continuous, :multiclass, "*
                                                   ":ordered_factor_finite, :ordered_factor_infinite, :missing]")

    inputs_quantity(M) in [:univariate, :multivariate] ||
        error(message*"`inputs_quantity($M)` must return :univariate or :multivariate.")
    
    is_pure_julia(M) in [:yes, :no, :unknown] ||
        error(message*"`is_pure_julia($M)` must return :yes, :no (or :unknown).")

    d = Dict{Symbol,Union{Symbol,Vector{Symbol},String}}()
    d[:load_path] = load_path(M)
    d[:name] = name(M)
    d[:inputs_can_be] = inputs_can_be(M)
    d[:inputs_quantity] = inputs_quantity(M)
    d[:is_pure_julia] = is_pure_julia(M)
    d[:package_name] = package_name(M)
    d[:package_uuid] = package_uuid(M)
    d[:package_url] = package_url(M)
    if M <: Supervised
        d[:target_is] = target_is(M)
    end
    
    return d
end
info(model::Supervised) = info(typeof(model))
