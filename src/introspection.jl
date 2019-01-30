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

    load_path != "unknown" || error(message*"load_path has fallback value \"unknown\".")

    if M <: Supervised
        target_kind(M) in [:numeric, :binary, :multiclass, :unknown] ||
            error(message*"target_kind must return :numeric, :binary, :multiclass (or :unknown).")
        target_quantity(M) in [:univariate, :multivariate] ||
            error(message*"target_quantity must return :univariate or :multivariate")
    end
    
    issubset(Set(inputs_can_be(M)), Set([:numeric, :nominal, :missing])) ||
        error(message*"inputs_can_be must return a vector with entries from [:numeric, :nominal, :missing]")
    is_pure_julia(M) in [:yes, :no, :unknown] ||
        error(message*"is_pure_julia must return :yes, :no (or :unknown).")

    d = Dict{Symbol,Union{Symbol,Vector{Symbol},String}}()
    d[:load_path] = load_path(M)
    d[:name] = name(M)
    d[:inputs_can_be] = inputs_can_be(M)
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
