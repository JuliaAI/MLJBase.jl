is_probabilistic(::Type{<:Model}) = :unknown
is_probabilistic(::Type{<:Deterministic}) = :no
is_probabilistic(::Type{<:Probabilistic}) = :yes

# TODO: depreciate? currently used by ensembles.jl
output_is(modeltype::Type{<:Supervised}) =
    [is_probabilistic(modeltype), output_kind(modeltype), output_quantity(modeltype)]


function coretype(M)
    if isdefined(M, :name)
        return M.name
    else
        return coretype(M.body)
    end
end
    
name(M::Type{<:Model}) = split(string(coretype(M)), '.')[end] |> String

if VERSION < v"1.0.0"
    import Base.info
end

function info(M::Type{<:Model})

    message = "$M has a bad trait declaration. "

    load_path != "unknown" || error(message*"`MLJBase.load_path($M) should be defined so that "* 
                                    "`using MLJ; import MLJ.load_path($M)` loads `$M` into current namespace.")

    output_kind(M) in [:continuous, :binary, :multiclass, :ordered_factor_finite,
                       :ordered_factor_infinite, :same_as_inputs, :unknown] ||
                           error(message*"`output_kind($M)` must return :numeric, :binary, :multiclass (or :unknown).")

    output_quantity(M) in [:univariate, :multivariate] ||
        error(message*"`output_quantity($M)` must return :univariate or :multivariate")

    issubset(Set(input_kinds(M)), Set([:continuous, :multiclass, :ordered_factor_finite,
                                         :ordered_factor_infinite, :missing])) ||
                                             error(message*"`input_kinds($M)` must return a vector with entries from [:continuous, :multiclass, "*
                                                   ":ordered_factor_finite, :ordered_factor_infinite, :missing]")

    input_quantity(M) in [:univariate, :multivariate] ||
        error(message*"`input_quantity($M)` must return :univariate or :multivariate.")
    
    is_pure_julia(M) in [:yes, :no, :unknown] ||
        error(message*"`is_pure_julia($M)` must return :yes, :no (or :unknown).")

    d = Dict{Symbol,Union{Symbol,Vector{Symbol},String}}()
    d[:load_path] = load_path(M)
    d[:name] = name(M)
    d[:input_kinds] = input_kinds(M)
    d[:input_quantity] = input_quantity(M)
    d[:is_pure_julia] = is_pure_julia(M)
    d[:package_name] = package_name(M)
    d[:package_uuid] = package_uuid(M)
    d[:package_url] = package_url(M)
    d[:output_kind] = output_kind(M)
    d[:output_is_probabilistic] = is_probabilistic(M)
    d[:output_quantity] = output_quantity(M)
    return d
end
info(model::Model) = info(typeof(model))
