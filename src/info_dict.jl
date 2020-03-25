# `info_dict` returns a dictionary of model trait values suitable, after
# encoding, to serializing to TOML file. Not intended to be exposed to
# user.

struct LazyInfoDict{K, V, M} <: AbstractDict{K, V}
    d::LittleDict{K, V}
    keys::Vector{K}
    m::M
end

function Base.:getindex(d::LazyInfoDict, key)
    if haskey(d.d, key)
        return d.d[key]
    else
        # safety measures
        !(key in d.keys) && throw(KeyError(key))

        val = getfield(MLJBase, key)(d.m)
        d.d[key] = val
        return val
    end
end

Base.:keys(d::LazyInfoDict) = d.keys
Base.:length(d::LazyInfoDict) = length(d.keys)
Base.:haskey(d::LazyInfoDict, key) = key in d.keys

info_dict(model::Model) = info_dict(typeof(model))

function info_dict(M::Type{<:Model})
    message = "$M has a bad trait declaration.\n"
    is_pure_julia(M) isa Bool ||
        error(message * "`is_pure_julia($M)` must return true or false")
    supports_weights(M) isa Bool ||
        error(message * "`supports_weights($M)` must return true, "*
              "false or missing. ")
    is_wrapper(M) isa Bool ||
        error(message * "`is_wrapper($M)` must return true, false. ")

    return LazyInfoDict(LittleDict{Symbol, Any}(), MLJModelInterface.MODEL_TRAITS, M)
end
