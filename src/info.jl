# `info` returns a dictionary of model trait values suitable, after
# encoding, to serializing to TOML file. Not intended to be exposed to
# user. The `traits` function, defined in MLJ, returns the trait
# values as a named-tuple, more friendly for user-interaction. One can
# similarly call `traits` on performance measures.

info_dict(M::Type{<:Supervised}) = info_dict(M, SUPERVISED_TRAITS)
info_dict(M::Type{<:Unsupervised}) = info_dict(M, UNSUPERVISED_TRAITS)
info_dict(model::Model) = info_dict(typeof(model))

function info_dict(M::Type{<:Model}, traits)

    message = "$M has a bad trait declaration.\n"

    ismissing(is_pure_julia(M)) || is_pure_julia(M) isa Bool ||
        error(message*"is_pure_julia($M) must return true, false or missing. ")

    :supports_weights in traits &&
        !(supports_weights(M) isa Bool) &&
        error(message*"supports_weights($M) must return true, false or missing. ")

    :is_wrapper in traits &&
        !(is_wrapper(M) isa Bool) &&
        error(message*"is_wrapper($M) must return true, false or missing. ")
    
    return LittleDict{Symbol,Any}(trait => eval(:($trait))(M)
                                  for trait in traits)
end

