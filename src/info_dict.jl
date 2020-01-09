# `info_dict` returns a dictionary of model trait values suitable, after
# encoding, to serializing to TOML file. Not intended to be exposed to
# user.

info_dict(M::Type{<:Supervised}) = info_dict(M, SUPERVISED_TRAITS)
info_dict(M::Type{<:Unsupervised}) = info_dict(M, UNSUPERVISED_TRAITS)
info_dict(model::Model) = info_dict(typeof(model))

function info_dict(M::Type{<:Model}, traits)

    message = "$M has a bad trait declaration.\n"

    is_pure_julia(M) == nothing || is_pure_julia(M) isa Bool ||
        error(message*"is_pure_julia($M) must return true, false or nothing. ")

    :supports_weights in traits &&
        !(supports_weights(M) isa Bool) &&
        error(message*"supports_weights($M) must return true, false or missing. ")

    :is_wrapper in traits &&
        !(is_wrapper(M) isa Bool) &&
        error(message*"is_wrapper($M) must return true, false. ")

    return LittleDict{Symbol,Any}(trait => eval(:($trait))(M)
                                  for trait in traits)
end
