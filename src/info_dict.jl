# `info_dict` returns a dictionary of model trait values suitable, after
# encoding, to serializing to TOML file. Not intended to be exposed to
# user.

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

    dd = DefaultDict{Symbol, Any}(passkey=true) do key
        getfield(MLJBase, key)(M)
    end

    return dd
end
