struct Checker{is_supervised,
                  supports_weights,
                  input_scitype,
                    target_scitype} end

function Base.getproperty(::Checker{is_supervised,
                                    supports_weights,
                                    input_scitype,
                                    target_scitype},
                          field::Symbol) where {is_supervised,
                                                supports_weights,
                                                input_scitype,
                                                target_scitype}
    if field === :is_supervised
        return is_supervised
    elseif field === :supports_weights
        return supports_weights
    elseif field === :input_scitype
        return input_scitype
    elseif field === :target_scitype
        return target_scitype
    else
        throw(ArgumentError("Unsupported property. "))
    end
end

Base.propertynames(::Checker) =
    (:is_supervised, :supports_weights, :input_scitype, :target_scitype)

function _as_named_tuple(s::Checker)
    names = propertynames(s)
    NamedTuple{names}(Tuple(getproperty(s, p) for p in names))
end

# function Base.show(io::IO, ::MIME"text/plain", S::Checker)
#     show(io, MIME("text/plain"), _as_named_tuple(S))
# end

matching(X)       = Checker{false,false,scitype(X),missing}()
matching(X, y)    = Checker{true,false,scitype(X),scitype(y)}()
matching(X, y, w) = Checker{true,true,scitype(X),scitype(y)}()
