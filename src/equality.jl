"""
    is_same_except(m1::MLJType, m2::MLJType, exceptions::Symbol...)

Returns `true` only the following conditions all hold:

- `m1` and `m2` have the same type.

- `m1` and `m2` have the same undefined fields.

- Corresponding fields agree, or are listed as
  `exceptions`, or have `AbstractRNG` as values (one or both)

Here "agree" is in the sense of "==", unless the objects are
themselves of `MLJType`, in which case agreement is in the sense of
`is_same_except` with no exceptions allowed.

Note that Base.== is overloaded such that `m1 == m2` if and only if
`is_same_except(m1, m2)`.

"""
is_same_except(x1, x2) = ==(x1, x2)
function is_same_except(m1::M1, m2::M2,
            exceptions::Symbol...) where {M1<:MLJType,M2<:MLJType}
    if typeof(m1) != typeof(m2)
        return false
    end
    defined1 = filter(fieldnames(M1)|>collect) do fld
        isdefined(m1, fld) && !(fld in exceptions)
    end
    defined2 = filter(fieldnames(M1)|>collect) do fld
        isdefined(m2, fld) && !(fld in exceptions)
    end
    if defined1 != defined2
        return false
    end
    same_values = true
    for fld in defined1
        same_values = same_values &&
            (is_same_except(getfield(m1, fld), getfield(m2, fld)) ||
             getfield(m1, fld) isa AbstractRNG ||
             getfield(m2, fld) isa AbstractRNG)
    end
    return same_values
end

import Base.==

==(m1::M1, m2::M2) where {M1<:MLJType,M2<:MLJType} = is_same_except(m1, m2)

# for using `replace` or `replace!` on collections of MLJType objects
# (eg, Model objects in a learning network) we need a stricter
# equality:
MLJBase.isequal(m1::MLJType, m2::MLJType) = (m1 === m2)

## TODO: Do we need to overload hash here?

# Note: To prevent julia crash, it seems we mysteriously need to
# annotate the type of itr:
function special_in(x, itr)
    anymissing = false
    for y in itr
        v = (y === x)
        if ismissing(v)
            anymissing = true
        elseif v
            return true
        end
    end
    return anymissing ? missing : false
end
Base.in(x::MLJType, itr::Set) = special_in(x, itr)
Base.in(x::MLJType, itr::AbstractVector) = special_in(x, itr)
Base.in(x::MLJType, itr::NTuple) = special_in(x, itr)

