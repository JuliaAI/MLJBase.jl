# by default, MLJType objects are `==` if: (i) they have a common
# supertype AND (ii) they have the same set of defined fields AND
# (iii) their defined field values are `==` OR the values are both
# AbstractRNG objects.
function ==(m1::M1, m2::M2) where {M1<:MLJType,M2<:MLJType}
    if M1 != M1
        return false
    end
    defined1 = filter(fieldnames(M1)|>collect) do fld
        isdefined(m1, fld)
    end
    defined2 = filter(fieldnames(M1)|>collect) do fld
        isdefined(m2, fld)
    end
    if defined1 != defined2
        return false
    end
    same_values = true
    for fld in defined1
        same_values = same_values &&
            (getfield(m1, fld) == getfield(m2, fld) ||
             getfield(m1, fld) isa AbstractRNG) 
    end
    return same_values
end
