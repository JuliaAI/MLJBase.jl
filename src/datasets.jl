# see also the non-macro versions in datasets_requires.jl

"""Load a well-known public regression dataset with `Continuous` features."""
macro load_boston()
    quote
        import CSV
        y, X = unpack(load_boston(), ==(:MedV), x->x != :Chas)
        (X, y)
    end
end

"""Load a reduced version of the well-known Ames Housing task"""
macro load_reduced_ames()
    quote
        import CSV
        y, X = unpack(load_reduced_ames(), ==(:target), x-> true)
        (X, y)
    end
end

"""Load the full version of the well-known Ames Housing task."""
macro load_ames()
    quote
        import CSV
        y, X = unpack(load_ames(), ==(:target), x->x != :Id)
        (X, y)
    end
end

"""Load a well-known public classification task with nominal features."""
macro load_iris()
    quote
        import CSV
        y, X = unpack(load_iris(), ==(:target), x-> true)
        (X, y)
    end
end

"""Load a well-known crab classification dataset with nominal features."""
macro load_crabs()
    quote
        import CSV
        y, X = unpack(load_crabs(), ==(:sp), x-> !(x in [:sex, :index]))
        (X, y)
    end
end

