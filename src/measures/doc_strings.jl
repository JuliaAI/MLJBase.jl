# the following creates doc-strings for the aliases (`instances`) of each measure:

for m in measures()
    name = m.name
    for instance in m.instances
        alias = Symbol(instance)
        quote
            @doc "An instance of type [`$($name)`](@ref). "*
                "Query the [`$($name)`](@ref) doc-string for details. " $alias
        end |> eval
    end
end
