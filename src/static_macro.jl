macro static(ex)
    if ex.args[1] isa Expr
        part = ex.args[1]
        fn   = part.args[1]
        if fn ∈ (:transform, :inverse_transform)
            m = part.args[2] # :(s::Scaling)
            T = m.args[2]

            return esc(
                quote
                    $ex
                    $fn($m, ::Nothing, args...) = $fn($m, args...)
                end
                )
        end
    end
    ex
end
