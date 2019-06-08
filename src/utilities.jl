"""
    partition(rows::AbstractVector{Int}, fractions...; shuffle=false)

Splits the vector `rows` into a tuple of vectors whose lengths are
given by the corresponding `fractions` of `length(rows)`. The last
fraction is not provided, as it is inferred from the preceding
ones. So, for example,

    julia> partition(1:1000, 0.2, 0.7)
    (1:200, 201:900, 901:1000)

"""
function partition(rows::AbstractVector{Int}, fractions...; shuffle::Bool=false)
    rows = collect(rows)
    shuffle && shuffle!(rows)
    rowss = []
    if sum(fractions) >= 1
        throw(DomainError)
    end
    n_patterns = length(rows)
    first = 1
    for p in fractions
        n = round(Int, p*n_patterns)
        n == 0 ? (@warn "A split has only one element"; n = 1) : nothing
        push!(rowss, rows[first:(first + n - 1)])
        first = first + n
    end
    if first > n_patterns
        @warn "Last vector in the split has only one element."
        first = n_patterns
    end
    push!(rowss, rows[first:n_patterns])
    return tuple(rowss...)
end

function finaltypes(T::Type)
    s = InteractiveUtils.subtypes(T)
    if isempty(s)
        return [T, ]
    else
        return reduce(vcat, [finaltypes(S) for S in s])
    end
end

import Random
import Base.length


function unique_inverse(A::AbstractArray)
    out = Array{eltype(A)}(undef, 0)
    out_idx = Array{Vector{Int}}(undef, 0)
    seen = Dict{eltype(A), Int}()
    for (idx, x) in enumerate(A)
        if !in(x, keys(seen))
            seen[x] = length(seen) + 1
            push!(out, x)
            push!(out_idx, Int[])
        end
        push!(out_idx[seen[x]], idx)
    end
    out, out_idx
end 


"""
    StratifiedKFold(strata,k)
Struct for StratifiedKFold
provide strata's and number of partitions(k) and simply collect the object for the indices. 
Taken from MLBase (https://github.com/JuliaStats/MLBase.jl).
"""
struct StratifiedKFold 
    n::Int                         #Total number of observations
    permseqs::Vector{Vector{Int}}  #Vectors of vectors of indexes for each stratum
    k::Int                         #Number of splits
    coeffs::Vector{Float64}        #About how many observations per strata are in a val set
    function StratifiedKFold(strata, k)
        2 <= k <= length(strata) || error("The value of k must be in [2, length(strata)].")
        strata_labels, permseqs = unique_inverse(strata)
        map(Random.shuffle!, permseqs)
        coeffs = Float64[]
        for (stratum, permseq) in zip(strata_labels, permseqs)
            k <= length(permseq) || error("k is greater than the length of stratum $stratum")
            push!(coeffs, length(permseq) / k)
        end
        new(length(strata), permseqs, k, coeffs)
    end
end
length(c::StratifiedKFold) = c.k
function Base.iterate(c::StratifiedKFold, s::Int=1)
    (s > c.k) && return nothing
    r = Int[]
    for (permseq, coeff) in zip(c.permseqs, c.coeffs)
        a, b = round.(Integer, [s-1, s] .* coeff)
        append!(r, view(permseq, a+1:b))
    end
    return setdiff(1:c.n, r), s+1
end

"""

    @set_defaults ModelType(args...)
    @set_defaults ModelType args

Create a keyword constructor for any type `ModelType<::MLJBase.Model`,
using as default values those listed in `args`. These must include a
value for every field, and in the order appearing in
`fieldnames(ModelType)`.

The constructor created calls `MLJBase.clean!(model)` on the
instantiated object `model` and calls `@warn messsage` if `messsage =
MLJBase.clean!(model)` is non-empty. Note that `MLJBase.clean!` has a
trivial fallback defined for all subtypes of `MLJBase.Model`.

### Example

   mutable struct Foo
      x::Int
      y
   end

   @set_defaults Foo(1,2)
   
   julia> Foo()
   Foo(1, 2)

   julia> Foo(x=1, y="house")
   Foo(1, "house")

   @set_defaults Foo [4, 5]

   julia> Foo()
   Foo(4, 5)

"""
macro set_defaults(ex)
    T_ex = ex.args[1]
    value_exs = ex.args[2:end]
    values = [__module__.eval(ex) for ex in value_exs]
    set_defaults_(__module__, T_ex, values)
    return nothing
end

macro set_defaults(T_ex, values_ex)
    values =__module__.eval(values_ex)
    set_defaults_(__module__, T_ex, values)
    return nothing
end

function set_defaults_(mod, T_ex, values)
    T = mod.eval(T_ex)
    fields = fieldnames(T)
    length(fields) == length(values) ||
        error("Provide the same number of default values as fields. ")

    equality_pair_exs = [Expr(:kw, fields[i], values[i]) for i in
                         eachindex(values)]

    program = quote
        $T_ex(; $(equality_pair_exs...)) = 
            $T_ex($(fields...))
    end
    mod.eval(program)

end



