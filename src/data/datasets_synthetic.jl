const EXTRA_KW_MAKE = """

* `eltype=Float64`: machine type of points (any subtype of
   `AbstractFloat`).

* `rng=nothing`: any `AbstractRNG` object, or integer to seed a
  `MersenneTwister` (for reproducibility).

* `as_table=true`: whether to return the points as a table (true)
  or a matrix (false). """

const EXTRA_CLASSIFICATION =
    "If `false` the target `y` has integer element type. "

"""
    finalize_Xy(X, y, shuffle, as_table, eltype, rng; clf)

Internal function to  finalize the `make_*` functions.

"""
function finalize_Xy(X, y, shuffle, as_table, eltype, rng; clf::Bool=true)
    # Shuffle the rows if required
    if shuffle
                X, y = shuffle_rows(X, y; rng=rng)
        end
        if eltype != Float64
                X = convert.(eltype, X)
        end
        # return as matrix if as_table=false
        as_table || return X, y
        clf && return MLJBase.table(X), categorical(y)
        return MLJBase.table(X), y
end

### CLASSIFICATION TOY DATASETS

"""
    runif_ab(rng, n, p, a, b)

Internal function to generate `n` points in `[a, b]ᵖ` uniformly at random.

"""
runif_ab(rng, n, p, a, b) = (b - a) .* rand(rng, n, p) .+ a

"""
    X, y = make_blobs(n=100, p=2; kwargs...)

Generate Gaussian blobs for clustering and classification
problems.

### Return value

By default, a table `X` with `p` columns (features) and `n` rows
(observations), together with a corresponding vector of `n`
`Multiclass` target observations `y`, indicating blob membership.

### Keyword arguments

* `shuffle=true`: whether to shuffle the resulting points,

* `centers=3`: either a number of centers or a `c x p` matrix with `c`
  pre-determined centers,

* `cluster_std=1.0`: the standard deviation(s) of each blob,

* `center_box=(-10. => 10.)`: the limits of the `p`-dimensional cube
  within which the cluster centers are drawn if they are not provided,
  $(EXTRA_KW_MAKE*EXTRA_CLASSIFICATION)

### Example

```
X, y = make_blobs(100, 3; centers=2, cluster_std=[1.0, 3.0])
```

"""
function make_blobs(n::Integer=100,
                    p::Integer=2;
                    shuffle::Bool=true,
                    centers::Union{<:Integer,Matrix{<:Real}}=3,
                    cluster_std::Union{<:Real,Vector{<:Real}}=1.0,
                    center_box::Pair{<:Real,<:Real}=(-10.0 => 10.0),
                    as_table::Bool=true,
                    eltype::Type{<:AbstractFloat}=Float64,
                    rng=nothing)

    # check arguments make sense
    if n < 1 || p < 1
        throw(ArgumentError(
            "Expected `n` and `p` to be at least 1."))
    end
    if center_box.first >= center_box.second
        throw(ArgumentError(
            "Domain for the centers improperly defined expected a pair " *
            "`a => b` with `a < b`."))
    end

    if rng === nothing
        rng = Random.GLOBAL_RNG
    elseif rng isa Integer
        rng = Random.MersenneTwister(rng)
    end

    if centers isa Matrix
        if size(centers, 2) != p
            throw(ArgumentError(
                "The centers provided have dimension ($(size(centers, 2))) " *
                "that doesn't match the one specified ($(p))."))
        end
        n_centers = size(centers, 1)
    else
        # in case the centers aren't provided, draw them from the box
        n_centers = centers
        centers   = runif_ab(rng, n_centers, p, center_box...)
    end
    if cluster_std isa Vector
        if length(cluster_std) != n_centers
            throw(ArgumentError(
                "$(length(cluster_std)) standard deviations given but there " *
                "are $(n_centers) centers."))
        end
        if any(cluster_std .<= 0)
            throw(ArgumentError(
                "Cluster(s) standard deviation(s) must be positive."))
        end
    else
        # In case only one std is given, repeat it for each center
        cluster_std = fill(cluster_std, n_centers)
    end

    # split points equally among centers
    ni, r    = divrem(n, n_centers)
    ns       = fill(ni, n_centers)
    ns[end] += r

    # vector of memberships
    y = vcat((fill(i, ni) for (i, ni) in enumerate(ns))...)

    # Pre-generate random points then modify for each center
    X    = randn(rng, n, p)
    nss  = [1, cumsum(ns)...]
    # ranges of rows for each center
    rows = [nss[i]:nss[i+1] for i in 1:n_centers]
    @inbounds for c in 1:n_centers
        Xc = view(X, rows[c], :)
        # adjust standard deviation
        Xc .*= cluster_std[c]
        # adjust center
        Xc .+= centers[c, :]'
    end

    return finalize_Xy(X, y, shuffle, as_table, eltype, rng)
end


"""
    X, y = make_circles(n=100; kwargs...)

Generate `n` labeled points close to two concentric circles for
classification and clustering models.

### Return value

By default, a table `X` with `2` columns and `n` rows (observations),
together with a corresponding vector of `n` `Multiclass` target
observations `y`. The target is either `0` or `1`, corresponding to
membership to the smaller or larger circle, respectively.

### Keyword arguments

* `shuffle=true`: whether to shuffle the resulting points,

* `noise=0`: standard deviation of the Gaussian noise added to the data,

* `factor=0.8`: ratio of the smaller radius over the larger one,

$(EXTRA_KW_MAKE*EXTRA_CLASSIFICATION)

### Example

```
X, y = make_circles(100; noise=0.5, factor=0.3)
```

"""
function make_circles(n::Integer=100;
                      shuffle::Bool=true,
                      noise::Real=0.,
                      factor::Real=0.8,
                      as_table::Bool=true,
                      eltype::Type{<:AbstractFloat}=Float64,
                      rng=nothing)

    # check arguments make sense
    if n < 1
        throw(ArgumentError(
            "Expected `n` to be at least 1."))
    end
    if noise < 0
        throw(ArgumentError(
            "Noise argument cannot be negative."))
    end
    if !(0 < factor < 1)
        throw(ArgumentError(
            "Factor argument must be strictly between 0 and 1."))
    end

        if rng === nothing
                rng = Random.GLOBAL_RNG
        elseif rng isa Integer
                rng = Random.MersenneTwister(rng)
        end

    # Generate points on a 2D circle
    θs = runif_ab(rng, n, 1, 0, 2pi)

    n0 = div(n, 2)

    X = hcat(cos.(θs), sin.(θs))
    X[1:n0, :] .*= factor

    y = ones(Int, n)
    y[1:n0] .= 0

    if !iszero(noise)
        X .+= noise .* randn(rng, n, 2)
    end

        return finalize_Xy(X, y, shuffle, as_table, eltype, rng)
end


"""
        make_moons(n::Int=100; kwargs...)

Generates labeled two-dimensional points lying close to two
interleaved semi-circles, for use with classification and clustering
models.

### Return value

By default, a table `X` with `2` columns and `n` rows (observations),
together with a corresponding vector of `n` `Multiclass` target
observations `y`. The target is either `0` or `1`, corresponding to
membership to the left or right semi-circle.


### Keyword arguments

* `shuffle=true`: whether to shuffle the resulting points,

* `noise=0.1`: standard deviation of the Gaussian noise added to the data,

* `xshift=1.0`: horizontal translation of the second center with respect to
  the first one.

* `yshift=0.3`: vertical translation of the second center with respect
  to the first one.  $(EXTRA_KW_MAKE*EXTRA_CLASSIFICATION)

### Example

```
X, y = make_moons(100; noise=0.5)
```

"""
function make_moons(n::Int=150;
                    shuffle::Bool=true,
                    noise::Real=0.1,
                    xshift::Real=1.0,
                    yshift::Real=0.3,
                    as_table::Bool=true,
                    eltype::Type{<:AbstractFloat}=Float64,
                    rng=nothing)

    # check arguments make sense
    if n < 1
        throw(ArgumentError(
            "Expected `n` to be at least 1."))
    end
    if noise < 0
        throw(ArgumentError(
            "Noise argument cannot be negative."))
    end

        if rng === nothing
                rng = Random.GLOBAL_RNG
        elseif rng isa Integer
                rng = Random.MersenneTwister(rng)
        end

    n1 = div(n, 2)
    n2 = n - n1

    θs = runif_ab(rng, n, 1, 0, pi)
    θs[n2+1:end] .*= -1

    X = hcat(cos.(θs), sin.(θs))

    X[n2+1:end, 1] .+= xshift
    X[n2+1:end, 2] .+= yshift

    y = ones(Int, n)
    y[1:n1] .= 0

    if !iszero(noise)
        X .+= noise .* randn(rng, n, 2)
    end

    return finalize_Xy(X, y, shuffle, as_table, eltype, rng)
end


### REGRESSION TOY DATASETS

"""
    augment_X(X, fit_intercept)

Given a matrix `X`, append a column of ones if `fit_intercept` is true.
See [`make_regression`](@ref).

"""
function augment_X(X::Matrix{<:Real}, fit_intercept::Bool)
        fit_intercept || return X
        return hcat(X, ones(eltype(X), size(X, 1)))
end

"""
    sparsify!(rng, θ, s)

Make portion `s` of vector `θ` exactly 0.

"""
sparsify!(rng, θ, s) =
        (θ .*= (rand(rng, length(θ)) .< s))

"""Add outliers to portion s of vector."""
outlify!(rng, y, s) =
        (n = length(y); y .+= 20 * randn(rng, n) .* (rand(rng, n) .< s))

const SIGMOID_64 = log(Float64(1)/eps(Float64) - Float64(1))
const SIGMOID_32 = log(Float32(1)/eps(Float32) - Float32(1))

"""
    sigmoid(x)

Return the sigmoid computed in a numerically stable way:

``σ(x) = 1/(1+exp(-x))``

"""
function sigmoid(x::Float64)
        x > SIGMOID_64  && return one(x)
        x < -SIGMOID_64 && return zero(x)
        return one(x) / (one(x) + exp(-x))
end
function sigmoid(x::Float32)
        x > SIGMOID_32  && return one(x)
        x < -SIGMOID_32 && return zero(x)
        return one(x) / (one(x) + exp(-x))
end
sigmoid(x) = sigmoid(float(x))


"""
    make_regression(n, p; kwargs...)

Generate Gaussian input features and a linear response with Gaussian
noise, for use with regression models.

### Return value

By default, a table `X` with `p` columns and `n` rows (observations),
together with a corresponding vector of `n` `Continuous` target
observations `y`.

### Keywords

* `intercept=true: whether to generate data from a model with
  intercept,

* `sparse=0`: portion of the generating weight vector that is sparse,

* `noise=0.1`: standard deviation of the Gaussian noise added to the
  response,

* `outliers=0`: portion of the response vector to make as outliers by
  adding a random quantity with high variance. (Only applied if
  `binary` is `false`)

* `binary=false`: whether the target should be binarized (via a sigmoid).
$EXTRA_KW_MAKE

### Example

```
X, y = make_regression(100, 5; noise=0.5, sparse=0.2, outliers=0.1)
```

"""
function make_regression(n::Int=100,
                         p::Int=2;
                         intercept::Bool=true,
                         sparse::Real=0,
                         noise::Real=0.1,
                         outliers::Real=0,
                         binary::Bool=false,
                         as_table::Bool=true,
                         eltype::Type{<:AbstractFloat}=Float64,
                         rng=nothing)

    # check arguments make sense
    if n < 1 || p < 1
        throw(ArgumentError(
            "Expected `n` and `p` to be at least 1."))
    end
    if !(0 <= sparse < 1)
        throw(ArgumentError(
            "Sparsity argument must be in [0, 1)."))
    end
    if noise < 0
        throw(ArgumentError(
            "Noise argument cannot be negative."))
    end
    if !(0 <= outliers <= 1)
        throw(ArgumentError(
            "Outliers argument must be in [0, 1]."))
    end

    if rng === nothing
        rng = Random.GLOBAL_RNG
    elseif rng isa Integer
        rng = Random.MersenneTwister(rng)
    end

    X = augment_X(randn(rng, n, p), intercept)
    θ = randn(rng, p + Int(intercept))
    sparse > 0 && sparsify!(rng, θ, sparse)
    y = X * θ

    if !iszero(noise)
        y .+= noise .* randn(rng, n)
    end

    if binary
        y = rand(rng, n) .< sigmoid.(y)
    else
        if !iszero(outliers)
            outlify!(rng, y, outliers)
        end
    end

    return finalize_Xy(X[:,1:end-Int(intercept)], y, false,
                       as_table, eltype, rng; clf=binary)
end
