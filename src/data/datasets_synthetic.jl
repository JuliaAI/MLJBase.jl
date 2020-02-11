const EXTRA_KW_MAKE = """
	* `as_table=true`:  whether to return the points as a table (true) or a
						matrix (false). If true, the target vector is a
						categorical vector.
	* `eltype=Float64`:	to specify another type for the points, can be any
	 					subtype of AbstractFloat.
	* `rng=nothing`:    specify a number to make the points reproducible.
	"""


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
runif_ab(n, p, a, b)

Internal function to generate `n` points in `[a, b]ᵖ` uniformly at random.
"""
runif_ab(n, p, a, b) = (b - a) .* rand(n, p) .+ a


"""
make_blobs(n=100, p=2; kwargs...)

Generate gaussian blobs with `n` examples of `p` features. The function returns
a `n x p` matrix with the samples and a `n` integer vector indicating the
membership of each point.

## Keyword arguments

* `shuffle=true`:             whether to shuffle the resulting points,
* `centers=3`:                either a number of centers or a `c x p` matrix with `c` pre-determined centers,
* `cluster_std=1.0`:          the standard deviation(s) of each blob,
* `center_box=(-10. => 10.)`: the limits of the `p`-dimensional cube within which the cluster centers are drawn if they are not provided,
$EXTRA_KW_MAKE

## Example

```
X, y = make_blobs(100, 3; centers=2, cluster_std=[1.0, 3.0])
```
"""
function make_blobs(n::Integer=100, p::Integer=2; shuffle::Bool=true,
                    centers::Union{<:Integer,Matrix{<:Real}}=3,
                    cluster_std::Union{<:Real,Vector{<:Real}}=1.0,
                    center_box::Pair{<:Real,<:Real}=(-10.0 => 10.0),
                    as_table::Bool=true, eltype::Type{<:AbstractFloat}=Float64,
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
        centers   = runif_ab(n_centers, p, center_box...)
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

    # Seed the random number generator if required
    rng === nothing || Random.seed!(rng)

    # split points equally among centers
    ni, r    = divrem(n, n_centers)
    ns       = fill(ni, n_centers)
    ns[end] += r

    # vector of memberships
    y = vcat((fill(i, ni) for (i, ni) in enumerate(ns))...)

    # Pre-generate random points then modify for each center
    X    = randn(n, p)
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
make_circles(n=100; kwargs...)

Generate `n` points along two circumscribed circles returning the `n x 2`
matrix of points and a vector of membership (0, 1) depending on whether the points are on the smaller circle (0) or the larger one (1).

## Keyword arguments

* `shuffle=true`:   whether to shuffle the resulting points,
* `noise=0`:        standard deviation of the gaussian noise added to the data,
* `factor=0.8`:     ratio of the smaller radius over the larger one,
$EXTRA_KW_MAKE

## Example

```
X, y = make_circles(100; noise=0.5, factor=0.3)
```
"""
function make_circles(n::Integer=100; shuffle::Bool=true, noise::Real=0.,
                      factor::Real=0.8, as_table::Bool=true,
					  eltype::Type{<:AbstractFloat}=Float64, rng=nothing)
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

    rng === nothing || Random.seed!(rng)

    # Generate points on a 2D circle
    θs = runif_ab(n, 1, 0, 2pi)

    n0 = div(n, 2)

    X = hcat(cos.(θs), sin.(θs))
    X[1:n0, :] .*= factor

    y = ones(Int, n)
    y[1:n0] .= 0

    if !iszero(noise)
        X .+= noise .* randn(n, 2)
    end

	return finalize_Xy(X, y, shuffle, as_table, eltype, rng)
end


"""
make_moons(n::Int=100; kwargs...)

Generates `n` examples sampling from two interleaved half-circles returning
the `n x 2` matrix of points and a vector of membership (0, 1) depending on
whether the points are on the half-circle on the left (0) or on the right (1).

## Keyword arguments

* `shuffle=true`:   whether to shuffle the resulting points,
* `noise=0.1`:      standard deviation of the gaussian noise added to the data,
* `xshift=1.0`:     horizontal translation of the second center with respect to
                    the first one.
* `yshift=0.3`:     vertical translation of the second center with respect to
                    the first one.
$EXTRA_KW_MAKE

## Example

```
X, y = make_moons(100; noise=0.5)
```
"""
function make_moons(n::Int=150; shuffle::Bool=true, noise::Real=0.1,
                    xshift::Real=1.0, yshift::Real=0.3, as_table::Bool=true,
					eltype::Type{<:AbstractFloat}=Float64, rng=nothing)
    # check arguments make sense
    if n < 1
        throw(ArgumentError(
            "Expected `n` to be at least 1."))
    end
    if noise < 0
        throw(ArgumentError(
            "Noise argument cannot be negative."))
    end

    rng === nothing || Random.seed!(rng)

    n1 = div(n, 2)
    n2 = n - n1

    θs = runif_ab(n, 1, 0, pi)
    θs[n2+1:end] .*= -1

    X = hcat(cos.(θs), sin.(θs))

    X[n2+1:end, 1] .+= xshift
    X[n2+1:end, 2] .+= yshift

    y = ones(Int, n)
    y[1:n1] .= 0

    if !iszero(noise)
        X .+= noise .* randn(n, 2)
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
sparsify!(θ, s)

Make portion `s` of vector `θ` exactly 0.
"""
sparsify!(θ, s) = (θ .*= (rand(length(θ)) .< s))

"""Add outliers to portion s of vector."""
outlify!(y, s) = (n = length(y); y .+= 20 * randn(n) .* (rand(n) .< s))

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

## Keywords

* `intercept=true`:	whether to generate data from a model with intercept,
* `sparse=0`:		portion of the generating weight vector that is sparse,
* `noise=0.1`:		standard deviation of the gaussian noise added to the
 					response,
* `outliers=0`:		portion of the response vector to make as outliers by ading
					a random quantity with high variance. (Only applied if
					`binary` is `false`)
* `binary=false`:	whether the target should be binarized (via a sigmoid).
$EXTRA_KW_MAKE

## Example

```
X, y = make_regression(100, 5; noise=0.5, sparse=0.2, outliers=0.1)
```
"""
function make_regression(n::Int=100, p::Int=2; intercept::Bool=true,
						 sparse::Real=0, noise::Real=0.1, outliers::Real=0,
						 binary::Bool=false, as_table::Bool=true,
						 eltype::Type{<:AbstractFloat}=Float64, rng=nothing)
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

    rng === nothing || Random.seed!(rng)

    X = augment_X(randn(n, p), intercept)
    θ = randn(p + Int(intercept))
	sparse > 0 && sparsify!(θ, sparse)
	y = X * θ

	if !iszero(noise)
		y .+= noise .* randn(n)
	end

	if binary
		y = rand(n) .< sigmoid.(y)
	else
		if !iszero(outliers)
			outlify!(y, outliers)
		end
	end

	return finalize_Xy(X, y, false, as_table, eltype, rng; clf=binary)
end
