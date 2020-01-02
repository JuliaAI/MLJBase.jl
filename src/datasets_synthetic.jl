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
* `rng=nothing`:              specify a number to make the blobs reproducible.

## Example

```
X, y = make_blobs(100, 3; centers=2, cluster_std=[1.0, 3.0])
```
"""
function make_blobs(n::Integer=100, p::Integer=2; shuffle::Bool=true,
                    centers::Union{<:Integer,Matrix{<:Real}}=3,
                    cluster_std::Union{<:Real,Vector{<:Real}}=1.0,
                    center_box::Pair{<:Real,<:Real}=(-10.0 => 10.0),
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

    # Shuffle the rows if required
    shuffle && return shuffle_rows(X, y; rng=rng)

    return X, y
end


"""
make_circles(n=100; kwargs...)

Generate `n` points in two circumscribed circles returning the `n x 2` matrix
of points and a vector of membership (0, 1) depending on whether the points are inside (0) or outside (1) the smaller circle.

## Keyword arguments

* `shuffle=true`:   whether to shuffle the resulting points,
* `noise=0`:        standard deviation of the gaussian noise added to the data,
* `factor=0.8`:     ratio of the smaller radius over the larger one,
* `rng=nothing`:    specify a number to make the blobs reproducible.

"""
function make_circles(n::Integer=100; shuffle::Bool=true, noise::Real=0.,
                      factor::Real=0.8, rng=nothing)
    # NOTE: the implementation is a bit different than in
    # sklearn in that here, the points are generated on the
    # larger circle and membership is determined based  on the
    # radius. So if `factor` is close to 1 or close to 0 then
    # the dataset is imbalanced.
    #  -------------------------
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

    # Generate points in a 2D circle (not uniformly but we don't care)
    rs = runif_ab(n, 1, eps(), 1)
    θs = runif_ab(n, 1, 0, 2pi)
    xs = rs .* cos.(θs)
    ys = rs .* sin.(θs)
    X  = hcat(xs, ys)
    y  = Int.(rs .> factor)

    if !iszero(noise)
        X .+= noise .* randn(n, 2)
    end

    shuffle && return shuffle_rows(X, y; rng=rng)

    return X, y
end



"""
make_moons(n::Int=100; shuffle::Bool=true, noise::Number=0.,
           translation::Number=0.5, factor::Number=1.0, random_seed=Random.GLOBAL_RNG)

Generates `n` examples sampling from two moons. The `noise` can be changed to add
noise to the samples.
"""
function make_moons(n::Int=100; shuffle::Bool=true, noise::Number=0.,
                   translation::Number=0.5, factor::Number=1.0, random_seed=Random.GLOBAL_RNG)

    Random.seed!(random_seed)

    n_out = div(n, 2)
    n_in = n - n_out

    linrange_out = Array(LinRange(0, pi, n_out))
    linrange_in  = Array(LinRange(0, pi, n_in))

    outer_circ_x = cos.(linrange_out)
    outer_circ_y = sin.(linrange_out)
    inner_circ_x = 1 .- cos.(linrange_in) .* factor
    inner_circ_y = 1 .- sin.(linrange_in) .* factor .- translation

    X = hcat(vcat(outer_circ_x, inner_circ_x), vcat(outer_circ_y, inner_circ_y))
    y = vcat(ones(Int,n_out), 2*ones(Int,n_in))

    if shuffle
       X, y = shuffle_Xy(X, y ; random_seed=random_seed)
    end

    X .+= noise .* rand(n, 2)

    return X, y
end
