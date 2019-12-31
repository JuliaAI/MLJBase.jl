module DatasetsSynthetic

using Random
export make_blobs, make_circles, make_moons

uniform_sample_in_zero_maxval(p, maxval) = maxval .* (1 .- rand(p))
uniform_sample_in_minval_maxval(p, minval, maxval) = (maxval-minval) .* rand(p) .+ minval
uniform_sample_in_minval_maxval(n, p, minval, maxval) = (maxval-minval) .* rand(n, p) .+ minval
normal_sample(p, mu, var) = mu .+ sqrt(var) .* randn(p)


"""
Shuffles the rows of an Array `X` and the values of a vector `y` using a randomly
generated permutation. The same permutation is used to shuffle both `X` and `y`.
"""
function shuffle_Xy(X, y; random_seed=Random.GLOBAL_RNG)
    Random.seed!(random_seed)
    perm = randperm(length(y))
    return X[perm,:], y[perm]
end


"""
make_blobs(n::Int=100;
           p::Int=2,
           centers::Int=3,
           cluster_std=1.0,
           center_box=(-10.,10.),
           element_type=Float64,
           random_seed=Random.GLOBAL_RNG,
           return_centers=false)

Generates a dataset with `n` examples of dimension `p` and returns a vector containing
as integers the membership of the different points generated.

The data is roughly grouped around several `centers`  which are created using `cluster_std`.
The data lives inside `center_box` in the case it is randomly generated.

- If `centers` is an integer the centroids are created randomly.
- If `centers` is an Array containing points the centroids are picked from `centers`.
- If `return_centers=true` the centroids of the blobs are returned.
"""
function make_blobs(n::Int=100; p::Int=2,
                    shuffle::Bool=false, centers::Int=3, cluster_std::Real=1.0, center_box=(-10.,10.),
                    element_type=Float64, random_seed=Random.GLOBAL_RNG, return_centers=false, verbose=0)

    Random.seed!(random_seed)
    X = zeros(n, p)
    y = zeros(n)

    if typeof(centers) <: Int
        n_centers = centers
        center_sample = uniform_sample_in_minval_maxval(n, p, center_box[1], center_box[2])
    else
        n_centers = length(centers)
    end

    if typeof(cluster_std) <: AbstractFloat
        cluster_std = cluster_std * randn(n_centers)
    end

    # generates div(n, n_centers) examples assigned to each center
    n_per_center = [div(n, n_centers) for x  in 1:n_centers]

    # adds the reamainding examples to each center up to n
    n_per_center = fill(div(n,n_centers), n_centers)
    n_per_center[end] += rem(n,n_centers)

    # generates the actual vectors close to each center blob
    start_ind = 1
    for (i, (n_blob, std, center)) in enumerate(zip(n_per_center, cluster_std, centers))
        ind_center = start_ind:(start_ind + n_per_center[i]-1)
        X[ind_center,:] .= center' .+ std .* randn(element_type, (n_per_center[i], p));
        y[ind_center] .= i
        if verbose>0
            println("center $i with $(n_per_center[i]) points created")
        end
        start_ind += n_per_center[i]
    end

    if shuffle
       X, y = shuffle_Xy(X, y ; random_seed=random_seed)
    end

    if return_centers
        return X, y, centers
    else
        return X, y
    end
end




"""
make_circles(n::Int=100; shuffle::Bool=true, noise::Number=0., random_seed=Random.GLOBAL_RNG, factor::Number=0.8)

Generates a dataset with `n` bi-dimensional examples. Samples are created
from two circles. One of the circles inside the other. The `noise` scalar
can be used to add noise to the generation process. The scalar `factor`
corresponds to the radius of the smallest circle.
"""
function make_circles(n::Int=100; shuffle::Bool=true, noise::Number=0., random_seed=Random.GLOBAL_RNG, factor::Number=0.8)
   
    Random.seed!(random_seed)

    n_out = div(n, 2)
    n_in = n - n_out

    linrange_out = Array(LinRange(0, 2 * pi, n_out))
    linrange_in  = Array(LinRange(0, 2 * pi, n_in))

    outer_circ_x = cos.(linrange_out)
    outer_circ_y = sin.(linrange_out)
    inner_circ_x = cos.(linrange_in) .* factor
    inner_circ_y = sin.(linrange_in) .* factor

    X = hcat(vcat(outer_circ_x, inner_circ_x), vcat(outer_circ_y, inner_circ_y))
    y = vcat(ones(Int,n_out), 2*ones(Int,n_in))

    if shuffle
       X, y = shuffle_Xy(X, y ; random_seed=random_seed)
    end

    X .+= noise .* rand(n, 2)

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

end #module
