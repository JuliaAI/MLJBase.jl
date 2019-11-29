
using Random

uniform_sample_in_zero_maxval(n_features, maxval) = maxval.*(1 .- rand(n_features)) 
uniform_sample_in_minval_maxval(n_features, minval, maxval) = (maxval-minval)* rand(n_features) .+ minval 
normal_sample(n_features, mu, var) = mu .+ sqrt(var) .* randn(n_features)

"""
    Return evenly spaced numbers over a specified interval.
    
    Returns `num` evenly spaced samples, calculated over the
    interval [`start`, `stop`].
    
    The endpoint of the interval can optionally be excluded.

"""
function linspace(start, stop; num=50, endpoint=true, retstep=false)
    dif  = stop-start
    step = dif/num
    result = []
       
    if endpoint 
        for i in 1:num
            push!(result, start+i*step)
        end
    else
        for i in 1:num
            push!(result, start+(i-1)*step)
        end
    end
    return result
    
end


"""
Shuffles the rows of an Array `X` and the values of a vector `y` using a randomly
generated permutation. The same permutation is used to shuffle both `X` and `y`.
"""
function shuffle_Xy(X, y; random_seed=1234)
    Random.seed!(random_seed)
    perm = randperm(length(y))
    return X[perm,:], y[perm]
end


"""
    make_blobs(n_examples=100; 
               n_features=2, 
               centers=3, 
               cluster_std=1.0, 
               center_box=(-10.,10.),
               element_type=Float64,
               random_seed=1234,
               return_centers=false)
               
     Generates a dataset with `n_examples` of dimension `n_features` and returns a vector containing
     as integers the membership of the different points generated.

     The data is roughly grouped around several `centers`  which are created using `cluster_std`. 
     The data lives inside `center_box` in the case it is randomly generated.
     
     - If `centers` is an integer the centroids are created randomly.    
     - If `centers` is an Array containing points the centroids are picked from `centers`.
     - If `return_centers=true` the centroids of the bloods are returned

"""
function make_blobs(n_examples=100; n_features=2, 
                    centers=3, cluster_std=1.0, center_box=(-10.,10.),
                    element_type=Float64, random_seed=1234, return_centers=false)

    Random.seed!(random_seed)

    X = [] 
    y = []

    if typeof(centers)<: Int
        n_centers = centers
        centers = []
        for c in 1:n_centers
            center_sample = uniform_sample_in_minval_maxval(n_features, center_box[1], center_box[2])
            push!(centers, center_sample)
        end
    else
        n_centers = length(centers)
    end

    if typeof(cluster_std)<: AbstractFloat
        cluster_std = 0.5 * randn(n_centers)
    end
    
    # generates the nunber of example for each center
    n_examples_per_center = [div(n_examples, n_centers) for x  in 1:n_centers]
    for i in 1:(n_examples % n_centers)
        n_examples_per_center[i] += 1
    end

    # generates the actual vectors close to each center blob 
    for (i, (n, std, center)) in enumerate(zip(n_examples_per_center, cluster_std, centers))
        X_current = center' .+ std .* randn(element_type, (n, n_features))
        push!(X, X_current)
        push!(y, [i for k in 1:n]) 
    end

    # stack all the previous arrays created for each of the centers
    X = cat(X..., dims=1)
    y = cat(y..., dims=1)
    
    if return_centers
        return X, y, centers  
    else
        return X, y
    end
end



"""
    make_circles(n=100; shuffle=true, noise=0., random_seed=1234, factor=0.8)
    
    Generates a dataset with `n` bi-dimensional examples. Samples are created 
    from two circles. One of the circles inside the other. The `noise` scalar
    can be used to add noise to the generation process. The scalar `factor` 
    can be used to make the radious of the smallest circle smaller.

"""
function make_circles(n=100; shuffle=true, noise=0., random_seed=1234, factor=0.8)
    
    @assert 0 <= factor <=1  #factor in [0,1] 

    n_out = div(n, 2)
    n_in = n - n_out
     
    linspace_out = linspace(0, 2 * pi; num=n_out, endpoint=false)
    linspace_in  = linspace(0, 2 * pi; num=n_in, endpoint=false)

    outer_circ_x = cos.(linspace_out)
    outer_circ_y = sin.(linspace_out)
    inner_circ_x = cos.(linspace_in) .* factor
    inner_circ_y = sin.(linspace_in) .* factor
    
    X = [[outer_circ_x..., inner_circ_x...] [outer_circ_y..., inner_circ_y...]]
    y = [ones(Int,n_out)..., 2*ones(Int,n_in)...]

    if shuffle
       X, y = shuffle_Xy(X, y ; random_seed=random_seed)
    end
    
    if noise >0.
        X .= X .+ noise.* rand(size(X)...)
    end

    return X,y
end


"""
    make_moons(n=100; shuffle=true, noise=0., translation=0.5, factor=1.0, random_seed=1234)
    
    Generates `n` examples sampling from two moons. The `noise` can be changed to add
    noise to the samples.

"""
function make_moons(n=100; shuffle=true, noise=0., translation=0.5, factor=1.0, random_seed=1234)
    
    @assert 0 <= factor <=1  

    n_out = div(n, 2)
    n_in = n - n_out
     
    linspace_out = linspace(0, pi; num=n_out, endpoint=true)
    linspace_in  = linspace(0, pi; num=n_in, endpoint=true)

    

    outer_circ_x = cos.(linspace_out)
    outer_circ_y = sin.(linspace_out)
    inner_circ_x = 1 .- cos.(linspace_in) .* factor
    inner_circ_y = 1 .- sin.(linspace_in) .* factor .- translation
    
    X = [[outer_circ_x..., inner_circ_x...] [outer_circ_y..., inner_circ_y...]]
    y = [ones(Int,n_out)..., 2*ones(Int,n_in)...]

    if shuffle
       X, y = shuffle_Xy(X, y ; random_seed=random_seed)
    end
    
    if noise >0.
        X .= X .+ noise.* rand(size(X)...)
    end

    return X,y
end
