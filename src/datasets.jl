datadir = joinpath(srcdir, "..", "data") # TODO: make OS agnostic

"""Load a well-known public regression dataset with nominal features."""
function load_boston()
    df = CSV.read(joinpath(datadir, "Boston.csv"), categorical=true)
    return SupervisedTask(verbosity=0, data=df,
                          target=:MedV,
                          ignore=[:Chas,],
                          is_probabilistic=false)
end

"""Load a reduced version of the well-known Ames Housing task,
having six numerical and six categorical features."""
function load_reduced_ames()
    df = CSV.read(joinpath(datadir, "reduced_ames.csv"), categorical=true)
    df[:target] = exp.(df[:target])
    # TODO: uncomment following after julia #29501 is resolved
#    df.OverallQual = categorical(df.OverallQual, ordered=true)
#    df[:GarageCars] = categorical(df[:GarageCars], ordered=true)
#    df[:YearBuilt] = categorical(df[:YearBuilt], ordered=true)
#    df[:YearRemodAdd] = categorical(df[:YearRemodAdd], ordered=true)
    return SupervisedTask(verbosity=0, data=df,
                          target=:target,
                          is_probabilistic=false)
end

"""Load the full version of the well-known Ames Housing task."""
function load_ames()
    df = CSV.read(joinpath(datadir, "ames.csv"), categorical=true)              
    df[:target] = exp.(df[:target])
    return SupervisedTask(verbosity=0, data=df,
                          target=:target,
                          ignore=[:Id,],
                          is_probabilistic=false)
end

"""Load a well-known public classification task with nominal features."""
function load_iris()
    df = CSV.read(joinpath(datadir, "iris.csv"), categorical=true)
    return SupervisedTask(verbosity=0, data=df,
                          target=:target,
                          is_probabilistic=false)
end

"""Load a well-known crab classification dataset with nominal features."""
function load_crabs()
    df = CSV.read(joinpath(datadir, "crabs.csv"), categorical=true)
    return SupervisedTask(verbosity=0, data=df,
                          target=:sp,
                          ignore=[:sex, :index],
                          is_probabilistic=true)
end

"""Get some supervised data now!!"""
function datanow()
    Xtable, y = X_and_y(load_boston())
    X = DataFrame(Xtable)  # force table to be dataframe; should become redundant

    return (X[1:75,:], y[1:75])
end
