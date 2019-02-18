datadir = joinpath(srcdir, "../data/") # TODO: make OS agnostic

"""Load a well-known public regression dataset with nominal features."""
function load_boston()
    df = CSV.read(joinpath(datadir, "Boston.csv"),
                  categorical=true, allowmissing=:none)
    return SupervisedTask(data=df,
                          targets=:MedV,
                          ignore=[:Chas],
                          is_probabilistic=:yes,
                          input_kinds=:continuous,
                          output_kind=:continuous)
end

"""Load a reduced version of the well-known Ames Housing task,
having six numerical and six categorical features."""
function load_ames()
    df = CSV.read(joinpath(datadir, "reduced_ames.csv"), categorical=true,
                  allowmissing=:none)
    df[:target] = exp.(df[:target])
    return SupervisedTask(data=df,
                          targets=:target,
                          is_probabilistic=:no,
                          input_kinds=[:continuous, :multiclass],
                          output_kind=:continuous)
end

"""Load a well-known public classification task with nominal features."""
function load_iris()
    df = CSV.read(joinpath(datadir, "iris.csv"),
                  categorical=true, allowmissing=:none)
    return SupervisedTask(data=df,
                          targets=:target,
                          is_probabilistic=:yes,
                          input_kinds=:continuous,
                          output_kind=:multiclass)
end

"""Load a well-known crab classification dataset with nominal features."""
function load_crabs()
    df = CSV.read(joinpath(datadir, "crabs.csv"),
                  categorical=true, allowmissing=:none)
    return SupervisedTask(data=df,
                          targets=:sp,
                          ignore=[:sex, :index],
                          is_probabilistic=:yes,
                          input_kinds=:continuous,
                          output_kind=:multiclass)
end

"""Get some supervised data now!!"""
function datanow()
    Xtable, y = X_and_y(load_boston())
    X = DataFrame(Xtable)  # force table to be dataframe; should become redundant

    return (X[1:75,:], y[1:75])
end
