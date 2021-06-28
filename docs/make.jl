using Documenter
using MLJBase

makedocs(;
    modules=[MLJBase],
    format=Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    pages=[
        "Home" => "index.md",
        "Measures" => "measures.md",
        "Resampling" => "resampling.md",
        "Composition" => "composition.md",
        "Datasets" => "datasets.md",
        "Distributions" => "distributions.md",
        "OpenML" => "openml.md",
        "Utilities" => "utilities.md"
    ],
    repo="https://github.com/JuliaAI/MLJBase.jl/blob/{commit}{path}#L{line}",
    sitename="MLJBase.jl"
)

# By default Documenter does not deploy docs just for PR
# this causes issues with how we're doing things and ends
# up choking the deployment of the docs, so  here we
# force the environment to ignore this so that Documenter
# does indeed deploy the docs
ENV["TRAVIS_PULL_REQUEST"] = "false"

deploydocs(;
           repo="github.com/alan-turing-institute/MLJBase.jl.git",
           push_preview=false,
)
