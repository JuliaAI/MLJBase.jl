using Documenter
using MLJBase

const REPO="github.com/JuliaAI/MLJBase.jl"

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
        "Utilities" => "utilities.md"
    ],
    repo="https://$REPO/blob/{commit}{path}#L{line}",
    sitename="MLJBase.jl"
)

deploydocs(; repo="$(REPO).git", push_preview=false)
