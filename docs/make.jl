using Documenter
using MLJBase

const REPO="github.com/JuliaAI/MLJBase.jl.git"

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
    repo="$REPO/blob/{commit}{path}#L{line}",
    sitename="MLJBase.jl"
)

deploydocs(; repo=REPO, push_preview=false)
