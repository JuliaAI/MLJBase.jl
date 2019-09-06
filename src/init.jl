using Requires

function __init__()
    @require(CSV="336ed68f-0bac-5ca0-87d4-7b16caf5d00b",
             include("datasets_requires.jl"))
end
