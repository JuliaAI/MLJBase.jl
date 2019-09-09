using Requires

function __init__()
    @require(CSV="336ed68f-0bac-5ca0-87d4-7b16caf5d00b",
             include("datasets_requires.jl"))
    @require(LossFunctions="30fc2ffe-d236-52d8-8643-a9d8f7c094a7",
             include("loss_functions_interface.jl"))
end
