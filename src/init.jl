# using Requires

ScientificTypes.TRAIT_FUNCTION_GIVEN_NAME[:supervised_model] =
    x-> x isa Supervised
ScientificTypes.TRAIT_FUNCTION_GIVEN_NAME[:unsupervised_model] =
    x-> x isa Unsupervised
ScientificTypes.TRAIT_FUNCTION_GIVEN_NAME[:measure] =  is_measure

include("loss_functions_interface.jl")

# function __init__()
#     ScientificTypes.TRAIT_FUNCTION_GIVEN_NAME[:supervised_model] =
#         x-> x isa Supervised
#     ScientificTypes.TRAIT_FUNCTION_GIVEN_NAME[:unsupervised_model] =
#         x-> x isa Unsupervised
#     ScientificTypes.TRAIT_FUNCTION_GIVEN_NAME[:measure] =  is_measure
#     # @require(CSV="336ed68f-0bac-5ca0-87d4-7b16caf5d00b",
#     #          include("datasets_requires.jl"))
#     # @require(LossFunctions="30fc2ffe-d236-52d8-8643-a9d8f7c094a7",
#     #          include("loss_functions_interface.jl"))
# end
