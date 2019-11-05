# using Requires

ScientificTypes.TRAIT_FUNCTION_GIVEN_NAME[:supervised_model] =
    x-> x isa Supervised
ScientificTypes.TRAIT_FUNCTION_GIVEN_NAME[:unsupervised_model] =
    x-> x isa Unsupervised
ScientificTypes.TRAIT_FUNCTION_GIVEN_NAME[:measure] =  is_measure

include("loss_functions_interface.jl")
