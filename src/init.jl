function __init__()
    global DEFAULT_RESOURCE = Ref{AbstractResource}(CPU1())

    TRAIT_FUNCTION_GIVEN_NAME[:measure]      = is_measure
    TRAIT_FUNCTION_GIVEN_NAME[:measure_type] = is_measure_type

    MLJModelInterface.set_interface_mode(MLJModelInterface.FullInterface())
end
