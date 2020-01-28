function __init__()
    global DEFAULT_RESOURCE = Ref{AbstractResource}(CPU1())

    ST.TRAIT_FUNCTION_GIVEN_NAME[:measure] = is_measure
    ST.TRAIT_FUNCTION_GIVEN_NAME[:measure_type] = is_measure_type

    MMI.set_interface_mode(MMI.FullInterface())
end
