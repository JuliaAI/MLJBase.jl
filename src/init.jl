function __init__()
    global DEFAULT_RESOURCE = Ref{AbstractResource}(CPU1())

    # for testing asynchronous training of learing networks:
    global MACHINE_CHANNEL =
        RemoteChannel(() -> Channel(100), myid())

    TRAIT_FUNCTION_GIVEN_NAME[:measure]      = is_measure
    TRAIT_FUNCTION_GIVEN_NAME[:measure_type] = is_measure_type

    MLJModelInterface.set_interface_mode(MLJModelInterface.FullInterface())
end
