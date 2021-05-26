function __init__()
    global DEFAULT_RESOURCE = Ref{AbstractResource}(CPU1())

    # for testing asynchronous training of learning networks:
    global TESTING = parse(Bool, get(ENV, "TEST_MLJBASE", "false"))
    if TESTING
        global MACHINE_CHANNEL =
            RemoteChannel(() -> Channel(100), myid())
    end

    TRAIT_FUNCTION_GIVEN_NAME[:measure]      = is_measure
    TRAIT_FUNCTION_GIVEN_NAME[:measure_type] = is_measure_type

    MLJModelInterface.set_interface_mode(MLJModelInterface.FullInterface())
end
