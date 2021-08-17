function __init__()
    global DEFAULT_RESOURCE = Ref{AbstractResource}(CPU1())

    # for testing asynchronous training of learning networks:
    global TESTING = parse(Bool, get(ENV, "TEST_MLJBASE", "false"))
    if TESTING
        global MACHINE_CHANNEL =
            RemoteChannel(() -> Channel(100), myid())
    end

    MLJModelInterface.set_interface_mode(MLJModelInterface.FullInterface())
end
