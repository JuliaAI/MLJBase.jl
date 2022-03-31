function __init__()
    global HANDLE_GIVEN_ID = Dict{UInt64,Symbol}()
    global DEFAULT_RESOURCE = Ref{AbstractResource}(CPU1())
    global DEFAULT_SCITYPE_CHECK_LEVEL = Ref{Int}(1)
    global SHOW_COLOR = Ref{Bool}(true)

    # for testing asynchronous training of learning networks:
    global TESTING = parse(Bool, get(ENV, "TEST_MLJBASE", "false"))
    if TESTING
        global MACHINE_CHANNEL =
            RemoteChannel(() -> Channel(100), myid())
    end


    MLJModelInterface.set_interface_mode(MLJModelInterface.FullInterface())
end
