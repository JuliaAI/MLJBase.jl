@test default_resource() == CPU1()

default_resource(ComputationalResources.CPUProcesses())
@test default_resource() == ComputationalResources.CPUProcesses()

true
