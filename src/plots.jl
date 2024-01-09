@recipe function default_machine_plot(mach::Machine)
    # Allow downstream packages to define plotting recipes
    # for their own machine types.
    mach.fitresult
end
