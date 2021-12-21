module TestUtilities

using Test
using MLJBase

export @testset_accelerated, include_everywhere, @test_mach_sequence,
    @test_model_sequence

using ComputationalResources
using ComputationalResources: CPUProcesses

macro testset_accelerated(name::String, var, ex)
    testset_accelerated(name, var, ex)
end
macro testset_accelerated(name::String, var, opts::Expr, ex)
    testset_accelerated(name, var, ex; eval(opts)...)
end
function testset_accelerated(name::String, var, ex; exclude=[])
    final_ex = quote
       local $var = CPU1()
        @testset $name $ex
    end

    resources = AbstractResource[CPUProcesses(), CPUThreads()]

    for res in resources
        if any(x->typeof(res)<:x, exclude)
            push!(final_ex.args, quote
               local $var = $res
               @testset $(name*" ($(typeof(res).name))") begin
                   @test_broken false
               end
            end)
        else
            push!(final_ex.args, quote
               local $var = $res
               @testset $(name*" ($(typeof(res).name))") $ex
            end)
        end
    end
    # preserve outer location if possible
    if ex isa Expr && ex.head === :block && !isempty(ex.args) && ex.args[1] isa LineNumberNode
        final_ex = Expr(:block, ex.args[1], final_ex)
    end
    return esc(final_ex)
end

function sedate!(fit_ex)
    kwarg_exs = filter(fit_ex.args) do arg
        arg isa Expr && arg.head == :kw
    end
    keys = map(kwarg_exs) do arg
        arg.args[1]
    end
    :verbosity in keys &&
        error("You cannot specify `verbosity` in @test_mach_sequence "*
              "or @test_model_sequence. ")
    push!(fit_ex.args, Expr(:kw, :verbosity, -5000))
    return fit_ex
end

macro test_mach_sequence(fit_ex, sequence_exs...)
    sedate!(fit_ex)
    seq = gensym(:sequence)
    esc(quote
        MLJBase.flush!(MLJBase.MACHINE_CHANNEL)
        $fit_ex
        local $seq = MLJBase.flush!(MLJBase.MACHINE_CHANNEL)
        # for s in $seq
        #     println(s)
        # end
        @test $seq in [$(sequence_exs...)]
    end)
end

# function weakly_in(object:Tuple{Symbol,Model}, itr)
#     for tup in itr
#         tup[1] === object[1] && tup[2] == tup

macro test_model_sequence(fit_ex, sequence_exs...)
    sedate!(fit_ex)
    seq = gensym(:sequence)
    esc(quote
        MLJBase.flush!(MLJBase.MACHINE_CHANNEL)
        $fit_ex
        local $seq = map(MLJBase.flush!(MLJBase.MACHINE_CHANNEL)) do tup
            (tup[1], tup[2].model)
        end
        # for s in $seq
        #     println(s)
        # end
        @test $seq in [$(sequence_exs...)]
    end)
end

end
