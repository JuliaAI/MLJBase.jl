# This defines a macro `mlj_model` which is a simpler version than the
# @sk_model macro defined to help import sklearn models.
# The difference is that the `mlj_model` macro only defines the constructor and the `clean!`
# and does not automatically define the `fit` and `predict` methods
#
# NOTE: it does NOT handle parametric types (yet).

"""
_process_model_def(ex)

Take an expression defining a model (`mutable struct Model ...`) and unpack key elements for
further processing:

- Model name (`modelname`)
- Names of parameters (`params`)
- Default values (`defaults`)
- Constraints (`constraints`)
"""
function _process_model_def(ex)
    defaults    = Dict{Symbol,Any}()
    constraints = Dict{Symbol,Any}()
    modelname   = ex.args[2] isa Symbol ? ex.args[2] : ex.args[2].args[1]
    params      = Symbol[]

    # inspect all lines which may define parameters, retrieve their names,
    # default values and constraints on values that can be given to them
    for i in 1:length(ex.args[3].args)
        # retrieve meaningful lines
        line = ex.args[3].args[i]
        line isa LineNumberNode && continue

        # line without information (e.g. just a name "a")
        if line isa Symbol
            param = line
            push!(params, param)
            defaults[param] = missing
        else
            # A meaningful line will look like
            #   line.args[1] = line.args[2]
            #
            # where line.args[1] will either be just `name`  or `name::Type`
            # and   line.args[2] will either be just `value` or `value::constraint`
            # ---------------------------------------------------------
            # 1. decompose `line.args[1]` appropriately (name and type)
            if line.args[1] isa Symbol # case :a
                param = line.args[1]
                type  = length(line.args) > 1 ? line.args[2] : :Any
            else                       # case :(a::Int)
                param, type = line.args[1].args[1:2] # (:a, Int)
            end
            push!(params, param)
            # ------------------------------------------------------------------
            # 2. decompose `line.args[2]` appropriately (values and constraints)
            if line.head == :(=) # assignment for default
                default = line.args[2]
                # if a constraint is given (value::constraint)
                if default isa Expr && length(default.args) > 1
                    constraints[param] = default.args[2]
                    # now discard the constraint to keep only the value
                    default = default.args[1]
                end
                defaults[param]    = default       # this will be a value not an expr
                ex.args[3].args[i] = line.args[1]  # name or name::Type (for the constructor)
            else
                # these are simple heuristics when no default value is given for the
                # field but an "obvious" one can be provided implicitly (ideally this should
                # not be used as it's not very clear that the intention matches the usage)
                eff_type = eval(type)
                if eff_type <: Number
                    defaults[param] = zero(eff_type)
                elseif eff_type <: AbstractString
                    defaults[param] = ""
                elseif eff_type == Any         # e.g. Any or no type given
                    defaults[param] = missing
                elseif eff_type >: Nothing     # e.g. Union{Nothing, ...}
                    defaults[param] = nothing
                elseif eff_type >: Missing     # e.g. Union{Missing, ...} (unlikely)
                    defaults[param] = missing
                else
                    @error "A default value for parameter '$param' (type '$type') must be given"
                end
            end
        end
    end
    return ex, modelname, params, defaults, constraints
end


"""
_unpack!(ex, rep)

Internal function to allow to read a constraint given after a default value for a parameter
and transform it in an executable condition (which is returned to be executed later).
For instance if we have

    alpha::Int = 0.5::(arg > 0.0)

Then it would transform the `(arg > 0.0)` in `(alpha > 0.0)` which is executable.
"""
function _unpack!(ex::Expr, rep)
    for i in eachindex(ex.args)
        if ex.args[i] ∈ (:_, :arg)
            ex.args[i] = rep
        end
        _unpack!(ex.args[i], rep)
    end
    return ex
end
_unpack!(ex, _) = ex # when it's been unpacked, it's not an expression anymore


"""
_model_constructor(modelname, params, defaults)

Build the expression of the keyword constructor associated with a model definition.
When the constructor is called, the `clean!` function is called as well to check that
parameter assignments are valid.
"""
function _model_constructor(modelname, params, defaults)
    Expr(:function, Expr(:call, modelname, Expr(:parameters, (Expr(:kw, p, defaults[p]) for p in params)...)),
        # body of the function
        Expr(:block,
             Expr(:(=), :model, Expr(:call, :new, params...)),
             :(message = clean!(model)),
			 :(isempty(message) || @warn message),
			 :(return model)
			 )
	 	)
end


"""
_model_cleaner(modelname, defaults, constraints)

Build the expression of the cleaner associated with the constraints specified in a model def.
"""
function _model_cleaner(modelname, defaults, constraints)
    Expr(:function, :(clean!(model::$modelname)),
        # body of the function
        Expr(:block,
             :(warning = ""),
             # condition and action for each constraint
             # each parameter is given as field::Type = default::constraint
             # here we recuperate the constraint and express it as an if statement
             # for instance if we had
             #     alpha::Real = 0.0::(arg > 0.0)
             # this would become
             #     if !(alpha > 0.0)
             (Expr(:if, Expr(:call, :!, _unpack!(constr, :(model.$param))),
                   # action of the constraint is violated:
                   # add a message and use default for the parameter
                   Expr(:block,
                        :(warning *= $("Constraint `$constr` failed; using default: $param=$(defaults[param]).")),
                        :(model.$param = $(defaults[param]))
                        )
                   ) for (param, constr) in constraints)...,
             # return full message
             :(return warning)
            )
        )
end

"""
mlj_model

Macro to help define MLJ models with constraints on the default parameters, this can be seen as
a tweaked version of the `@with_kw` macro from `Parameters`.
"""
macro mlj_model(ex)
    ex, modelname, params, defaults, constraints = _process_model_def(ex)
	# keyword constructor
    const_ex = _model_constructor(modelname, params, defaults)
	# associate the constructor with the definition of the struct
    push!(ex.args[3].args, const_ex)
	# cleaner
    clean_ex = _model_cleaner(modelname, defaults, constraints)
    esc(
        quote
            Base.@__doc__ $ex
            export $modelname
            $ex
            $clean_ex
        end
    )
end
