# implementation of MLJ measure interface for LossFunctions.jl

function naked(T::Type)
    without_module_name = split(string(T), '.') |> last
    without_type_parameters = split(without_module_name, '{') |> first
    return Symbol(without_type_parameters)
end

const WITHOUT_PARAMETERS =
    setdiff(LOSS_FUNCTIONS, WITH_PARAMETERS)

## WRAPPER

abstract type SupervisedLoss <: Unaggregated end


struct MarginLoss{L<:LossFunctions.MarginLoss} <: SupervisedLoss
    loss::L
end

struct DistanceLoss{L<:LossFunctions.DistanceLoss} <: SupervisedLoss
    loss::L
end

# INTERFACE FOR EXTRACTING PARAMETERS

# LossFunctions.jl does not have a uniform interface for extacting
# parameters, and hence:

_parameter(loss::LossFunctions.DWDMarginLoss) = loss.q
_parameter(loss::LossFunctions.SmoothedL1HingeLoss) = loss.gamma
_parameter(loss::LossFunctions.HuberLoss) = loss.d
_parameter(loss::LossFunctions.L1EpsilonInsLoss) = loss.ε
_parameter(loss::LossFunctions.L2EpsilonInsLoss) = loss.ε
_parameter(::LossFunctions.LPDistLoss{P}) where P = P
_parameter(::LossFunctions.L1DistLoss) = 1
_parameter(::LossFunctions.L2DistLoss) = 2
_parameter(loss::LossFunctions.QuantileLoss) = loss.τ


## CONSTRUCTORS AND CALLING BEHAVIOUR

err_wrap(n) = ArgumentError("Bad @wrap syntax: $n. ")

# We define amacro to wrap a concrete `LossFunctions.SupervisedLoss`
# type and define its constructor, and to define property access in
# case of parameters; the macro also defines calling behaviour:
macro wrap_loss(ex)
    ex.head == :call || throw(err_wrap(1))
    Loss_ex = ex.args[1]
    Loss_str = string(Loss_ex)
    if Loss_ex in MARGIN_LOSSES
        T = :MarginLoss
    else
        T = :DistanceLoss
    end

    # bind name to wrapped version of LossFunctions loss:
    program = quote
        const $Loss_ex = $T{<:LossFunctions.$Loss_ex}
        name(M::Type{<:$Loss_ex}) = $Loss_str
    end

    # defined instances
    alias = snakecase(string(Loss_ex))
    push!(program.args, quote
          instances(::Type{<:$Loss_ex}) = [$alias, ]
          end)

    # define kw constructor and expose any parameter as a property:
    if length(ex.args) == 1
        push!(program.args, quote
              $Loss_ex() = $T(LossFunctions.$Loss_ex())
              Base.propertynames(::$Loss_ex) = ()
              end)
    elseif length(ex.args) > 1
        sub_ex = ex.args[2]
        sub_ex.head == :parameters || throw(err_wrap(2))
        length(sub_ex.args) == 1 || throw(err_wrap("Only 1 kwarg supported"))
        sub_ex.args[1].head == :kw || throw(err_wrap(3))
        var_ex = sub_ex.args[1].args[1]
        var_str = string(var_ex)
        val_ex = sub_ex.args[1].args[2]
        push!(program.args, quote
              $Loss_ex(; $var_ex=$val_ex) =
                  $T(LossFunctions.$Loss_ex($var_ex))
              $Loss_ex(p) = $Loss_ex($var_ex=p)
              Base.propertynames(::$Loss_ex) = (Symbol($var_str), )
              function Base.getproperty(wrapper::$Loss_ex, name::Symbol)
                  if name === Symbol($var_str)
                      return _parameter(getfield(wrapper, :loss)) # see below
                  end
                  error("type $($Loss_ex) has no property $name")
              end
              end)
    else
        throw(err_wrap(4))
    end

    esc(program)
end

for Loss in WITHOUT_PARAMETERS
    eval(:(@wrap_loss $Loss()))
end

@wrap_loss DWDMarginLoss(; q=1.0)
@wrap_loss SmoothedL1HingeLoss(; gamma=1.0)
@wrap_loss HuberLoss(; d=1.0)
@wrap_loss L1EpsilonInsLoss(; ε=1.0)
@wrap_loss L2EpsilonInsLoss(; ε=1.0)
@wrap_loss LPDistLoss(; P=2)
@wrap_loss QuantileLoss(; τ=0.7)


## GENERIC TRAITS

const LossFunctions = LossFunctions
is_measure_type(::Type{<:SupervisedLoss})          = true
orientation(::Type{<:SupervisedLoss})              = :loss
reports_each_observation(::Type{<:SupervisedLoss}) = true
is_feature_dependent(::Type{<:SupervisedLoss})     = false
supports_weights(::Type{<:SupervisedLoss}) = true
docstring(M::Type{<:SupervisedLoss})       = name(M)


## CALLING - DISTANCE BASED LOSS FUNCTIONS

MMI.prediction_type(::Type{<:DistanceLoss}) = :deterministic
MMI.target_scitype(::Type{<:DistanceLoss}) = Union{Vec{Continuous},Vec{Count}}

call(measure::DistanceLoss, yhat, y) =
    (getfield(measure, :loss)).(yhat, y)

function call(measure::DistanceLoss, yhat, y, w::AbstractArray)
    return w .* call(measure, yhat, y)
end


## CALLING - MARGIN BASED LOSS FUNCTIONS

MMI.prediction_type(::Type{<:MarginLoss}) = :probabilistic
MMI.target_scitype(::Type{<:MarginLoss})  = AbstractArray{<:Finite{2}}

# rescale [0, 1] -> [-1, 1]:
_scale(p) = 2p - 1

function call(measure::MarginLoss, yhat, y)
    probs_of_observed = broadcast(pdf, yhat, y)
    loss = getfield(measure, :loss)
    return loss.(_scale.(probs_of_observed), 1)
end

call(measure::MarginLoss, yhat, y, w::AbstractArray) =
    w .* call(measure, yhat, y)


## ADJUSTMENTS

human_name(::Type{<:L1EpsilonInsLoss}) = "l1 ϵ-insensitive loss"
human_name(::Type{<:L2EpsilonInsLoss}) = "l2 ϵ-insensitive loss"
human_name(::Type{<:DWDMarginLoss}) = "distance weighted discrimination loss"

_signature(::Any) = ""
_signature(::Type{<:HuberLoss}) = "`HuberLoss(; d=1.0)`"
_signature(::Type{<:DWDMarginLoss}) = "`DWDMarginLoss(; q=1.0)`"
_signature(::Type{<:SmoothedL1HingeLoss}) = "`SmoothedL1HingeLoss(; gamma=1.0)`"
_signature(::Type{<:L1EpsilonInsLoss}) = "`L1EpsilonInsLoss(; ε=1.0)`"
_signature(::Type{<:L2EpsilonInsLoss}) = "`L2EpsilonInsLoss(; ε=1.0)`"
_signature(::Type{<:LPDistLoss}) = "`LPDistLoss(; P=2)`"
_signature(::Type{<:QuantileLoss}) = "`QuantileLoss(; τ=0.7)`"


## ALIASES AND DOCSTRINGS

const DOC_LOSS_FUNCTIONS =
"""
For more detail, see the original LossFunctions.jl documentation *but
note differences in the signature.*

Losses from LossFunctions.jl do not support `missing` values. To use
with `missing` values, replace `(ŷ, y)` with `skipinvalid(ŷ, y))`.
"""

for Loss_ex in DISTANCE_LOSSES
    eval(quote
         sig = _signature($Loss_ex)
         isempty(sig) || (sig = "Constructor signature: "*sig)
         @create_aliases $Loss_ex
         @create_docs($Loss_ex,
                      typename = name($Loss_ex),
                      body=DOC_LOSS_FUNCTIONS,
                      footer=sig)
         end)
end

for Loss_ex in MARGIN_LOSSES
    eval(quote
         sig = _signature($Loss_ex)
         isempty(sig) || (sig = "Constructor signature: "*sig)
         @create_aliases $Loss_ex
         @create_docs($Loss_ex,
                      typename = name($Loss_ex),
                      body=DOC_LOSS_FUNCTIONS,
                      scitype=DOC_FINITE_BINARY,
                      footer= sig)
         end)
end
