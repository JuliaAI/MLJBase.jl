const PREDICT_OPERATIONS_STRING = begin
    strings = map(PREDICT_OPERATIONS) do op
        "`"*string(op)*"`"
    end
    join(strings, ", ", ", or ")
end
const PROG_METER_DT = 0.1
const ERR_WEIGHTS_LENGTH =
    DimensionMismatch("`weights` and target have different lengths. ")
const ERR_WEIGHTS_DICT =
    ArgumentError("`class_weights` must be a "*
                  "dictionary with `Real` values. ")
const ERR_WEIGHTS_CLASSES =
    DimensionMismatch("The keys of `class_weights` "*
                      "are not the same as the levels of the "*
                      "target, `y`. Do `levels(y)` to check levels. ")
const ERR_OPERATION_MEASURE_MISMATCH = DimensionMismatch(
    "The number of operations and the number of measures are different. ")
const ERR_INVALID_OPERATION = ArgumentError(
    "Invalid `operation` or `operations`. "*
    "An operation must be one of these: $PREDICT_OPERATIONS_STRING. ")
_ambiguous_operation(model, measure) =
    "`$measure` does not support a `model` with "*
    "`prediction_type(model) == :$(prediction_type(model))`. "
err_incompatible_prediction_types(model, measure) = ArgumentError(
    _ambiguous_operation(model, measure)*
    "If your model is truly making probabilistic predictions, try explicitly "*
    "specifying operations. For example, for "*
    "`measures = [area_under_curve, accuracy]`, try "*
    "`operations=[predict, predict_mode]`. ")
const LOG_AVOID = "\nTo override measure checks, set check_measure=false. "
const LOG_SUGGESTION1 =
    "\nPerhaps you want to set `operation="*
    "predict_mode` or need to "*
    "specify multiple operations, "*
    "one for each measure. "
const LOG_SUGGESTION2 =
    "\nPerhaps you want to set `operation="*
    "predict_mean` or `operation=predict_median`, or "*
    "specify multiple operations, "*
    "one for each measure. "
ERR_MEASURES_OBSERVATION_SCITYPE(measure, T_measure, T) = ArgumentError(
    "\nobservation scitype of target = `$T` but ($measure) only supports "*
        "`$T_measure`."*LOG_AVOID
)
ERR_MEASURES_PROBABILISTIC(measure, suggestion) = ArgumentError(
    "The model subtypes `Probabilistic`, and so is not supported by "*
        "`$measure`. $suggestion"*LOG_AVOID
)
ERR_MEASURES_DETERMINISTIC(measure) = ArgumentError(
    "The model subtypes `Deterministic`, "*
        "and so is not supported by `$measure`. "*LOG_AVOID
)

err_ambiguous_operation(model, measure) = ArgumentError(
    _ambiguous_operation(model, measure)*
    "\nUnable to infer an appropriate operation for `$measure`. "*
    "Explicitly specify `operation=...` or `operations=...`. "*
    "Possible value(s) are: $PREDICT_OPERATIONS_STRING. "
)

const ERR_UNSUPPORTED_PREDICTION_TYPE = ArgumentError(
    """

    The `prediction_type` of your model needs to be one of: `:deterministic`,
    `:probabilistic`, or `:interval`. Does your model implement one of these operations:
    $PREDICT_OPERATIONS_STRING? If so, you can try explicitly specifying `operation=...`
    or `operations=...` (and consider posting an issue to have the model review it's
    definition of `MLJModelInterface.prediction_type`). Otherwise, performance
    evaluation is not supported.

   """
)

const ERR_NEED_TARGET = ArgumentError(
   """

    To evaluate a model's performance you must provide a target variable `y`, as in
    `evaluate(model, X, y; options...)` or

        mach = machine(model, X, y)
        evaluate!(mach; options...)

    """
)

const ERR_BAD_RESAMPLING_OPTION = ArgumentError(
    "`resampling` must be an "*
        "`MLJ.ResamplingStrategy` or a vector (or tuple) of tuples "*
        "of the form `(train_rows, test_rows)`"
)

const ERR_EMPTY_RESAMPLING_OPTION = ArgumentError(
    "`resampling` cannot be empty. It must be an "*
        "`MLJ.ResamplingStrategy` or a vector (or tuple) of tuples "*
        "of the form `(train_rows, test_rows)`"
)
