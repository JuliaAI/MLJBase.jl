abstract type AbstractPerformanceEvaluation <: MLJType end

# ---------------------------------------------------------------
# PerformanceEvaluation (type)

"""
    PerformanceEvaluation <: AbstractPerformanceEvaluation

Type of object returned by [`evaluate`](@ref) (for models plus data) or
[`evaluate!`](@ref) (for machines). Such objects encode estimates of the performance
(generalization error) of a supervised model or outlier detection model, and store other
information ancillary to the computation.

If [`evaluate`](@ref) or [`evaluate!`](@ref) is called with the `compact=true` option,
then a [`CompactPerformanceEvaluation`](@ref) object is returned instead.

When `evaluate`/`evaluate!` is called, a number of train/test pairs ("folds") of row
indices are generated, according to the options provided, which are discussed in the
[`evaluate!`](@ref) doc-string. Rows correspond to observations. The generated train/test
pairs are recorded in the `train_test_rows` field of the `PerformanceEvaluation` struct,
and the corresponding estimates, aggregated over all train/test pairs, are recorded in
`measurement`, a vector with one entry for each measure (metric) recorded in `measure`.

When displayed, a `PerformanceEvaluation` object includes a value under the heading
`1.96*SE`, derived from the standard error of the `per_fold` entries. This value is
suitable for constructing a formal 95% confidence interval for the given
`measurement`. Such intervals should be interpreted with caution. See, for example, [Bates
et al. (2021)](https://arxiv.org/abs/2104.00673). It is also stored in the field
`uncertainty_radius_95`.

### Fields

These fields are part of the public API of the `PerformanceEvaluation` struct.

- `model`: model used to create the performance evaluation. In the case a
    tuning model, this is the best model found.

- `tag`: a string label associated with the evaluation, specified by the user when
  replacing `mach` in `evaluate!(mach, ...)` with `tag => mach` (or `model` in
  `evaluate(model, ...)` with `tag => model`). If unspecified, it is auto-generated, but
  tag-uniqueness is not 100% guaranteed.

 - `measure`: vector of measures (metrics) used
  to evaluate performance

- `measurement`: vector of measurements - one for each element of `measure` - aggregating
  the performance measurements over all train/test pairs (folds). The aggregation method
  applied for a given measure `m` is
  `StatisticalMeasuresBase.external_aggregation_mode(m)` (commonly `Mean()` or `Sum()`)

- `uncertainty_radius_95`: vector of radii of uncertainty for 95% confidence intervals,
  one for each element of `meaures`. See cautionary note above.

- `operation` (e.g., `predict_mode`): the operations applied for each measure to generate
  predictions to be evaluated. Possibilities are: $PREDICT_OPERATIONS_STRING.

- `per_fold`: a vector of vectors of individual test fold evaluations (one vector per
  measure). Useful for obtaining a rough estimate of the variance of the performance
  estimate.

- `per_observation`: a vector of vectors of vectors containing individual per-observation
  measurements: for an evaluation `e`, `e.per_observation[m][f][i]` is the measurement for
  the `i`th observation in the `f`th test fold, evaluated using the `m`th measure.  Useful
  for some forms of hyper-parameter optimization. Note that an aggregregated measurement
  for some measure `measure` is repeated across all observations in a fold if
  `StatisticalMeasures.can_report_unaggregated(measure) == true`. If `e` has been computed
  with the `per_observation=false` option, then `e_per_observation` is a vector of
  `missings`.

- `fitted_params_per_fold`: a vector containing `fitted params(mach)` for each machine
  `mach` trained during resampling - one machine per train/test pair. Use this to extract
  the learned parameters for each individual training event.

- `report_per_fold`: a vector containing `report(mach)` for each machine `mach` training
  in resampling - one machine per train/test pair.

- `train_test_rows`: a vector of tuples, each of the form `(train, test)`, where `train`
  and `test` are vectors of row (observation) indices for training and evaluation
  respectively.

- `resampling`: the user-specified resampling strategy to generate the train/test pairs
  (or literal train/test pairs if that was directly specified).

- `repeats`: the number of times the resampling strategy was repeated.

See also [`CompactPerformanceEvaluation`](@ref).

"""
struct PerformanceEvaluation{M,
                             Measure,
                             Measurement,
                             Uncertainty,
                             Operation,
                             PerFold,
                             PerObservation,
                             FittedParamsPerFold,
                             ReportPerFold,
                             R} <: AbstractPerformanceEvaluation
    model::M
    tag::String
    measure::Measure
    measurement::Measurement
    uncertainty_radius_95::Uncertainty
    operation::Operation
    per_fold::PerFold
    per_observation::PerObservation
    fitted_params_per_fold::FittedParamsPerFold
    report_per_fold::ReportPerFold
    train_test_rows::TrainTestPairs
    resampling::R
    repeats::Int
end

# ---------------------------------------------------------------
# CompactPerformanceEvaluation (type)

"""
    CompactPerformanceEvaluation <: AbstractPerformanceEvaluation

Type of object returned by [`evaluate`](@ref) (for models plus data) or
[`evaluate!`](@ref) (for machines) when called with the option `compact = true`. Such
objects have the same structure as the [`PerformanceEvaluation`](@ref) objects returned by
default, except that the following fields are omitted to save memory:
`fitted_params_per_fold`, `report_per_fold`, `train_test_rows`.

For more on the remaining fields, see [`PerformanceEvaluation`](@ref).

"""
struct CompactPerformanceEvaluation{M,
                                    Measure,
                                    Measurement,
                                    Uncertainty,
                                    Operation,
                                    PerFold,
                                    PerObservation,
                                    R} <: AbstractPerformanceEvaluation
    model::M
    tag::String
    measure::Measure
    measurement::Measurement
    uncertainty_radius_95::Uncertainty
    operation::Operation
    per_fold::PerFold
    per_observation::PerObservation
    resampling::R
    repeats::Int
end

compactify(e::CompactPerformanceEvaluation) = e
compactify(e::PerformanceEvaluation) = CompactPerformanceEvaluation(
    e.model,
    e.tag,
    e.measure,
    e.measurement,
    e.uncertainty_radius_95,
    e.operation,
    e.per_fold,
    e. per_observation,
    e.resampling,
    e.repeats,
)

# pretty printing:
round3(x) = x
round3(x::AbstractFloat) = round(x, sigdigits=3)

# to address #874, while preserving the display worked out in #757:
_repr_(f::Function) = repr(f)
_repr_(x) = repr("text/plain", x)
_repr(::Nothing) = ""

function uncertainty_as_string(δ)
    isnothing(δ) && return ""
    δ isa Real && isinf(δ) && return ""
    return string(round3(δ))
end

# helper for row labels: _label(1) ="A", _label(2) = "B", _label(27) = "BA", etc
const alphabet = Char.(65:90)
_label(i) = map(digits(i - 1, base=26)) do d alphabet[d + 1] end |> join |> reverse

function Base.show(io::IO, ::MIME"text/plain", e::AbstractPerformanceEvaluation)
    # for getting rid of bold in table headings:
    style = PrettyTables.TextTableStyle(
        first_line_column_label = PrettyTables.crayon"black",
    )

    _measure = [_repr_(m) for m in e.measure]
    _measurement = round3.(e.measurement)
    _per_fold = reshape([round3.(v) for v in e.per_fold], length(e.per_fold), 1)
    _uncertainty_radius_95 = uncertainty_as_string.(e.uncertainty_radius_95)
    show_radius = any(x -> !isempty(x), _uncertainty_radius_95)
    row_labels = _label.(eachindex(e.measure))

    # Define header and data for main table

    data = hcat(_measure, e.operation, _measurement)
    header = ["measure", "operation", "measurement"]
    if length(row_labels) > 1 && length(first(e.per_fold)) > 1
        data = hcat(row_labels, data)
        header =["", header...]
    end

    if e isa PerformanceEvaluation
        println(io, "PerformanceEvaluation object "*
            "with these fields:")
        println(io, "  model, tag, measure, operation,\n"*
            "  measurement, uncertainty_radius_95, per_fold, per_observation,\n"*
            "  fitted_params_per_fold, report_per_fold,\n"*
            "  train_test_rows, resampling, repeats")
    else
        println(io, "CompactPerformanceEvaluation object "*
            "with these fields:")
        println(io, "  model, measure, operation,\n"*
            "  measurement, per_fold, per_observation,\n"*
            "  resampling, repeats")
    end
    println(io, "Tag: $(e.tag)")
    println(io, "Extract:")
    show_color = MLJBase.SHOW_COLOR[]
    color_off()
    PrettyTables.pretty_table(
        io,
        data;
        column_labels = [header,],
        alignment=:l,
        line_breaks=true,
        style,
    )

    # Show the per-fold table if needed:

    if length(first(e.per_fold)) > 1
        data2 = _per_fold
        header2 = ["per_fold", ]
        if show_radius
            data2 = hcat(_per_fold, _uncertainty_radius_95)
            header2 = [header2..., "1.96*SE"]
        end
        if length(row_labels) > 1
            data2 = hcat(row_labels, data2)
            header2 =["", header2...]
        end
        PrettyTables.pretty_table(
            io,
            data2;
            column_labels = [header2,],
            alignment=:l,
            line_breaks=true,
            style,
        )
    end
    show_color ? color_on() : color_off()
end

function _summary(e)
    confidence_intervals = map(zip(e.measurement, e.uncertainty_radius_95)) do (μ, δ)
        a = round3(μ)
        b = uncertainty_as_string(δ)
        isempty(b) ? a : "$a ± $b"
    end
    return "(\"$(e.tag)\", "*join(confidence_intervals, ", ")*")"
end

Base.show(io::IO, e::PerformanceEvaluation) =
    print(io, "PerformanceEvaluation$(_summary(e))")
Base.show(io::IO, e::CompactPerformanceEvaluation) =
    print(io, "CompactPerformanceEvaluation$(_summary(e))")

