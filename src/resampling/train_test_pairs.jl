# # LOCAL TYPE ALIASES

const AbstractRow = Union{AbstractVector{<:Integer}, Colon}
const TrainTestPair = Tuple{AbstractRow, AbstractRow}
const TrainTestPairs = Union{
    Tuple{Vararg{TrainTestPair}},
    AbstractVector{<:TrainTestPair},
}
