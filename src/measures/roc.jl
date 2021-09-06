## ROC COMPUTATION

"""
    _idx_unique_sorted(v)

Internal function to return the index of unique elements in `v` under the
assumption that the vector `v` is sorted in decreasing order.
"""
function _idx_unique_sorted(v::Vec{<:Real})
    n    = length(v)
    idx  = ones(Int, n)
    p, h = 1, 1
    cur  = v[1]
    @inbounds while h < n
        h     += 1                  # head position
        cand   = v[h]               # candidate value
        cand   < cur || continue    # is it new? otherwise skip
        p     += 1                  # if new store it
        idx[p] = h
        cur    = cand               # and update the last seen value
    end
    p < n && deleteat!(idx, p+1:n)
    return idx
end

"""
    fprs, tprs, ts = roc_curve(ŷ, y) = roc(ŷ, y)

Return the ROC curve for a two-class probabilistic prediction `ŷ` given the
ground  truth `y`. The true positive rates, false positive rates over a range
of thresholds `ts` are returned. Note that if there are `k` unique scores,
there are correspondingly  `k` thresholds and `k+1` "bins" over which the FPR
and TPR are constant:

* `[0.0 - thresh[1]]`
* `[thresh[1] - thresh[2]]`
* ...
* `[thresh[k] - 1]`

consequently, `tprs` and `fprs` are of length `k+1` if `ts` is of length `k`.

To draw the curve using your favorite plotting backend, do `plot(fprs, tprs)`.
"""
function roc_curve(ŷm, ym)
    ŷ, y    = skipinvalid(ŷm, ym)
    n       = length(y)
    lab_pos = levels(y)[2]
    scores  = pdf.(ŷ, lab_pos)
    ranking = sortperm(scores, rev=true)

    scores_sort = scores[ranking]
    y_sort_bin  = (y[ranking] .== lab_pos)

    idx_unique = _idx_unique_sorted(scores_sort)
    thresholds = scores_sort[idx_unique]

    # detailed computations with example:
    # y = [  1   0   0   1   0   0   1]
    # s = [0.5 0.5 0.2 0.2 0.1 0.1 0.1] thresh are 0.5 0.2, 0.1 // idx [1, 3, 5]
    # ŷ = [  0   0   0   0   0   0   0] (0.5 - 1.0] # no pos pred
    # ŷ = [  1   1   0   0   0   0   0] (0.2 - 0.5] # 2 pos pred
    # ŷ = [  1   1   1   1   0   0   0] (0.1 - 0.2] # 4 pos pred
    # ŷ = [  1   1   1   1   1   1   1] [0.0 - 0.1] # all pos pre

    idx_unique_2 = idx_unique[2:end]   # [3, 5]
    n_ŷ_pos      = idx_unique_2 .- 1   # [2, 4] implicit [0, 2, 4, 7]

    cs   = cumsum(y_sort_bin)          # [1, 1, 1, 2, 2, 2, 3]
    n_tp = cs[n_ŷ_pos]                 # [1, 2] implicit [0, 1, 2, 3]
    n_fp = n_ŷ_pos .- n_tp             # [1, 2] implicit [0, 1, 2, 4]

    # add end points
    P = sum(y_sort_bin) # total number of true positives
    N = n - P           # total number of true negatives

    n_tp = [0, n_tp..., P] # [0, 1, 2, 3]
    n_fp = [0, n_fp..., N] # [0, 1, 2, 4]

    tprs = n_tp ./ P  # [0/3, 1/3, 2/3, 1]
    fprs = n_fp ./ N  # [0/4, 1/4, 2/4, 1]

    return fprs, tprs, thresholds
end

const roc = roc_curve
