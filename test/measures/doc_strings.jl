using MLJBase

docstring = (Base.Docs.doc)((Base.Docs.Binding)(Main, :multiclass_recall))

@test string(docstring) == "An instance of type "*
    "[`MulticlassTruePositiveRate`](@ref). Query the "*
    "[`MulticlassTruePositiveRate`](@ref) doc-string for details. \n"

true
