# Datasets
```@index
Pages   = ["data/datasets_synthetic.jl"]
```


## Standard datasets

To add a new dataset assuming it has a header and is, at path
`data/newdataset.csv`

Start by loading it with CSV:
```julia
fpath = joinpath("datadir", "newdataset.csv")
data = CSV.read(fpath, copycols=true,
                categorical=true)
```

Load it with DelimitedFiles and Tables
```julia
data_raw, data_header = readdlm(fpath, ',', header=true)
data_table = Tables.table(data_raw; header=Symbol.(vec(data_header)))
```

Retrieve the conversions:
```julia
for (n, st) in zip(names(data), scitype_union.(eachcol(data)))
    println(":$n=>$st,")
end
```

Copy and paste the result in a coerce
```julia
data_table = coerce(data_table, ...)
```


```@autodocs
Modules = [MLJBase]
Pages   = ["data/datasets.jl"]
```

## Synthetic datasets
```@autodocs
Modules = [MLJBase]
Pages   = ["data/datasets_synthetic.jl"]
```


## Utility functions

```@autodocs
Modules = [MLJBase]
Pages   = ["data/data.jl"]
```
