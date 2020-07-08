module OpenML

using HTTP
using JSON

const API_URL = "https://www.openml.org/api/v1/json"

# Data API
# The structures are based on these descriptions
# https://github.com/openml/OpenML/tree/master/openml_OS/views/pages/api_new/v1/xsd
# https://www.openml.org/api_docs#!/data/get_data_id

# To do:
# - Save the file in a local folder
# - Check downloaded files in local folder before downloading it again
# - Use local stored file whenever possible

"""
Returns information about a dataset. The information includes the name,
information about the creator, URL to download it and more.

- 110 - Please provide data_id.
- 111 - Unknown dataset. Data set description with data_id was not found in the database.
- 112 - No access granted. This dataset is not shared with you.
"""
function load_Dataset_Description(id::Int; api_key::String="")
    url = string(API_URL, "/data/$id")
    try
        r = HTTP.request("GET", url)
        if r.status == 200
            return JSON.parse(String(r.body))
        elseif r.status == 110
            println("Please provide data_id.")
        elseif r.status == 111
            println("Unknown dataset. Data set description with data_id was not found in the database.")
        elseif r.status == 112
            println("No access granted. This dataset is not shared with you.")
        end
    catch e
        println("Error occurred : $e")
        return nothing
    end
    return nothing
end

"""
Returns a Vector of NamedTuples.
Receives an `HTTP.Message.response` that has an
ARFF file format in the `body` of the `Message`.
"""
function convert_ARFF_to_rowtable(response)
    data = String(response.body)
    data2 = split(data, "\n")

    featureNames = String[]
    dataTypes = String[]
    # TODO: make this more performant by anticipating types?
    named_tuples = [] # `Any` type here bad
    for line in data2
        if length(line) > 0
            if line[1:1] != "%"
                d = []
                if occursin("@attribute", lowercase(line))
                    push!(featureNames, replace(replace(split(line, " ")[2], "'" => ""), "-" => "_"))
                    push!(dataTypes, split(line, " ")[3])
                elseif occursin("@relation", lowercase(line))
                    nothing
                elseif occursin("@data", lowercase(line))
                    # it means the data starts
                    nothing
                else
                    values = split(line, ",")
                    for i in eachindex(featureNames)
                        if lowercase(dataTypes[i]) in ["real","numeric"]
                            push!(d, featureNames[i] => Meta.parse(values[i]))
                        else
                            # all the rest will be considered as String
                            push!(d, featureNames[i] => values[i])
                        end
                    end
                    push!(named_tuples, (; (Symbol(k) => v for (k,v) in d)...))
                end
            end
        end
    end
    return identity.(named_tuples) # not performant; see above
end

"""
    OpenML.load(id)

Load the OpenML dataset with specified `id`, from those listed on the
[OpenML site](https://www.openml.org/search?type=data).

Returns a "row table", i.e., a `Vector` of identically typed
`NamedTuple`s. A row table is compatible with the
[Tables.jl](https://github.com/JuliaData/Tables.jl) interface and can
therefore be readily converted to other compatible formats. For
example:

```julia
using DataFrames
rowtable = OpenML.load(61);
df = DataFrame(rowtable);
df2 = coerce(df, :class=>Multiclass)
```
"""
function load(id::Int)
    response = load_Dataset_Description(id)
    arff_file = HTTP.request("GET", response["data_set_description"]["url"])
    return convert_ARFF_to_rowtable(arff_file)
end


"""
Returns a list of all data qualities in the system.

- 412 - Precondition failed. An error code and message are returned
- 370 - No data qualities available. There are no data qualities in the system.
"""
function load_Data_Qualities_List()
    url = string(API_URL, "/data/qualities/list")
    try
        r = HTTP.request("GET", url)
        if r.status == 200
            return JSON.parse(String(r.body))
        elseif r.status == 370
            println("No data qualities available. There are no data qualities in the system.")
        end
    catch e
        println("Error occurred : $e")
        return nothing
    end
    return nothing
end

"""
Returns a list of all data qualities in the system.

- 271 - Unknown dataset. Data set with the given data ID was not found (or is not shared with you).
- 272 - No features found. The dataset did not contain any features, or we could not extract them.
- 273 - Dataset not processed yet. The dataset was not processed yet, features are not yet available. Please wait for a few minutes.
- 274 - Dataset processed with error. The feature extractor has run into an error while processing the dataset. Please check whether it is a valid supported file. If so, please contact the API admins.
"""
function load_Data_Features(id::Int; api_key::String = "")
    if api_key == ""
        url = string(API_URL, "/data/features/$id")
    end
    try
        r = HTTP.request("GET", url)
        if r.status == 200
            return JSON.parse(String(r.body))
        elseif r.status == 271
            println("Unknown dataset. Data set with the given data ID was not found (or is not shared with you).")
        elseif r.status == 272
            println("No features found. The dataset did not contain any features, or we could not extract them.")
        elseif r.status == 273
            println("Dataset not processed yet. The dataset was not processed yet, features are not yet available. Please wait for a few minutes.")
        elseif r.status == 274
            println("Dataset processed with error. The feature extractor has run into an error while processing the dataset. Please check whether it is a valid supported file. If so, please contact the API admins.")
        end
    catch e
        println("Error occurred : $e")
        return nothing
    end
    return nothing
end

"""
Returns the qualities of a dataset.

- 360 - Please provide data set ID
- 361 - Unknown dataset. The data set with the given ID was not found in the database, or is not shared with you.
- 362 - No qualities found. The registered dataset did not contain any calculated qualities.
- 363 - Dataset not processed yet. The dataset was not processed yet, no qualities are available. Please wait for a few minutes.
- 364 - Dataset processed with error. The quality calculator has run into an error while processing the dataset. Please check whether it is a valid supported file. If so, contact the support team.
- 365 - Interval start or end illegal. There was a problem with the interval start or end.
"""
function load_Data_Qualities(id::Int; api_key::String = "")
    if api_key == ""
        url = string(API_URL, "/data/qualities/$id")
    end
    try
        r = HTTP.request("GET", url)
        if r.status == 200
            return JSON.parse(String(r.body))
        elseif r.status == 360
            println("Please provide data set ID")
        elseif r.status == 361
            println("Unknown dataset. The data set with the given ID was not found in the database, or is not shared with you.")
        elseif r.status == 362
            println("No qualities found. The registered dataset did not contain any calculated qualities.")
        elseif r.status == 363
            println("Dataset not processed yet. The dataset was not processed yet, no qualities are available. Please wait for a few minutes.")
        elseif r.status == 364
            println("Dataset processed with error. The quality calculator has run into an error while processing the dataset. Please check whether it is a valid supported file. If so, contact the support team.")
        elseif r.status == 365
            println("Interval start or end illegal. There was a problem with the interval start or end.")
        end
    catch e
        println("Error occurred : $e")
        return nothing
    end
    return nothing
end

"""
List datasets, possibly filtered by a range of properties.
Any number of properties can be combined by listing them one after
the other in the
form '/data/list/{filter}/{value}/{filter}/{value}/...'
Returns an array with all datasets that match the constraints.

Any combination of these filters /limit/{limit}/offset/{offset} -
returns only {limit} results starting from result number {offset}.
Useful for paginating results. With /limit/5/offset/10,
    results 11..15 will be returned.

Both limit and offset need to be specified.
/status/{status} - returns only datasets with a given status,
either 'active', 'deactivated', or 'in_preparation'.
/tag/{tag} - returns only datasets tagged with the given tag.
/{data_quality}/{range} - returns only tasks for which the
underlying datasets have certain qualities.
{data_quality} can be data_id, data_name, data_version, number_instances,
number_features, number_classes, number_missing_values. {range} can be a
specific value or a range in the form 'low..high'.
Multiple qualities can be combined, as in
'number_instances/0..50/number_features/0..10'.

- 370 - Illegal filter specified.
- 371 - Filter values/ranges not properly specified.
- 372 - No results. There where no matches for the given constraints.
- 373 - Can not specify an offset without a limit.
"""
function load_List_And_Filter(filters::String; api_key::String = "")
    if api_key == ""
        url = string(API_URL, "/data/list/$filters")
    end
    try
        r = HTTP.request("GET", url)
        if r.status == 200
            return JSON.parse(String(r.body))
        elseif r.status == 370
            println("Illegal filter specified.")
        elseif r.status == 371
            println("Filter values/ranges not properly specified.")
        elseif r.status == 372
            println("No results. There where no matches for the given constraints.")
        elseif r.status == 373
            println("Can not specify an offset without a limit.")
        end
    catch e
        println("Error occurred : $e")
        return nothing
    end
    return nothing
end

# Flow API

# Task API

# Run API

end # module
