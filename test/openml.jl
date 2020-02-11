using Test
using HTTP
import MLJBase

response_test = MLJBase.getDatasetDescription(61)
@test typeof(response_test) <: Dict
@test response_test["data_set_description"]["name"] == "iris"

df_test = MLJBase.getDataframeFromOpenmlAPI(61)
@test length(df_test) == 5

arff_file_test = HTTP.request("GET", response_test["data_set_description"]["url"])
df_test2 = MLJBase.convertArffToDataFrame(arff_file_test)
@test length(df_test2) == 5

dqlist_test = MLJBase.getDataQualitiesList()
@test typeof(response_test) <: Dict

dataFeatures_test = MLJBase.getDataFeatures(61)
@test typeof(dataFeatures_test) <: Dict

dataQualities_test = MLJBase.getDataQualities(61)
@test typeof(dataQualities_test) <: Dict

limit = 5
offset = 8
filters_test = MLJBase.getListAndFilter("limit/$limit/offset/$offset")
@test length(filters_test["data"]["dataset"]) == limit
@test length(filters_test["data"]["dataset"][1]) == offset

true
