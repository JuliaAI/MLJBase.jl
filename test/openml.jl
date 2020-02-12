using Test
using HTTP
using MLJBase

response_test = OpenML.load_Dataset_Description(61)
@test typeof(response_test) <: Dict
@test response_test["data_set_description"]["name"] == "iris"

df_test = OpenML.load(61)
@test length(df_test) == 150

dqlist_test = OpenML.load_Data_Qualities_List()
@test typeof(dqlist_test["data_qualities_list"]) <: Dict

data_features_test = OpenML.load_Data_Features(61)
@test typeof(data_features_test) <: Dict
@test length(data_features_test["data_features"]["feature"]) == 5

data_qualities_test = OpenML.load_Data_Qualities(61)
@test typeof(data_qualities_test) <: Dict

limit = 5
offset = 8
filters_test = OpenML.load_List_And_Filter("limit/$limit/offset/$offset")
@test length(filters_test["data"]["dataset"]) == limit
@test length(filters_test["data"]["dataset"][1]) == offset

true
