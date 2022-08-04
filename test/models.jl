@testset "message_expecting_model" begin

    m1 = MLJBase.message_expecting_model(123)
    @test !isempty(match(r"(Expected)", m1))
    @test isnothing(match(r"(type)", m1))
    @test isnothing(match(r"(mispelled)", m1))

    m2 = MLJBase.message_expecting_model(ConstantClassifier)
    @test !isempty(match(r"(Expected)", m2))
    @test !isempty(match(r"(type)", m2))
    @test isnothing(match(r"(mispelled)", m2))

    m3 = MLJBase.message_expecting_model(123, spelling=true)
    @test !isempty(match(r"(Expected)", m3))
    @test isnothing(match(r"(type)", m3))
    @test !isempty(match(r"(misspelled)", m3))

    m4 = MLJBase.message_expecting_model(ConstantClassifier, spelling=true)
    @test !isempty(match(r"(Expected)", m4))
    @test !isempty(match(r"(type)", m4))
    @test isnothing(match(r"(misspelled)", m4))

end

@testset "check_ismodel" begin
    @test isnothing(MLJBase.check_ismodel(ConstantClassifier()))
    @test_throws(
        MLJBase.err_expecting_model(123),
        MLJBase.check_ismodel(123),
    )
    @test_throws(
        MLJBase.err_expecting_model(123, spelling=true),
        MLJBase.check_ismodel(123, spelling=true),
    )
    @test_throws(
        MLJBase.err_expecting_model(ConstantClassifier),
        MLJBase.check_ismodel(ConstantClassifier),
    )
end
