struct Potato <: Model end

@testset "message_expecting_model" begin

    m1 = MLJBase.message_expecting_model(123)
    @test !isnothing(match(r"(Expected)", m1))
    @test isnothing(match(r"(type)", m1))
    @test isnothing(match(r"(mispelled)", m1))

    m2 = MLJBase.message_expecting_model(Potato)
    @test !isnothing(match(r"(Expected)", m2))
    @test !isnothing(match(r"(type)", m2))
    @test isnothing(match(r"(mispelled)", m2))

    m3 = MLJBase.message_expecting_model(123, spelling=true)
    @test !isnothing(match(r"(Expected)", m3))
    @test isnothing(match(r"(type)", m3))
    @test !isnothing(match(r"(misspelled)", m3))

    m4 = MLJBase.message_expecting_model(Potato, spelling=true)
    @test !isnothing(match(r"(Expected)", m4))
    @test !isnothing(match(r"(type)", m4))
    @test isnothing(match(r"(misspelled)", m4))

end

@testset "check_ismodel" begin
    @test isnothing(MLJBase.check_ismodel(Potato()))
    @test_throws(
        MLJBase.err_expecting_model(123),
        MLJBase.check_ismodel(123),
    )
    @test_throws(
        MLJBase.err_expecting_model(123, spelling=true),
        MLJBase.check_ismodel(123, spelling=true),
    )
    @test_throws(
        MLJBase.err_expecting_model(Potato),
        MLJBase.check_ismodel(Potato),
    )
end
true
