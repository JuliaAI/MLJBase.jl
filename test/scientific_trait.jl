import ScientificTypes.trait


struct Foo1 <: Supervised end
struct Bar1 <: Unsupervised end

@test trait(rms) == :measure
@test trait(Foo1()) == :supervised_model
@test trait(Bar1()) == :unsupervised_model

true
