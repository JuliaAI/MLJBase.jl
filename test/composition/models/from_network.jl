module TestFromComposite

using Test
using Tables
using MLJBase
using ..Models
using ..TestUtilities
using CategoricalArrays
using StableRNGs
using Parameters
rng = StableRNG(616161)

ridge_model = FooBarRegressor(lambda=0.1)
selector_model = FeatureSelector()


## FROM_NETWORK_PREPROCESS

# supervised:
Xs = source(nothing)
ys = source(nothing)
z  = log(ys)
stand = UnivariateStandardizer()
standM = machine(stand, z)
u = transform(standM, z)
hot = OneHotEncoder()
hotM = machine(hot, Xs)
W = transform(hotM, Xs)
knn = KNNRegressor()
knnM = machine(knn, W, u)
oak = DecisionTreeRegressor()
oakM = machine(oak, W, u)
uhat = 0.5*(predict(knnM, W) + predict(oakM, W))
zhat = inverse_transform(standM, uhat)
yhat = exp(zhat)

mach_ex = :(machine(Deterministic(), Xs, ys; predict=yhat))

## TESTING `from_network_preprocess`

ex = Meta.parse(
    "begin
         mutable struct CompositeX
             knn_rgs=knn
             one_hot_enc=hot
         end
         target_scitype=AbstractVector{<:Continuous}
         input_scitype=Table(Continuous,Multiclass)
     end")
mach_, modeltype_ex, struct_ex, no_fields, dic =
    MLJBase.from_network_preprocess(TestFromComposite, mach_ex, ex)
eval(Parameters.with_kw(struct_ex, TestFromComposite, false))
@test supertype(CompositeX) == DeterministicComposite
composite = CompositeX()
@test composite.knn_rgs == knn
@test composite.one_hot_enc == hot
@test dic[:target_scitype] == :(AbstractVector{<:Continuous})
@test dic[:input_scitype] == :(Table(Continuous, Multiclass))

ex = Meta.parse(
    "begin
         mutable struct Composite4 <: ProbabilisticComposite
             knn_rgs=knn
             one_hot_enc=hot
         end
     end")
mach_, modeltype_ex, struct_ex =
    MLJBase.from_network_preprocess(TestFromComposite, mach_ex, ex)
eval(Parameters.with_kw(struct_ex, TestFromComposite, false))
@test supertype(Composite4) == ProbabilisticComposite

ex = Meta.parse(
    "mutable struct Composite2
        knn_rgs=knn
        one_hot_enc=hot
     end")
mach_, modeltype_ex, struct_ex, no_fields, dic =
    MLJBase.from_network_preprocess(TestFromComposite, mach_ex, ex)
eval(Parameters.with_kw(struct_ex, TestFromComposite, false))
composite = Composite2()
@test composite.knn_rgs == knn
@test composite.one_hot_enc == hot

ex = Meta.parse(
    "begin
         mutable struct Composite6 <: Probabilistic
             knn_rgs=knn
             one_hot_enc=hot
         end
     end")
@test_logs((:warn, r"New composite"),
           MLJBase.from_network_preprocess(TestFromComposite, mach_ex, ex))

ex = Meta.parse(
    "begin
         mutable struct Composite20
             knn_rgs=knn
             one_hot_enc=hot
         end
         target_scitype == Continuous
     end")
@test_throws(ArgumentError,
             MLJBase.from_network_preprocess(TestFromComposite, mach_ex, ex))

ex = Meta.parse(
    "begin
         mutable struct Composite20
             knn_rgs=knn
             one_hot_enc=hot
         end
         Continuous
     end")
@test_throws(ArgumentError,
             MLJBase.from_network_preprocess(TestFromComposite, mach_ex, ex))

ex = Meta.parse(
    "begin
         mutable struct Composite20
             knn_rgs=knn
             one_hot_enc=hot
         end
         43 = Continuous
     end")
@test_throws(ArgumentError,
             MLJBase.from_network_preprocess(TestFromComposite, mach_ex, ex))

ex = Meta.parse(
    "begin
         mutable struct Composite7 < Probabilistic
             knn_rgs=knn
             one_hot_enc=hot
         end
     end")
@test_throws(ArgumentError,
           MLJBase.from_network_preprocess(TestFromComposite, mach_ex, ex))

@test_throws(ArgumentError,
             MLJBase.from_network_preprocess(TestFromComposite, knn, ex))

ex = Meta.parse(
    "begin
         Composite3(
             knn_rgs=knn,
             one_hot_enc=hot)
     end")
@test_throws(ArgumentError,
             MLJBase.from_network_preprocess(TestFromComposite, mach_ex, ex))

ex = Meta.parse(
    "begin
         mutable struct Composite8
             knn_rgs::KNNRegressor=knn
             one_hot_enc=hot
         end
     end")
mach_, modeltype_ex, struct_ex =
    MLJBase.from_network_preprocess(TestFromComposite, mach_ex, ex)
eval(Parameters.with_kw(struct_ex, TestFromComposite, false))
VERSION ≥ v"1.3.0-" &&
    @test fieldtypes(Composite8) == (KNNRegressor, Any)

# test that you cannot leave "default" component models unspecified:
modeltype_ex = :Composite9
struct_ex = :(mutable struct Composite9 <: DeterministicComposite
              knn_rgs::KNNRegressor
              one_hot_enc = hot
              end)
@test_logs (:error, r"Problem instantiating") begin
    @test_throws Exception begin
        MLJBase.from_network_(TestFromComposite,
                              mach_ex, modeltype_ex,
                              struct_ex, false, Dict{Symbol,Any}())
    end
end


## TEST MACRO-EXPORTED  NETWORKS
# (CANNOT WRAP IN @testset)

# some actual data:
N = 10
X = MLJBase.table(rand(N, 3))
y = rand(N)
w = rand(N)

# supervised with sample weights:
ws = source()
knnM = machine(knn, W, u, ws)
uhat = 0.5*(predict(knnM, W) + predict(oakM, W))
zhat = inverse_transform(standM, uhat)
yhat = exp(zhat)

@from_network machine(Deterministic(), Xs, ys, ws; predict=yhat) begin
    mutable struct CompositeX1
        knn_rgs=knn
        one_hot_enc=hot
    end
    supports_weights = true
    target_scitype = AbstractVector{<:Continuous}
end
model = CompositeX1()
@test supports_weights(model)
@test target_scitype(model) == AbstractVector{<:Continuous}
predict(fit!(machine(model, X, y, w), verbosity=-1), X);

# unsupervised:
@from_network machine(Unsupervised(), Xs; transform=W) begin
    mutable struct CompositeX2
        one_hot_enc=hot
    end
end
model = CompositeX2()
transform(fit!(machine(model, X), verbosity=-1), X)


# second supervised test:
fea = FeatureSelector()
feaM = machine(fea, Xs)
G = transform(feaM, Xs)
hotM = machine(hot, G)
H = transform(hotM, G)
elm = DecisionTreeClassifier()
elmM = machine(elm, H, ys)
yhat = predict(elmM, H)

@from_network machine(Probabilistic(), Xs, ys; predict=yhat) begin
    mutable struct CompositeX3
        selector=fea
        one_hot=hot
        tree=elm
    end
end
model = CompositeX3()
y = coerce(y, Multiclass)
@test predict(fit!(machine(model, X, y), verbosity=-1), X) isa
    AbstractVector{<:UnivariateFinite}

# yet more examples:
x1 = map(n -> mod(n,3), rand(rng, UInt8, 100)) |> categorical;
x2 = randn(rng, 100);
X = (x1=x1, x2=x2);
y = x2.^2;

Xs = source(X)
ys = source(y)
z = log(ys)
stand = UnivariateStandardizer()
standM = machine(stand, z)
u = transform(standM, z)
hot = OneHotEncoder()
hotM = machine(hot, Xs)
W = transform(hotM, Xs)
knn = KNNRegressor()
knnM = machine(knn, W, u)
oak = DecisionTreeRegressor()
oakM = machine(oak, W, u)
uhat = 0.5*(predict(knnM, W) + predict(oakM, W))
zhat = inverse_transform(standM, uhat)
yhat = exp(zhat)

mach = machine(Deterministic(), Xs, ys; predict=yhat)

@from_network mach begin
    mutable struct Composite10
        knn_rgs::KNNRegressor=knn
        one_hot_enc=hot
    end
end

model_ = Composite10()

mach = machine(model_, X, y)

@test_model_sequence(fit_only!(mach),
                     [(:train, model_), (:train, stand), (:train, hot),
                      (:train, knn), (:train, oak)],
                     [(:train, model_), (:train, hot), (:train, stand),
                      (:train, knn), (:train, oak)],
                     [(:train, model_), (:train, stand), (:train, hot),
                      (:train, oak), (:train, knn)],
                     [(:train, model_), (:train, hot), (:train, stand),
                      (:train, oak), (:train, knn)])

model_.knn_rgs.K = 55
knn = model_.knn_rgs
@test_model_sequence(fit_only!(mach),
                     [(:update, model_), (:skip, stand), (:skip, hot),
                      (:update, knn), (:skip, oak)],
                     [(:update, model_), (:skip, hot), (:skip, stand),
                      (:update, knn), (:skip, oak)],
                     [(:update, model_), (:skip, stand), (:skip, hot),
                      (:skip, oak), (:update, knn)],
                     [(:update, model_), (:skip, hot), (:skip, stand),
                      (:skip, oak), (:update, knn)])

@test MLJBase.tree(mach.fitresult.predict).arg1.arg1.arg1.arg1.model.K == 55

# check data anonymity:
@test all(x->(x===nothing),
          [s.data for s in sources(mach.fitresult.predict)])


multistand = Standardizer()
multistandM = machine(multistand, W)
W2 = transform(multistandM, W)

mach = machine(Unsupervised(), Xs; transform=W2)

@from_network mach begin
    mutable struct MyTransformer
        one_hot=hot
    end
end

model_ = MyTransformer()

mach = machine(model_, X)
@test_model_sequence fit_only!(mach) [(:train, model_),
                                      (:train, hot), (:train, multistand)]

model_.one_hot.drop_last=true
hot = model_.one_hot
@test_model_sequence fit_only!(mach) [(:update, model_),
                                      (:update, hot), (:train, multistand)]

# check nested fitted_params:
FP = MLJBase.fitted_params(mach)
@test keys(FP) == (:one_hot, :machines, :fitted_params_given_machine)
@test Set(FP.one_hot.fitresult.all_features) == Set(keys(X))

# check data anomynity:
@test all(x->(x===nothing),
          [s.data for s in sources(mach.fitresult.transform)])

transform(mach, X);


## TEST MACRO-EXPORTED SUPERVISED NETWORK WITH SAMPLE WEIGHTS

rng = StableRNG(56161)
N = 500
X = (x = rand(rng, 3N), );
y = categorical(rand(rng, "abc", 3N));
# define class weights :a, :b, :c in ration 2:4:1
w = map(y) do η
    if η == 'a'
        return 2
    elseif η == 'b'
        return 4
    else
        return 1
    end
end;
Xs = source(X)
ys = source(y)
ws = source(w)

standM = machine(Standardizer(), Xs)
W = transform(standM, Xs)

rgs = ConstantClassifier() # supports weights
rgsM = machine(rgs, W, ys, ws)
yhat = predict(rgsM, W)

fit!(yhat, verbosity=0)
fit!(yhat, rows=1:div(N,2), verbosity=0)
yhat(rows=1:div(N,2));

mach = machine(Probabilistic(), Xs, ys, ws; predict=yhat)

@from_network mach begin
    mutable struct MyComposite
        regressor=rgs
    end
    supports_weights=true
end

my_composite = MyComposite()
@test MLJBase.supports_weights(my_composite)
mach = fit!(machine(my_composite, X, y), verbosity=0)
Xnew = selectrows(X, 1:div(N,2))
predict(mach, Xnew)[1]
posterior = predict(mach, Xnew)[1]

# "posterior" is roughly uniform:
@test abs(pdf(posterior, 'b')/(pdf(posterior, 'a'))  - 1) < 0.15
@test abs(pdf(posterior, 'b')/(pdf(posterior, 'c'))  - 1) < 0.15

# now add weights:
mach = fit!(machine(my_composite, X, y, w), rows=1:div(N,2), verbosity=0)
posterior = predict(mach, Xnew)[1]

# "posterior" is skewed appropriately in weighted case:
@test abs(pdf(posterior, 'b')/(2*pdf(posterior, 'a'))  - 1) < 0.15
@test abs(pdf(posterior, 'b')/(4*pdf(posterior, 'c'))  - 1) < 0.19

# composite with no fields:
mach = machine(Probabilistic(), Xs, ys, ws; predict=yhat)
@from_network mach begin
    struct CompositeWithNoFields
    end
end
composite_with_no_fields = CompositeWithNoFields()
mach = fit!(machine(composite_with_no_fields, X, y), verbosity=0)


## EXPORTING A TRANSFORMER WITH PREDICT AND TRANSFORM

# A dummy clustering model:
mutable struct DummyClusterer <: Unsupervised
    n::Int
end
DummyClusterer(; n=3) = DummyClusterer(n)
function MLJBase.fit(model::DummyClusterer, verbosity::Int, X)
    Xmatrix = Tables.matrix(X)
    n = min(size(Xmatrix, 2), model.n)
    centres = Xmatrix[1:n, :]
    levels = categorical(1:n)
    report = (centres=centres,)
    fitresult = levels
    return fitresult, nothing, report
end
MLJBase.transform(model::DummyClusterer, fitresult, Xnew) =
    selectcols(Xnew, 1:length(fitresult))
MLJBase.predict(model::DummyClusterer, fitresult, Xnew) =
    [fill(fitresult[1], nrows(Xnew))...]

N = 20
X = (a = rand(N), b = categorical(rand("FM", N)))

Xs = source(X)
W = transform(machine(OneHotEncoder(), Xs), Xs)
clust = DummyClusterer(n=2)
m = machine(clust, W)
yhat = predict(m, W)
Wout = transform(m, W)
foo = first(yhat)
mach = machine(Unsupervised(), Xs; predict=yhat, transform=Wout, foo=foo)

@from_network mach begin
    mutable struct WrappedClusterer
        clusterer::Unsupervised = clust
    end
    input_scitype = Table(Continuous,Multiclass)
end

model = WrappedClusterer()
mach = fit!(machine(model, X), verbosity=0)
fit!(yhat)
@test predict(mach, X) == yhat()
@test transform(mach, X).a ≈ Wout().a
Θ = fitted_params(mach)
@test Θ.foo == yhat() |> first


## EXPORTING A STATIC LEARNING NETWORK (NO TRAINING ARGUMENTS)

age = [23, 45, 34, 25, 67]
X = (age = age,
     gender = categorical(['m', 'm', 'f', 'm', 'f']))

struct MyStaticTransformer <: Static
    ftr::Symbol
end

MLJBase.transform(transf::MyStaticTransformer, verbosity, X) =
    selectcols(X, transf.ftr)

Xs = source()
W = transform(machine(MyStaticTransformer(:age)), Xs)
Z = 2*W

@from_network machine(Static(), Xs; transform=Z) begin
    struct NoTraining
    end
end

mach = fit!(machine(NoTraining()), verbosity=0)
@test transform(mach, X) == 2*X.age
 

## TESTINGS A STACK AND IN PARTICULAR FITTED_PARAMS

folds(data, nfolds) =
    partition(1:nrows(data), (1/nfolds for i in 1:(nfolds-1))...);

model1 = RidgeRegressor()
model2 = KNNRegressor(K=1)
judge = KNNRegressor(K=1)

X = source()
y = source()

folds(X::AbstractNode, nfolds) = node(XX->folds(XX, nfolds), X)
MLJBase.restrict(X::AbstractNode, f::AbstractNode, i) =
    node((XX, ff) -> restrict(XX, ff, i), X, f);
MLJBase.corestrict(X::AbstractNode, f::AbstractNode, i) =
    node((XX, ff) -> corestrict(XX, ff, i), X, f);

f = folds(X, 3)

m11 = machine(model1, corestrict(X, f, 1), corestrict(y, f, 1))
m12 = machine(model1, corestrict(X, f, 2), corestrict(y, f, 2))
m13 = machine(model1, corestrict(X, f, 3), corestrict(y, f, 3))

y11 = predict(m11, restrict(X, f, 1));
y12 = predict(m12, restrict(X, f, 2));
y13 = predict(m13, restrict(X, f, 3));

m21 = machine(model2, corestrict(X, f, 1), corestrict(y, f, 1))
m22 = machine(model2, corestrict(X, f, 2), corestrict(y, f, 2))
m23 = machine(model2, corestrict(X, f, 3), corestrict(y, f, 3))

y21 = predict(m21, restrict(X, f, 1));
y22 = predict(m22, restrict(X, f, 2));
y23 = predict(m23, restrict(X, f, 3));

y1_oos = vcat(y11, y12, y13);
y2_oos = vcat(y21, y22, y23);

X_oos = MLJBase.table(hcat(y1_oos, y2_oos))

m_judge = machine(judge, X_oos, y)

m1 = machine(model1, X, y)
m2 = machine(model2, X, y)

y1 = predict(m1, X);
y2 = predict(m2, X);

X_judge = MLJBase.table(hcat(y1, y2))
yhat = predict(m_judge, X_judge)

@from_network machine(Deterministic(), X, y; predict=yhat) begin
    mutable struct MyStack
        regressor1=model1
        regressor2=model2
        judge=judge
    end
end

my_stack = MyStack()
X, y = make_regression(18, 2)
mach = machine(my_stack, X, y)
fit!(mach, verbosity=0)

fp = fitted_params(mach)
@test keys(fp.judge) == (:tree,)
@test length(fp.regressor1) == 4
@test length(fp.regressor2) == 4
@test keys(fp.regressor1[1]) == (:coefficients, :intercept)
@test keys(fp.regressor2[1]) == (:tree,)


## ISSUE #377

target_stand = Standardizer()
stand = Standardizer()
rgs = KNNRegressor()

X = source()
y = source()

mach1 = machine(target_stand, y)
z = transform(mach1, y)
mach2 = machine(stand, X)
W = transform(mach2, X)
mach3 = machine(rgs, W, z)
zhat = predict(mach3, W)
yhat = inverse_transform(mach1, zhat)

@from_network machine(Deterministic(), X, y; predict=yhat) begin
    mutable struct CompositeA
        rgs=rgs
        stand=stand
        target=target_stand
    end
end

X, y = make_regression(20, 2);
model = CompositeA(stand=stand, target=stand)
mach = machine(model, X, y)
@test_logs((:error, r"The hyper"),
           (:error, r"Problem"),
           (:info, r"Running"),
           (:info, r"Type checks okay"),
           @test_throws(ArgumentError,
                        fit!(mach, verbosity=-1)))

end

true
