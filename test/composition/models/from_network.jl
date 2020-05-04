module TestFromComposite

using Test
using MLJBase
using ..Models
using ..TestUtilities
using CategoricalArrays
import Random.seed!
seed!(1234)

@load KNNRegressor

N = 50
Xin = (a=rand(N), b=rand(N), c=rand(N));
yin = rand(N);

train, test = partition(eachindex(yin), 0.7);
Xtrain = MLJBase.selectrows(Xin, train);
ytrain = yin[train];

ridge_model = FooBarRegressor(lambda=0.1)
selector_model = FeatureSelector()

@load DecisionTreeRegressor
@load DecisionTreeClassifier

## FROM_NETWORK_PREPROCESS

# supervised:
Xs = source(nothing)
ys = source(nothing, kind=:target)
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

ex = Meta.parse("Composite(knn_rgs=knn, one_hot_enc=hot) <= yhat")
modeltype_ex, fieldname_exs, model_exs, N_ex, kind, trait_dic =
    MLJBase.from_network_preprocess(TestFromComposite, ex)
@test modeltype_ex == :Composite
@test fieldname_exs == [:knn_rgs, :one_hot_enc]
@test model_exs == [:knn, :hot]
@test N_ex == :yhat
@test kind == :DeterministicComposite
@test !(haskey(trait_dic, :supports_weights)) ||
    !trait_dic[:supports_weights]

# supervised with sample weights:
ws = source(kind=:weights)
knnM = machine(knn, W, u, ws)
uhat = 0.5*(predict(knnM, W) + predict(oakM, W))
zhat = inverse_transform(standM, uhat)
yhat = exp(zhat)
ex = Meta.parse("Composite(knn_rgs=knn, one_hot_enc=hot) <= yhat")
modeltype_ex, fieldname_exs, model_exs, N_ex, kind, trait_dic =
    MLJBase.from_network_preprocess(TestFromComposite, ex)
@test trait_dic[:supports_weights]

# unsupervised:
ex = Meta.parse("Composite(one_hot_enc=hot) <= W")
modeltype_ex, fieldname_exs, model_exs, N_ex, kind, trait_dic =
    MLJBase.from_network_preprocess(TestFromComposite, ex)
@test modeltype_ex == :Composite
@test fieldname_exs == [:one_hot_enc,]
@test model_exs == [:hot,]
@test N_ex == :W
@test kind == :UnsupervisedComposite
@test !(haskey(trait_dic, :supports_weights))

# second supervised test:
fea = FeatureSelector()
feaM = machine(fea, Xs)
G = transform(feaM, Xs)
hotM = machine(hot, G)
H = transform(hotM, G)
elm = DecisionTreeClassifier()
elmM = machine(elm, H, ys)
yhat = predict(elmM, H)

ex = Meta.parse("Composite(selector=fea,one_hot=hot,tree=elm) <= yhat")
modeltype_ex, fieldname_exs, model_exs, N_ex, kind, trait_dic =
    MLJBase.from_network_preprocess(TestFromComposite,
                                ex, :(prediction_type=:probabilistic))
@test modeltype_ex == :Composite
@test fieldname_exs == [:selector, :one_hot, :tree]
@test model_exs == [:fea, :hot, :elm]
@test N_ex == :yhat
@test kind == :ProbabilisticComposite
@test !(haskey(trait_dic, :supports_weights)) ||
    !trait_dic[:supports_weights]

ex = Meta.parse("45")
@test_throws(ArgumentError,
             MLJBase.from_network_preprocess(TestFromComposite,
                                         ex, :(prediction_type=:probabilistic)))

ex = Meta.parse("Composite(elm=elm) << yhat")
@test_throws(ArgumentError,
             MLJBase.from_network_preprocess(TestFromComposite,
                                         ex, :(prediction_type=:probabilistic)))

ex = Meta.parse("45")
@test_throws(ArgumentError,
             MLJBase.from_network_preprocess(TestFromComposite,
                                         ex, :(prediction_type=:probabilistic)))

ex = Meta.parse("45 <= yhat")
@test_throws(ArgumentError,
             MLJBase.from_network_preprocess(TestFromComposite,
                                         ex, :(prediction_type=:probabilistic)))
ex = Meta.parse("Comp(elm=45) <= yhat")
@test_throws(ArgumentError,
             MLJBase.from_network_preprocess(TestFromComposite,
                                         ex, :(prediction_type=:probabilistic)))

ex = Meta.parse("Comp(elm=>elm) <= yhat")
@test_throws(ArgumentError,
             MLJBase.from_network_preprocess(TestFromComposite,
                                         ex, :(prediction_type=:probabilistic)))

ex = Meta.parse("Comp(34=elm) <= yhat")
@test_throws(ArgumentError,
             MLJBase.from_network_preprocess(TestFromComposite,
                                         ex, :(prediction_type=:probabilistic)))

ex = Meta.parse("Comp(elm=elm) <= 45")
@test_throws(ArgumentError,
             MLJBase.from_network_preprocess(TestFromComposite,
                                         ex, :(prediction_type=:probabilistic)))

z = vcat(ys, ys)
ex = Meta.parse("Comp() <= z")
@test_throws(ArgumentError,
             MLJBase.from_network_preprocess(TestFromComposite,ex))

X2s = source(nothing)
# z = @test_logs (:warn, r"^A node ref") vcat(Xs, X2s)
z = vcat(Xs, X2s)
ex = Meta.parse("Comp() <= z")
@test_throws(ArgumentError,
             MLJBase.from_network_preprocess(TestFromComposite, ex))


y2s = source(nothing, kind=:target)
# z = @test_logs (:warn, r"^A node ref") vcat(ys, y2s, Xs)
z = vcat(ys, y2s, Xs)
@test_throws Exception z(Xs())

ex = Meta.parse("Comp() <= z")
@test_throws(ArgumentError,
             MLJBase.from_network_preprocess(TestFromComposite, ex))

ex = Meta.parse("Composite(one_hot_enc=hot) <= W")
@test_throws(ArgumentError,
             MLJBase.from_network_preprocess(TestFromComposite,
                                         ex, :(prediction_type=:probabilistic)))

## TEST MACRO-EXPORTED SUPERVISED NETWORK
# (CANNOT WRAP IN @testset)

x1 = map(n -> mod(n,3), rand(UInt8, 100)) |> categorical;
x2 = randn(100);
X = (x1=x1, x2=x2);
y = x2.^2;

Xs = source(X)
ys = source(y, kind=:target)
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

fit!(yhat)

# test nested reporting:
r = MLJBase.report(yhat)
d = r.report_given_machine
ms = machines(yhat)
@test ms == r.machines
@test all(mach -> report(mach) == d[mach], ms)

hot2 = deepcopy(hot)
knn2 = deepcopy(knn)
ys2 = source(nothing, kind=:target)

# duplicate a network:
yhat2 = replace(yhat, hot=>hot2, knn=>knn2,
                ys=>source(ys.data, kind=:target);
                empty_unspecified_sources=true)
@test isempty(sources(yhat2, kind=:input)[1])
yhat2 = @test_logs((:warn, r"No replacement"),
                   replace(yhat, hot=>hot2, knn=>knn2,
                           ys=>source(ys.data, kind=:target)))
@test !isempty(sources(yhat2, kind=:input)[1])

# pickout the newly created machines:
standM2 = machines(yhat2, stand) |> first
oakM2 = machines(yhat2, oak) |> first
knnM2 = machines(yhat2, knn) |> first
hotM2 = machines(yhat2, hot) |> first

@test_mach_sequence(fit!(yhat2),
                   [(:train, standM2), (:train, hotM2),
                     (:train, knnM2), (:train, oakM2)],
                   [(:train, hotM2), (:train, standM2),
                    (:train, knnM2), (:train, oakM2)],
                   [(:train, standM2), (:train, hotM2),
                    (:train, oakM2), (:train, knnM2)],
                   [(:train, hotM2), (:train, standM2),
                    (:train, oakM2), (:train, knnM2)])

@test length(MLJBase.machines(yhat)) == length(MLJBase.machines(yhat2))
@test models(yhat) == models(yhat2)
@test sources(yhat) == sources(yhat2)
@test MLJBase.tree(yhat) == MLJBase.tree(yhat2)
@test yhat() ≈ yhat2()

# this change should trigger retraining of all machines except the
# univariate standardizer:
hot2.drop_last = true
@test_mach_sequence(fit!(yhat2),
               [(:skip, standM2), (:update, hotM2),
                (:train, knnM2), (:train, oakM2)],
               [(:update, hotM2), (:skip, standM2),
                (:train, knnM2), (:train, oakM2)],
               [(:skip, standM2), (:update, hotM2),
                (:train, oakM2), (:train, knnM2)],
               [(:update, hotM2), (:skip, standM2),
                (:train, oakM2), (:train, knnM2)])

# export a supervised network:
model_ = @from_network Composite(knn_rgs=knn, one_hot_enc=hot) <= yhat

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

@test MLJBase.tree(mach.fitresult).arg1.arg1.arg1.arg1.model.K == 55

# check data anonymity:
@test all(x->(x===nothing), [s.data for s in sources(mach.fitresult)])


## TEST MACRO-EXPORTED UNSUPERVISED NETWORK
# (CANNOT WRAP IN @testset)

multistand = Standardizer()
multistandM = machine(multistand, W)
W2 = transform(multistandM, W)
model_ = @from_network Transf(one_hot=hot) <= W2
mach = machine(model_, X)
@test_model_sequence fit_only!(mach) [(:train, model_),
                                      (:train, hot), (:train, multistand)]

model_.one_hot.drop_last=true
hot = model_.one_hot
@test_model_sequence fit_only!(mach) [(:update, model_),
                                      (:update, hot), (:train, multistand)]

# check nested fitted_params:
FP = MLJBase.fitted_params(mach)
d = FP.fitted_params_given_machine
ms = FP.machines
@test all(mach -> fitted_params(mach) == d[mach], ms)

# check data anomynity:
@test all(x->(x===nothing), [s.data for s in sources(mach.fitresult)])

transform(mach, X);


## TEST MACRO-EXPORTED SUPERVISED NETWORK WITH SAMPLE WEIGHTS

seed!(1234)
N = 100
X = (x = rand(3N), );
y = categorical(rand("abc", 3N));
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
ys = source(y, kind=:target)
ws = source(w, kind=:weights)

standM = machine(Standardizer(), Xs)
W = transform(standM, Xs)

rgs = ConstantClassifier() # supports weights
rgsM = machine(rgs, W, ys, ws)
yhat = predict(rgsM, W)

fit!(yhat)
fit!(yhat, rows=1:div(N,2))
yhat(rows=1:div(N,2));

composite = @from_network Composite3(regressor=rgs) <= yhat

@test MLJBase.supports_weights(composite)
mach = fit!(machine(composite, X, y))
Xnew = selectrows(X, 1:div(N,2))
predict(mach, Xnew)[1]
posterior = predict(mach, Xnew)[1]

# "posterior" is roughly uniform:
@test abs(pdf(posterior, 'b')/(pdf(posterior, 'a'))  - 1) < 0.15
@test abs(pdf(posterior, 'b')/(pdf(posterior, 'c'))  - 1) < 0.15

# now add weights:
mach = fit!(machine(composite, X, y, w), rows=1:div(N,2))
posterior = predict(mach, Xnew)[1]

# "posterior" is skewed appropriately in weighted case:
@test abs(pdf(posterior, 'b')/(2*pdf(posterior, 'a'))  - 1) < 0.15
@test abs(pdf(posterior, 'b')/(4*pdf(posterior, 'c'))  - 1) < 0.15

composite_with_no_fields = @from_network CompositeWithNoFields() <= yhat
mach = fit!(machine(composite_with_no_fields, X, y))


end
true
