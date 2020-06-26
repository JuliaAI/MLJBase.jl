module TestPipelines

using MLJBase
using ..Models
using Test
using Statistics
using MLJScientificTypes
using StableRNGs

rng = StableRNG(698790187)

@load KNNRegressor

NN = 7
X = MLJBase.table(rand(rng,NN, 3));
y = 2X.x1 - X.x2 + 0.05*rand(rng,NN);
Xs = source(X); ys = source(y)

broadcast_mode(v) = mode.(v)

t = MLJBase.table
m = MLJBase.matrix
f = FeatureSelector()
h = OneHotEncoder()
k = KNNRegressor()
u = UnivariateStandardizer()
c = ConstantClassifier()

# build a linear network for training:
N = MLJBase.linear_learning_network(Xs, ys, nothing, u, nothing, f, k)
fM = machine(f, Xs)
Xt = transform(fM, Xs)

# build the same network by hand:
uM = machine(u, ys)
yt = transform(uM, ys)
kM = machine(k, Xt, yt)
zhat = predict(kM, Xt)

k.K = 3; f.features = [:x3,]
fit!(N)
hand_built = N();


## SIMPLE SUPERVISED PIPELINE WITH TARGET TRANSFORM

# test a simple pipeline prediction agrees with prediction of
# hand-built learning network built earlier:
p = @pipeline(Pipe(sel=FeatureSelector(), knn=KNNRegressor(),
                   target=UnivariateStandardizer()))
p.knn.K = 3; p.sel.features = [:x3,]
mach = machine(p, X, y)
fit!(mach)
@test MLJBase.tree(mach.fitresult.predict).arg1.model.K == 3
MLJBase.tree(mach.fitresult.predict).arg1.arg1.model.features == [:x3, ]
@test predict(mach, X) ≈ hand_built

# test a simple probabilistic classifier pipeline:
X = MLJBase.table(rand(rng,7,3))
y = categorical(collect("ffmmfmf"))
Xs = source(X)
ys = source(y)
p = @pipeline(Pipe21(hot=OneHotEncoder(),
                    cnst=ConstantClassifier()),
              prediction_type=:probabilistic)
mach = machine(p, X, y)
fit!(mach)
@test p isa ProbabilisticComposite
pdf(predict(mach, X)[1], 'f') ≈ 4/7

# test a simple deterministic classifier pipeline:
X = MLJBase.table(rand(rng,7,3))
y = categorical(collect("ffmmfmf"))
Xs = source(X)
ys = source(y)
p = @pipeline(Piper3(hot=OneHotEncoder(), cnst=ConstantClassifier(),
                     broadcast_mode))
mach = machine(p, X, y)
fit!(mach)
@test predict(mach, X) == fill('f', 7)

# test pipelines with weights:
w = map(y) do η
    η == 'm' ? 100 : 1
end
mach = machine(p, X, y, w)
fit!(mach)
@test predict(mach, X) == fill('m', 7)

# test a pipeline with static transformation of target:
NN = 100
X = (x1=rand(rng,NN), x2=rand(rng,NN), x3=categorical(rand(rng,"abc", NN)));
y = 1000*abs.(2X.x1 - X.x2 + 0.05*rand(rng,NN))
# by hand:
Xs =source(X); ys = source(y)
hot = OneHotEncoder()
hot_=machine(hot, Xs)
W = transform(hot_, Xs)
sel = FeatureSelector(features=[:x1,:x3__a])
sel_ = machine(sel, W)
Wsmall = transform(sel_, W)
z = log(ys)
knn = KNNRegressor(K=4)
knn_ = machine(knn, Wsmall, z)
zhat = predict(knn_, Wsmall)
yhat = exp(zhat)
fit!(yhat)
pred1 = yhat()
# with pipeline:
p = @pipeline Pipe4(hot=OneHotEncoder(),
                    sel=FeatureSelector(),
                    knn=KNNRegressor(),
                    target=v->log.(v),
                    inverse=v->exp.(v))
p.sel.features = [:x1, :x3__a]
p.knn.K = 4
p_ = machine(p, X, y)
fit!(p_)
pred2 = predict(p_, X)
@test pred1 ≈ pred2

# and another:
age = [23, 45, 34, 25, 67]
X = (age = age,
     gender = categorical(['m', 'm', 'f', 'm', 'f']))
height = [67.0, 81.5, 55.6, 90.0, 61.1]
p = @pipeline Pipe9(X -> coerce(X, :age=>Continuous),
                    hot = OneHotEncoder(),
                    knn = KNNRegressor(K=3),
                    target = UnivariateStandardizer())
fit!(machine(p, X, height))


# STATIC TRANSFORMERS IN PIPELINES

struct MyTransformer <: Static
    ftr::Symbol
end

MLJBase.transform(transf::MyTransformer, verbosity, X) =
    selectcols(X, transf.ftr)

p99 = @pipeline Pipe99(X -> coerce(X, :age=>Continuous),
                       hot = OneHotEncoder(),
                       selector = MyTransformer(:age))

mach  = machine(p99, X) |> fit!

@test transform(mach, X) == float.(X.age)

end
true

