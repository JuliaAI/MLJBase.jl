module TestTasks

# using Revise
using Test
using MLJBase
using DataFrames

task = load_boston();

@test X_and_y(task) == task()
@test task() == (X_(task), y_(task))

XX = X_(task);
allnames = collect(MLJBase.schema(XX).names)

task = SupervisedTask(data=XX, targets=[:Crim, :Zn], is_probabilistic=true, ignore=:Dis)
y = y_(task);
X = X_(task);
s = MLJBase.schema(y);
t = MLJBase.schema(X);
@test collect(s.names) == [:Crim, :Zn]
@test collect(t.names) == filter(allnames) do ftr
    !(ftr in [:Crim, :Zn, :Dis])
end

task = SupervisedTask(data=XX, targets=:Crim, is_probabilistic=true, ignore=[:Dis, :Rm])
y = y_(task);
X = X_(task);
t = MLJBase.schema(X)
@test collect(t.names) == filter(allnames) do ftr
    !(ftr in [:Crim, :Dis, :Rm])
end

task = UnsupervisedTask(data=XX, ignore=[:Dis, :Rm])
X = task();
t = MLJBase.schema(X)
@test collect(t.names) == filter(allnames) do ftr
    !(ftr in [:Dis, :Rm])
end

task = UnsupervisedTask(data=XX, ignore=:Rm)
X = X_(task);
t = MLJBase.schema(X)
@test collect(t.names) == filter(allnames) do ftr
    !(ftr in [:Rm, ])
end


end # module
true
