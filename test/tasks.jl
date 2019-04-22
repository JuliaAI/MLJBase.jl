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

task = SupervisedTask(data=XX, target=[:Crim, :Zn], is_probabilistic=true, ignore=:Dis)
y = y_(task)
X = X_(task);
t = MLJBase.schema(X);
@test collect(t.names) == filter(allnames) do ftr
    !(ftr in [:Crim, :Zn, :Dis])
end
y1 = [(XX.Crim[i], XX.Zn[i]) for i in 1:length(y)]
@test y == y1


task = SupervisedTask(data=XX, target=:Crim, is_probabilistic=true, ignore=[:Dis, :Rm])
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

# single feature for input:
task = UnsupervisedTask(data=XX[:,[:Crim,]])
@test task.X == XX.Crim
task = SupervisedTask(data=XX[:,[:Crim, :Zn]], target=:Zn, is_probabilistic=true)
@test task.X == XX.Crim
@test task.y == XX.Zn

end # module
true
