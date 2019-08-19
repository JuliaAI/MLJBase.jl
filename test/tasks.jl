module TestTasks

# using Revise
using Test
using MLJBase

XX = (Crim = [0.00632, 0.02731, 0.02729],
      Zn = [18.0, 0.0, 0.0],
      Indus = [2.31, 7.07, 7.07],
      Chas = [0, 0, 0],
      NOx = [0.538, 0.469, 0.469],
      Rm = [6.575, 6.421, 7.185],
      Age = [65.2, 78.9, 61.1],
      Dis = [4.09, 4.9671, 4.9671],
      Rad = [1.0, 2.0, 2.0],
      Tax = [296.0, 242.0, 242.0],
      PTRatio = [15.3, 17.8, 17.8],
      Black = [396.9, 396.9, 392.83],
      LStat = [4.98, 9.14, 4.03],
      MedV = [24.0, 21.6, 34.7],)

allnames = collect(MLJBase.schema(XX).names)

task = SupervisedTask(data=XX, target=[:Crim, :Zn], is_probabilistic=true, ignore=:Dis)

y = y_(task);
X = X_(task);
t = MLJBase.schema(X);
@test collect(t.names) == filter(allnames) do ftr
    !(ftr in [:Crim, :Zn, :Dis])
end
y1 = [(XX.Crim[i], XX.Zn[i]) for i in 1:length(y)];
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
task = UnsupervisedTask(data=(Crim=XX.Crim,))
@test task.X == XX.Crim
task = SupervisedTask(data=MLJBase.selectcols(XX, [:Crim, :Zn]), target=:Zn, is_probabilistic=true)
@test task.X == XX.Crim
@test task.y == XX.Zn

end # module
true
