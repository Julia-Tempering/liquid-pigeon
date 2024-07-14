using Pigeons
using Random
using StatsFuns

struct MyLogPotential
    n_trials::Int
    n_successes::Int
end

function (log_potential::MyLogPotential)(x)
    p1, p2 = x
    if !(0 < p1 < 1) || !(0 < p2 < 1)
        return -Inf64
    end
    p = p1 * p2

    println("n_trails:$(log_potential.n_trials)")
    println("n_succ:$(log_potential.n_successes)")
    return StatsFuns.binomlogpdf(log_potential.n_trials, p, log_potential.n_successes)

end

# e.g.:
my_log_potential = MyLogPotential(100, 50)
my_log_potential([0.5, 0.5])

Pigeons.initialization(::MyLogPotential, ::AbstractRNG, ::Int) = [0.5, 0.5]
pt = pigeons(
        target = MyLogPotential(100, 50),
        reference = MyLogPotential(0, 0),
        n_rounds = 1
    )