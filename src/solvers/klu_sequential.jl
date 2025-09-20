using Base.Threads
using KLU

"""
    KLUSequentialSolver

Solves batch of linear systems sequentially using CPU-based LU solver.
"""
@kwdef struct KLUSequentialSolver
end

function solve!(B::BatchLinearSystemCPU, s::KLUSequentialSolver; nsolve=1)
    K = length(B.As)  # batch size

    @threads for i in 1:K
        F = klu(B.As[i])
        for _ in 1:nsolve
            ldiv!(B.xs[i], F, B.bs[i])
        end
    end

    return nothing
end
