"""
    CUDSSSequentialSolver

Solves batch of linear systems sequentially using CUDSS.
"""
@kwdef struct CUDSSSequentialSolver
    fac::String="G"
    view::Char='F'
end

batch_type(::CUDSSSequentialSolver) = BatchLinearSystemGPU
name(::CUDSSSequentialSolver) = "CUDSS_Sequential"

function solve!(B::BatchLinearSystemGPU, s::CUDSSSequentialSolver; nsolve=1)
    K = length(B.As)  # batch size
    
    for i in 1:K
        CUDA.@sync begin
            solver = CudssSolver(B.As[i], s.fac, s.view)
            NVTX.@range "analysis" begin
                cudss("analysis", solver, B.xs[i], B.bs[i])
            end
            NVTX.@range "factorization" begin
                cudss("factorization", solver, B.xs[i], B.bs[i])
            end
            NVTX.@range "solve" begin
                for _ in 1:nsolve
                    cudss("solve", solver, B.xs[i], B.bs[i])
                end
            end
        end
    end

    return nothing
end
