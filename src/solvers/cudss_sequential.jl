"""
    CUDSSSequentialSolver

Solves batch of linear systems sequentially using CUDSS.
"""
@kwdef struct CUDSSSequentialSolver end

batch_type(::CUDSSSequentialSolver) = BatchLinearSystemGPU
name(::CUDSSSequentialSolver) = "CUDSS_Sequential"

function solve!(B::BatchLinearSystemGPU, s::CUDSSSequentialSolver; nsolve=1)
    K = length(B.As)  # batch size

    fac = cudss_matrix_type(matrix_type(B))
    view = cudss_matrix_view(matrix_view(B))
    
    for i in 1:K
        CUDA.@sync begin
            solver = CudssSolver(B.As[i], fac, view)
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
