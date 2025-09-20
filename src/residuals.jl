
"""
    compute_residuals(B; to_cpu=true)

Compute residuals `rᵢ = bᵢ - Aᵢ xᵢ`.

Computations are performed on the same device as where the data resides.
If `to_cpu` is set to true, the result is collected to host (cpu).
"""
function compute_residuals(B::BatchLinearSystemCPU; to_cpu=true)
    K = length(B.As)
    Rs = [
        (B.bs[i] - B.As[i] * B.xs[i])
        for i in 1:K
    ]

    return Rs
end

function compute_residuals(B::BatchLinearSystemGPU; to_cpu=true)
    K = length(B.As)

    Rs_gpu = [
        (B.bs[i] - B.As[i] * B.xs[i])
        for i in 1:K
    ]

    if to_cpu
        return [collect(R) for R in Rs_gpu]
    else
        return Rs_gpu        
    end
end

function compute_residuals(B::UniformBatchLinearSystemGPU{T}; to_cpu=true) where{T}
    k = B.batch_size
    m = B.nrows
    n = B.ncols
    nnzA, q = divrem(length(B.nzVal), B.batch_size)
    q == 0 || error("Inconsistent size for nzVal")

    Rs_gpu = CuVector{T}[]

    for i in 1:k
        nz = B.nzVal[1 + (i-1) * nnzA : i * nnzA]
        A = CuSparseMatrixCSR{T,Cint}(B.rowPtr, B.colVal, nz, (n,n))
        b = B.b_dat[1 + (i-1) * n : i * n]
        x = B.x_dat[1 + (i-1) * n : i * n]
        r = b - A * x
        push!(Rs_gpu, r)
    end

    if to_cpu
        return [collect(R) for R in Rs_gpu]
    else
        return Rs_gpu        
    end
end
