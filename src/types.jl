using LinearAlgebra
using SparseArrays

using CUDA
using CUDA.CUSPARSE
using CUDSS
using NVTX

struct BatchLinearSystemCPU{T}
    As::Vector{SparseMatrixCSC{T,Int}} # Vector of sparse matrices (LHS)
    bs::Vector{Vector{T}}              # Vector of (dense) RHS
    xs::Vector{Vector{T}}              # Vector of (dense) solutions
end

function BatchLinearSystemCPU(As::Vector{SparseMatrixCSC{T,Int}}, bs::Vector{Vector{T}}) where T
    length(As) == length(bs) || throw(DimensionMismatch("Length of As and bs must be equal"))
    xs = [zeros(T, size(b)) for b in bs]
    return BatchLinearSystemCPU{T}(As, bs, xs)
end

struct BatchLinearSystemGPU{T,Ti}
    As::Vector{CUDA.CUSPARSE.CuSparseMatrixCSR{T,Ti}}  # Vector of sparse matrices (LHS)
    bs::Vector{CuVector{T}}                            # Vector of dense RHS
    xs::Vector{CuVector{T}}                            # Vector of dense solution
end

function BatchLinearSystemGPU(B::BatchLinearSystemCPU{T}) where T
    K = length(B.As)  # batch size

    # Move data to GPU
    As_gpu = [CUDA.CUSPARSE.CuSparseMatrixCSR{T}(B.As[i]) for i in 1:K]
    bs_gpu = [CUDA.CuArray(B.bs[i]) for i in 1:K]
    xs_gpu = [CUDA.zeros(T, size(bs_gpu[i])) for i in 1:K]
    
    return BatchLinearSystemGPU{T,Cint}(As_gpu, bs_gpu, xs_gpu)
end

struct UniformBatchLinearSystemGPU{T}
    batch_size::Int
    nrows::Int
    ncols::Int

    # LHS data
    rowPtr::CuVector{Cint}
    colVal::CuVector{Cint}
    nzVal::CuVector{T}  # size nnz * batch_size

    # RHS data
    x_dat::CuVector{T}  # size ncols * batch_size

    # Solution data
    b_dat::CuVector{T}  # size nrows * batch_size
end

function UniformBatchLinearSystemGPU(B::BatchLinearSystemCPU{T}) where T
    # Check that all matrices have the same sparsity pattern
    is_uniform_batch = all(
        (size(B.As[1], 1) == size(A, 1)) && (A.colptr == B.As[1].colptr) && (A.rowval == B.As[1].rowval)
        for A in B.As
    )
    is_uniform_batch || throw(DimensionMismatch("All matrices must have the same sparsity pattern"))

    k = length(B.As)  # batch size
    m, n = size(B.As[1])

    # Move data to GPU
    A1_gpu = CUDA.CUSPARSE.CuSparseMatrixCSR{T}(B.As[1])
    rowPtr = copy(A1_gpu.rowPtr)
    colVal = copy(A1_gpu.colVal)
    nzVal = reduce(vcat, CUDA.CUSPARSE.CuSparseMatrixCSR{T}(A).nzVal for A in B.As)

    # RHS data
    b_dat = CuVector{T}(reduce(vcat, b for b in B.bs))  # RHS coeffs

    # Solution working memory
    x_dat = CUDA.zeros(T, n*k)
    
    return UniformBatchLinearSystemGPU{T}(k, m, n, rowPtr, colVal, nzVal, x_dat, b_dat)
end
