using LinearAlgebra
using SparseArrays

using CUDA
using CUDA.CUSPARSE
using CUDSS
using NVTX

import Base.parse
import Base.string

@enum MatrixType begin
    MatrixType_GEN = 0
    MatrixType_SYM = 10
    MatrixType_SPD = 11
    MatrixType_HER = 20
    MatrixType_HPD = 21
end

function MatrixType(s::AbstractString)::MatrixType
    if s == "GEN"
        return MatrixType_GEN
    elseif s == "SYM"
        return MatrixType_SYM
    elseif s == "SPD"
        return MatrixType_SPD
    elseif s == "HER"
        return MatrixType_HER
    elseif s == "HPD"
        return MatrixType_HPD
    else
        error("Invalid MatrixType string: $(s)")
    end
end

function Base.string(mt::MatrixType)::String
    if mt == MatrixType_GEN
        return "GEN"
    elseif mt == MatrixType_SYM
        return "SYM"
    elseif mt == MatrixType_SPD
        return "SPD"
    elseif mt == MatrixType_HER
        return "HER"
    elseif mt == MatrixType_HPD
        return "HPD"
    end
end

@enum MatrixViewType begin
    MatrixView_Full = 0
    MatrixView_Lower = 1
    MatrixView_Upper = 2
end

function MatrixViewType(s::AbstractString)::MatrixViewType
    if s == "Full" || s == "F"
        return MatrixView_Full
    elseif s == "Lower" || s == "L"
        return MatrixView_Lower
    elseif s == "Upper" || s == "U"
        return MatrixView_Upper
    else
        error("Invalid MatrixViewType string: $(s)")
    end
end

function Base.string(mv::MatrixViewType)::String
    if mv == MatrixView_Full
        return "F"
    elseif mv == MatrixView_Lower
        return "L"
    elseif mv == MatrixView_Upper
        return "U"
    end
end

function cudss_matrix_view(mview::MatrixType)
    if mview == MatrixView_Full
        return 'F'
    elseif mview == MatrixView_Lower
        return 'L'
    elseif mview == MatrixView_Upper
        return 'U'
    else
        error("Invalid MatrixViewType: $(mview)")
    end
end

function cudss_matrix_type(mtype::MatrixType)
    if mtype == MatrixType_GEN
        return "G"
    elseif mtype == MatrixType_SYM
        return "S"
    elseif mtype == MatrixType_SPD
        return "SPD"
    elseif mtype == MatrixType_HER
        return "H"
    elseif mtype == MatrixType_HPD
        return "HPD"
    else
        error("Invalid MatrixType: $(mtype)")
    end
end

struct BatchLinearSystemCPU{T}
    nbatch::Int
    is_uniform::Bool
    mtype::MatrixType
    mview::MatrixViewType

    As::Vector{SparseMatrixCSC{T,Int}} # Vector of sparse matrices (LHS)
    bs::Vector{Vector{T}}              # Vector of (dense) RHS
    xs::Vector{Vector{T}}              # Vector of (dense) solutions
end

function _is_uniform_batch(As::Vector{SparseMatrixCSC{T,Ti}}) where{T, Ti}
    return allequal(size, As) && allequal(x -> x.colptr, As) && allequal(x -> x.rowval, As)
end

function BatchLinearSystemCPU(As::Vector{SparseMatrixCSC{T,Int}}, bs::Vector{Vector{T}}) where T
    length(As) == length(bs) || throw(DimensionMismatch("Length of As and bs must be equal"))
    nbatch = length(As)

    is_uniform = _is_uniform_batch(As)
    mt = MatrixType_GEN
    mv = MatrixView_Full

    xs = [zeros(T, size(A, 2)) for A in As]
    return BatchLinearSystemCPU{T}(nbatch, is_uniform, mt, mv, As, bs, xs)
end

# Accessors
batch_size(B::BatchLinearSystemCPU) = B.nbatch
matrix_type(B::BatchLinearSystemCPU) = B.mtype
matrix_view(B::BatchLinearSystemCPU) = B.mview

struct BatchLinearSystemGPU{T,Ti}
    nbatch::Int
    is_uniform::Bool
    mtype::MatrixType
    mview::MatrixViewType

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
    
    return BatchLinearSystemGPU{T,Cint}(B.nbatch, B.is_uniform, B.mtype, B.mview, As_gpu, bs_gpu, xs_gpu)
end

# Accessors
batch_size(B::BatchLinearSystemGPU) = B.nbatch
matrix_type(B::BatchLinearSystemGPU) = B.mtype
matrix_view(B::BatchLinearSystemGPU) = B.mview

struct UniformBatchLinearSystemGPU{T}
    nbatch::Int
    nrows::Int
    ncols::Int
    mtype::MatrixType
    mview::MatrixViewType

    # LHS data
    rowPtr::CuVector{Cint}
    colVal::CuVector{Cint}
    nzVal::CuVector{T}  # size nnz * nbatch

    # RHS data
    x_dat::CuVector{T}  # size ncols * nbatch

    # Solution data
    b_dat::CuVector{T}  # size nrows * nbatch
end

function UniformBatchLinearSystemGPU(B::BatchLinearSystemCPU{T}) where T
    # Check that all matrices have the same sparsity pattern
    is_uniform_batch(B.As) || throw(DimensionMismatch("All matrices must have the same sparsity pattern"))

    k = batch_size(B)  # batch size
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
    
    return UniformBatchLinearSystemGPU{T}(k, m, n, B.mtype, B.mview, rowPtr, colVal, nzVal, x_dat, b_dat)
end

# Accessors
batch_size(B::UniformBatchLinearSystemGPU) = B.nbatch
matrix_type(B::UniformBatchLinearSystemGPU) = B.mtype
matrix_view(B::UniformBatchLinearSystemGPU) = B.mview
