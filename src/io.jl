using HDF5
using SparseArrays

function _parse_h5_to_csc(mdat)
    m = mdat["lhs_numrows"]
    n = mdat["lhs_numcols"]
    colptr = mdat["lhs_colPtr"]
    rowval = mdat["lhs_rowVal"]
    nzval = mdat["lhs_nzVal"]

    A = SparseMatrixCSC(m, n, colptr, rowval, nzval)

    b = mdat["rhs"]

    return A, b
end

"""
    load_from_h5(f5path; nbatch=-1)

Load up to `nbatch` linear systems from HDF5 file. If `nbatch==-1`, load all systems.

Returns a `BatchLinearSystemCPU{Float64}` instance.
"""
function load_from_h5(f5path::AbstractString; nbatch=-1)
    d = h5read(f5path, "meta")

    # How many systems do we have?
    num_systems = d["num_systems"]

    if nbatch == -1 || nbatch > num_systems
        nbatch = num_systems
    end

    # Read 
    As = SparseMatrixCSC{Float64,Int}[]
    bs = Vector{Float64}[]

    h5open(f5path, "r") do io
        for i in 1:nbatch
            mdat = read(io["$(i)"])
            A, b = _parse_h5_to_csc(mdat)
            push!(As, A)
            push!(bs, b)
        end
    end

    return BatchLinearSystemCPU(As, bs)
end

function save_to_h5(f5path::AbstractString, B::BatchLinearSystemCPU)
    As = B.As
    bs = B.bs

    # Grab batch metadata
    num_systems = length(As)
    is_uniform_batch = (
        allequal(size, As) && allequal(x -> x.colptr, As) && allequal(x -> x.rowval, As)
    )  # a batch is uniform if all matrices have same sparsity pattern
    is_symmetric_batch = all(
        issymmetric(A) for A in As
    )  # a batch is symmetric is all matrices are symmetric

    metadata = Dict{String,Any}(
        "num_systems" => num_systems,
        "is_uniform_batch" => is_uniform_batch,
        "is_symmetric_batch" => is_symmetric_batch,
        "nrow_min" => minimum(size(A, 1) for A in As),
        "nrow_max" => maximum(size(A, 1) for A in As),
        "ncol_min" => minimum(size(A, 2) for A in As),
        "ncol_max" => maximum(size(A, 2) for A in As),
        "nzz_min" => minimum(nnz(A) for A in As),
        "nzz_max" => maximum(nnz(A) for A in As),
    )

    # Save to disk
    h5open(f5path, "w") do io
        gmeta = create_goup(io, "meta")
        write(gmeta, "meta", metadata)

        for i in 1:num_systems
            g = create_group(io, "$i")

            A = As[i]
            b = bs[i]
            write(g, k2, v2)
            write(g, "lhs_numrows", size(A, 1))
            write(g, "lhs_numcols", size(A, 2))
            write(g, "lhs_nzz", nnz(A))
            write(g, "lhs_colPtr", A.colptr)
            write(g, "lhs_rowVal", A.rowval)
            write(g, "lhs_nzVal", A.nzval)
            write(g, "rhs", b)
        end
    end

    return nothing
end

