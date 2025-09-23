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

    is_uniform = get(d, "is_uniform_batch", false)
    mtype = MatrixType(get(d, "matrix_type", "GEN"))
    mview = MatrixViewType(get(d, "matrix_view", "F"))

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

    xs = [
        zeros(Float64, size(A, 2))
        for A in As
    ]

    return BatchLinearSystemCPU(nbatch, is_uniform, mtype, mview, As, bs, xs)
end

function save_to_h5(f5path::AbstractString, B::BatchLinearSystemCPU)
    As = B.As
    bs = B.bs

    # Grab batch metadata
    num_systems = length(As)

    # Save to disk
    HDF5.h5open(f5path, "w") do io
        gmeta = HDF5.create_group(io, "meta")

        HDF5.write(gmeta, "num_systems", num_systems)
        HDF5.write(gmeta, "is_uniform_batch", B.is_uniform)
        HDF5.write(gmeta, "matrix_type", string(B.mtype))
        HDF5.write(gmeta, "matrix_view", string(B.mview))
        HDF5.write(gmeta, "nrow_min", minimum(size(A, 1) for A in As))
        HDF5.write(gmeta, "nrow_max", maximum(size(A, 1) for A in As))
        HDF5.write(gmeta, "ncol_min", minimum(size(A, 2) for A in As))
        HDF5.write(gmeta, "ncol_max", maximum(size(A, 2) for A in As))
        HDF5.write(gmeta, "nzz_min", minimum(nnz(A) for A in As))
        HDF5.write(gmeta, "nzz_max", maximum(nnz(A) for A in As))

        for i in 1:num_systems
            g = HDF5.create_group(io, "$i")

            A = As[i]
            b = bs[i]
            HDF5.write(g, "lhs_numrows", size(A, 1))
            HDF5.write(g, "lhs_numcols", size(A, 2))
            HDF5.write(g, "lhs_nzz", nnz(A))
            HDF5.write(g, "lhs_colPtr", A.colptr)
            HDF5.write(g, "lhs_rowVal", A.rowval)
            HDF5.write(g, "lhs_nzVal", A.nzval)
            HDF5.write(g, "rhs", b)
        end
    end

    return nothing
end

