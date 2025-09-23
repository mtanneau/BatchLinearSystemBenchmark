using Random

"""
    generate_random_batch(m::Int, p::Float64, nbatch::Int; seed=42, symmetric=false)

Generates a random uniform batch of `nbatch` sparse `m x m` linear systems with density `p`.
"""
function generate_random_batch(m::Int, p::Float64, nbatch::Int; seed=42, symmetric=false)
    rng = MersenneTwister(seed)

    As = [
        (sprandn(rng, m, m, p) + Diagonal(randn(rng, m)))
        for _ in 1:nbatch
    ]
    if symmetric
        As = sparse.(Symmetric.(As))
    end

    mtype = symmetric ? MatrixType_SYM : MatrixType_GEN
    mv = MatrixView_Full

    bs = [randn(rng, m) for _ in 1:nbatch]
    xs = [zeros(Float64, m) for _ in 1:nbatch]

    return BatchLinearSystemCPU(nbatch, false, mtype, mv, As, bs, xs)
end

"""
    generate_random_uniform_batch(m::Int, p::Float64, nbatch::Int; seed=42, symmetric=false)

Generates a random uniform batch of `nbatch` sparse `m x m` linear systems with density `p`.
"""
function generate_random_uniform_batch(m::Int, p::Float64, nbatch::Int; seed=42, symmetric=false)
    rng = MersenneTwister(seed)

    # Generate random sparsity pattern
    A = sprand(rng, m, m, p) + Diagonal(randn(rng, m))
    if !symmetric
        As = [copy(A) for _ in 1:nbatch]
        for i in 1:nbatch
            As[i].nzval .= randn(rng, nnz(A))
        end
    else
        # We need to ensure that the perturbed matrices all remain symmetric
        A = sparse(UpperTriangular(A))
        As = [copy(A) for _ in 1:nbatch]
        # Perturb coefficients...
        for i in 1:nbatch
            As[i].nzval .*= (1.0 .+ 1e-2 .* randn(rng, nnz(A)))
        end
        # ... then re-symmetrize everything
        # ⚠️ there is a _small_ chance that some coefficients got zeroed-out
        As = [sparse(Symmetric(A)) for A in As]
    end
    
    mtype = symmetric ? MatrixType_SYM : MatrixType_GEN
    mv = MatrixView_Full

    bs = [randn(rng, m) for _ in 1:nbatch]
    xs = [zeros(Float64, m) for _ in 1:nbatch]

    return BatchLinearSystemCPU(nbatch, true, mtype, mv, As, bs, xs)
end
