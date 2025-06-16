function Lq_(
    B::Union{Matrix{Int},SparseMatrixCSC{Int64,Int64}}, q::AbstractVector{T}
) where {T<:Union{Number,AffExpr}}
    return (B .> 0) * Diagonal(q) * B'
end