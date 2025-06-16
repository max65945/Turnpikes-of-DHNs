# Check turnpike
function check_turnpike(T, P_p)
    tol = 1.0 * 10.0^-1
    k_entry = 0
    k_exit = N
    multentry = false
    for i in 1:p.N_n
        idxs = filter(
            k -> abs(T[i, k] - T_inf[i, N0 + k]) / (sum(T[i, :]) / length(T[i, :])) < tol,
            eachindex(T[i, :]),
        )
        minimum(idxs) > k_entry ? k_entry = minimum(idxs) : nothing
        maximum(idxs) < k_exit ? k_exit = maximum(idxs) : nothing
        sort(idxs) == collect(minimum(idxs):maximum(idxs)) ? multentry = true : nothing
    end
    println(k_entry)
    println(k_exit)
    for i in 1:p.N_p
        idxs = filter(
            k ->
                abs(P_p[i, k] - P_p_inf[i, N0 + k]) / (sum(P_p[i, :]) / length(P_p[i, :])) <
                tol,
            eachindex(P_p[i, :]),
        )
        minimum(idxs) > k_entry ? k_entry = minimum(idxs) : nothing
        maximum(idxs) < k_exit ? k_exit = maximum(idxs) : nothing
        sort(idxs) == collect(minimum(idxs):maximum(idxs)) ? multentry = true : nothing
    end
    if multentry == true 
        @info "Turnpike is entered/left multiple times."
    end
    return k_entry, k_exit
end