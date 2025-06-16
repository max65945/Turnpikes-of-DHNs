function solve_OCP(P_d, K, dt, N, p_N_inf, Q, p; N0=0, T0=nothing)

    A = -Lq_(p.B0, p.F * p.q_n) - K
    @assert maximum(real(eigvals(Matrix(-Lq_(p.B0, p.F * p.q_n) - K)))) < 0 "Flow Laplacian is not Hurwitz for given flows."

    # Predefine vectors that store results
    T_return = zeros(p.N_p, N)    # temperature result
    P_p_return = zeros(p.N_p, N - 1)  # heat power result     

    # Define (empty) optimization model
    m = Model(Ipopt.Optimizer)
    set_silent(m)
    # Add temperature variable
    @variable(m, T[1:p.N_n, 1:N]) # one temperature for each node for each time instant
    T0 == nothing ? nothing : @constraint(m, T[:, 1] == T0) # fix initial temperature to nominal temperature
    foreach(i -> foreach(k -> set_start_value(T[i, k], p.T_n[i]), 1:N), 1:p.N_n) # set a start value to improve numerical performance

    # Add heat power variable
    @variable(m, P_p[1:p.N_p, 1:(N - 1)]) # one heat power for each producer for each time instant
    foreach(i -> foreach(k -> @constraint(m, P_p[i, k] >= 0.0), 1:(N - 1)), 1:p.N_p) # add lower bound for each heat power variable
    foreach(i -> foreach(k -> @constraint(m, P_p[i, k] <= 2.0 * p.P_n[i]), 1:(N - 1)), 1:p.N_p) # add upper bound for each heat power variable
    foreach(i -> foreach(k -> set_start_value(P_p[i, k], p.P_n[i]), 1:(N - 1)), 1:p.N_p) # set a start value to improve numerical performance

    # ODE constraint
    foreach(
        k -> @constraint(
            m,
            T[:, k + 1] ==
                T[:, k] +
            dt *
            inv(Diagonal(p.m_v)) *
            (
                A * T[:, k] + (10^3 / water_c_p) * p.B_p * P_p[:, k] -
                (10^3 / water_c_p) * p.B_d * P_d[:, N0 + k]
            )
        ),
        1:(N - 1),
    )

    # Objective functional
    global J = 0
    for k in 1:(N - 1)
        xQx = 0.5 * T[:, k]' * Q * T[:, k] #TODO: Norm costs
        uSx = 10^(0) * (P_p[:, k])' * p.B_p' * T[:, k]#-p.T_n)
        rx = -p.T_n' * Q * (T[:, k])
        pu = p_N_inf[1, N0 + k] * P_p[1, k] + p_N_inf[2, N0 + k] * P_p[2, k]
        performance_helper = 0#10.0^(-6) * (P_p[:,k])' * (P_p[:,k])
        global J += xQx + pu + rx + uSx + performance_helper
    end

    @objective(m, Min, J) # add objective to optimization model

    optimize!(m)
    println(termination_status(m))

    T_return = value.(T)
    P_p_return = value.(P_p)

    @info "Solving OCP succeeded."
    return T_return, P_p_return
end