using Serialization
using LinearAlgebra
using RowEchelon
using SparseArrays
using JuMP
using Ipopt
using HiGHS
using Serialization

#=
    # Inport DHN data: All required DHN data is stored in the binary file `DHN_data.bin` that is converted into a named tuple `p`.

    ## Fields of p:

    - N_n: Number of vertices,
    - N_e: Number of edges,
    - N_p: Number of producers,
    - N_d: Number of demands,
    - producer_id: IDs of vertices that are producer,
    - consumer_id: IDs of vertices that are consumer,
    - valves_idx: IDs of edges that are valves, 
    - B_p: Maps producers_idx to vertices, as B matrix from paper,
    - B_d: Maps consumers_idx to vertices, as E matrix from paper,
    - B0: vertex-edge incidence matrix,
    - F: fundamental loop matrix,
    - indep: IDs od independent edges,
    - Q_P: Weight matrix for penalizing power, 
    - K: Diagonal matrix containing heat loss coefficients,
    - D_e: Vetor containing pipe diameters,
    - A_e: Vetor containing pipe cross sectional areas,
    - m_e: Vetor containing pipe masses,
    - m_v: Vetor containing vertex masses,
    - L_e: Vetor containing pipe lengths,
    - P_n: Vector containing nominal heat demands,
    - T_n: Vector containing nominal vertex temperatures,
    - q_n: Vector containing nominal mass flows,
    - water_rho: densitiy of water,
    - water_c_p: heat capacity of water.
=#
p = open("./DHN_data.bin", "r") do io
    deserialize(io)
end

include("lq.jl")
include("solve_OCP.jl")
include("check_turnpike.jl")
#=
    "Infinite" horizon parameter
=#
Horizon_inf = 4.0 * 24.0 * 60.0 * 60.0  # 4 days as prediction horizon of infinite horizon OCP [s]
dt = 1.0 * 60.0                         # 1 min as sampling time [s]
N_inf = Int(Horizon_inf / dt)           # prediction steps of infinite horizon OCP
#=
    MPC parameter
=#
Horizon = 1.0 * 24.0 * 60.0 * 60.0      # 1 day as prediction horizon of MPC [s]
N = Int(Horizon / dt)                   # prediction steps of MPC
Horizont_price = 1.0 * 24.0 * 60.0 * 60.0 # horizon of price data time series [s]
N_price = Int(Horizont_price / dt)      # length of price data time series array

#=
    Define energy price profiles
=#
p_N_inf = zeros(p.N_p, N_inf - 1)
P_d = zeros(p.N_d, N_inf - 1)
for i in 1:(N_inf - 1)
    alpha = 5.0 * 10.0^3
    beta = 1.0 * 10.0^4
    p_N_inf[1, i] = alpha * sin(2 * pi / N_price * i) + beta
    p_N_inf[2, i] = alpha * sin(2 * pi / N_price * i + pi) + beta
    P_d[1, i] = (0.2 * sin(2 * pi / N_price * i + pi) + 0.8) * p.P_n[1]
    P_d[2, i] =
        (0.2 * sin(2 * pi / N_price * i + pi / 3.0) + 0.8) * p.P_n[2]
    P_d[3, i] =
        (0.2 * sin(2 * pi / N_price * i + 2.0 * pi / 3.0) + 0.8) *
        p.P_n[3]
end
Q = 10.0^3 * I(p.N_n)
foreach(i -> Q[i, i] = 10.0^4, p.consumer_id)
foreach(i -> Q[i, i] = 10.0^4, p.producer_id)

# Approximate infinite horizon OCP
T_inf, P_p_inf = solve_OCP(P_d, p.K, dt, N_inf, p_N_inf, Q, p; T0=p.T_n)

# Finite horizon OCP 1
N0 = 500
T_ocp_1, P_p_ocp_1 = solve_OCP(
    P_d, p.K, dt, N, p_N_inf, Q, p; N0=N0, T0=0.8 * T_inf[:, N0]
)

# Finite horizon OCP 2
T_ocp_2, P_p_ocp_2 = solve_OCP(
    P_d, p.K, dt, N, p_N_inf, Q, p; N0=N0, T0=1.1 * T_inf[:, N0]
)

# Finite horizon OCP 3
N_3 = Int(1.2 * N)
T_ocp_3, P_p_ocp_3 = solve_OCP(
    P_d, p.K, dt, N_3, p_N_inf, Q, p; N0=N0, T0=0.8 * T_inf[:, N0]
)

# Get time steps of entring and leaving the turnpike for each OCP solution
k_entry_1, k_exit_1 = check_turnpike(T_ocp_1, P_p_ocp_1)
k_entry_2, k_exit_2 = check_turnpike(T_ocp_2, P_p_ocp_2)
k_entry_3, k_exit_3 = check_turnpike(T_ocp_3, P_p_ocp_3)

# Check that OCP 1 and OCP 3 enter tunrpike at same point in time (2% tolerance allowed)
Δ = abs(k_entry_1 - k_entry_3)
μ = (abs(k_entry_1) + abs(k_entry_3)) / 2
@assert Δ <= 0.02 * μ "k_exit mismatch: |$k_entry_1 − $k_entry_3| = $Δ > 2% of average $μ"

# Check that OCP 1 and OCP 2 exit tunrpike at same point in time (2% tolerance allowed)
Δ = abs(k_exit_1 - k_exit_2)
μ = (abs(k_exit_1) + abs(k_exit_2)) / 2
@assert Δ <= 0.02 * μ "k_exit mismatch: |$k_exit_1 − $k_exit_2| = $Δ > 2% of average $μ"

# Check that OCP 1 leaves turnpike earlier than OCP 3
@assert k_exit_1 <= k_exit_3

#=
    # Postprocessing
=#
using CairoMakie
dN = 20
x_T_inf = (N0 + 1 - dN):(N0 + N_3 + dN)
x_P_inf = (N0 + 1 - dN):(N0 + N_3 - 1 + dN)

x_T_ocp_1 = (N0 + 1):(N0 + N)
x_P_ocp_1 = (N0 + 1):(N0 + N - 1)

x_T_ocp_2 = (N0 + 1):(N0 + N)
x_P_ocp_2 = (N0 + 1):(N0 + N - 1)

x_T_ocp_3 = (N0 + 1):(N0 + N_3)
x_P_ocp_3 = (N0 + 1):(N0 + N_3 - 1)

fig = Figure(size=(1200,800)); # Create a new figure

ax = Axis(fig[1, 1]; ylabel="[°C]")
lines!(
    ax, x_T_inf, T_inf[1, (N0 + 1 - dN):(N0 + N_3 + dN)]; label="T_1_inf", linestyle=:dash
)
lines!(ax, x_T_ocp_1, T_ocp_1[1, :]; label="T_1", linestyle=:dot)
lines!(ax, x_T_ocp_2, T_ocp_2[1, :]; label="T_1", linestyle=:dot)
lines!(ax, x_T_ocp_3, T_ocp_3[1, :]; label="T_1", linestyle=:dot)

ax = Axis(fig[2, 1]; ylabel="[°C]")
lines!(
    ax, x_T_inf, T_inf[14, (N0 + 1 - dN):(N0 + N_3 + dN)]; label="T_14_inf", linestyle=:dash
)
lines!(ax, x_T_ocp_1, T_ocp_1[14, :]; label="T_14", linestyle=:dot)
lines!(ax, x_T_ocp_2, T_ocp_2[14, :]; label="T_14", linestyle=:dot)
lines!(ax, x_T_ocp_3, T_ocp_3[14, :]; label="T_14", linestyle=:dot)

ax = Axis(fig[3, 1]; ylabel="[°C]")
lines!(
    ax, x_T_inf, T_inf[4, (N0 + 1 - dN):(N0 + N_3 + dN)]; label="T_4_inf", linestyle=:dash
)
lines!(ax, x_T_ocp_1, T_ocp_1[4, :]; label="T_4", linestyle=:dot)
lines!(ax, x_T_ocp_2, T_ocp_2[4, :]; label="T_4", linestyle=:dot)
lines!(ax, x_T_ocp_3, T_ocp_3[4, :]; label="T_4", linestyle=:dot)

ax = Axis(fig[4, 1]; ylabel="[kW]")
lines!(
    ax,
    x_P_inf,
    P_p_inf[1, (N0 + 1 - dN):(N0 + N_3 - 1 + dN)];
    label="P_p_1_inf",
    linestyle=:dash,
)
lines!(ax, x_P_ocp_1, P_p_ocp_1[1, :]; label="P_p_1", linestyle=:dot)
lines!(ax, x_P_ocp_2, P_p_ocp_2[1, :]; label="P_p_1", linestyle=:dot)
lines!(ax, x_P_ocp_3, P_p_ocp_3[1, :]; label="P_p_1", linestyle=:dot)

ax = Axis(fig[5, 1]; ylabel="[kW]")
lines!(
    ax,
    x_P_inf,
    P_p_inf[2, (N0 + 1 - dN):(N0 + N_3 - 1 + dN)];
    label="P_p_2_inf",
    linestyle=:dash,
)
lines!(ax, x_P_ocp_1, P_p_ocp_1[2, :]; label="P_p_2", linestyle=:dot)
lines!(ax, x_P_ocp_2, P_p_ocp_2[2, :]; label="P_p_2", linestyle=:dot)
lines!(ax, x_P_ocp_3, P_p_ocp_3[2, :]; label="P_p_2", linestyle=:dot)

# Third plot: p_N_inf
ax = Axis(fig[6, 1]; xlabel="time instant [-]", ylabel="")
lines!(ax, x_P_inf, p_N_inf[1, (N0 + 1 - dN):(N0 + N_3 - 1 + dN)]; label="p_1")
lines!(ax, x_P_inf, p_N_inf[2, (N0 + 1 - dN):(N0 + N_3 - 1 + dN)]; label="p_2")

fig  # Display the figure
save("results.png", fig)