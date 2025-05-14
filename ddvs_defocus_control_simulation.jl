#= 
Converted from MATLAB to Julia: AV-Defocus Simulation Script (RAL 2021)
Assumes helper functions `gradient_h` and `laplacien_h` are defined elsewhere
=#

using Plots
using LinearAlgebra
using Printf

Z_unknown = true
exp_sampling = true
threshold = true
di = 6

# Camera parameters
f = 17e-3  # meters
FNumber = 0.95

fact = 2
Nc = Int(round(1280 / fact))
NcP = Nc - 2 * di
ku = fact * 5.3e-6  # meters
au = f / ku
u0 = Nc / 2

alpha = 1 / (6 * ku)
D = f / FNumber
csZf = 0.25


Zshift = 0.02#exp_sampling ? 0.0036 : 1e-6
csX = -0.03
csZ = csZf + Zshift
L = 1.0
tX0 = 0.01
tZ0 = 0
δZf0 = -0.1

Is = zeros(Nc)
us = au * csX / csZ + u0
Is[clamp(Int(round(us)), 1, Nc)] = L
ϵ = 0.001
lcsZf = csZf - f;
lcsZ = csZ - f;
c = alpha * D * f / (lcsZf - f)

lambda_e = abs(c * (1 - lcsZf / lcsZ))

if exp_sampling
    for ud in 1:Nc
        Is[ud] = L * (1 / (sqrt(2 * pi) * lambda_e)) * exp(-((ud - us)^2) / (2 * lambda_e^2))
        if threshold
            Is[ud] = (Is[ud]<ϵ) ? ϵ : L * (1 / (sqrt(2 * pi) * lambda_e)) * exp(-((ud - us)^2) / (2 * lambda_e^2))
        end
    end
else
    for ud in 1:Nc
        fun(x) = (1 / (sqrt(2 * pi) * lambda_e)) * exp(-((x - us)^2) / (2 * lambda_e^2))
        Is[ud] = L * quadgk(fun, ud - 0.5, ud + 0.5)[1]
        if threshold
            Is[ud] = (Is[ud]<ϵ) ? ϵ : L * quadgk(fun, ud - 0.5, ud + 0.5)[1]
        end
    end
end


global cX = csX + tX0
global cZ = csZ + tZ0
global cZf = csZf + δZf0
global vX = 0.0
global vZ = 0.0
global δZf = 0.0

#cst = D * f / (6 * ku * (cZf - f))

mu = 0.02

cost = Float64[]
tab_cX = Float64[]
tab_cZ = Float64[]
tab_l = Float64[]
tab_uR = Float64[]
tab_czf = Float64[]
Niter = 600

function gradient_h(v, d)
    grad = zeros(Float64, length(v))
    for i in d+1:length(v)-d
        grad[i] = (v[i+1] - v[i-1]) / 2
    end
    return grad
end

function laplacien_h(v, d)
    lap = zeros(Float64, length(v))
    for i in d+1:length(v)-d
        lap[i] = v[i+1] - 2*v[i] + v[i-1]
    end
    return lap
end
for iter in 1:Niter
    global c_zI = cZ
    #lambda_e_back = lambda_e
    @printf("\rProgress: %3d%% (%d / %d iterations)", round(Int, 100 * iter /Niter), iter, Niter)
    flush(stdout)

    global cX += vX
    global cZ += vZ
    global cZf += δZf

    if cZ <= 0
        break
    end
    if cZf <=0 
        break
    end
    
    if cZf >= cZ
        break
    end
    if abs(cZ-cZf) == 0.001
        break
    end

    

    push!(tab_cX, cX)
    push!(tab_cZ, cZ)
    push!(tab_czf, cZf)
    lcZ = cZ - f;
    lcZf = cZf - f
    c = alpha * D * f / (lcZf - f)
    
    lambda_e = abs(c * (1 - lcZf / lcZ))
    push!(tab_l, lambda_e)

    uR = au * cX / cZ + u0
    push!(tab_uR, uR)
    if uR < 1 || uR > Nc
        break
    end
   
    β_zf = -(alpha*D*f*(lcZ-f)/(lcZ*(lcZf-f)^2))
    cst = D * f / (6 * ku * (lcZf - f))
    βc_zf = -(alpha*D*f*(lcsZ-f)/(lcsZ*(lcZf-f)^2))
    I = zeros(Nc)
    uI = clamp(Int(round(uR)), 1, Nc)
    I[uI] = L

    Id = zeros(Nc)
    if exp_sampling
        for ud in 1:Nc
            Id[ud] = I[uI] * (1 / (sqrt(2 * pi) * lambda_e)) * exp(-((ud - uR)^2) / (2 * lambda_e^2))
            if threshold
                Id[ud] = (Id[ud]<ϵ) ? ϵ : I[uI] * (1 / (sqrt(2 * pi) * lambda_e)) * exp(-((ud - uR)^2) / (2 * lambda_e^2))
            end
        end
    else
        for ud in 1:Nc
            fun(x) = (1 / (sqrt(2 * pi) * lambda_e)) * exp(-((x - uR)^2) / (2 * lambda_e^2))
            Id[ud] = I[uI] * quadgk(fun, ud - 0.5, ud + 0.5)[1]
            if threshold
                Id[ud] = (Id[ud]<ϵ) ? ϵ : I[uI] * quadgk(fun, ud - 0.5, ud + 0.5)[1]
            end       
        end
    end
    
    plt = plot(1:Nc, Is,
           linewidth=2,
           label="Desired (Is) Zf_d =$(round(csZf, digits=3)), Z_d =$(round(csZ, digits=3)), X_d = $(round(csX, digits=3))",
           legend =:topright,
           foreground_color_legend = nothing,
           background_color_legend = nothing,
           xlabel="pixels",
           ylabel="brightness",
           title="Desired vs. Current Image iter $(iter)",
           grid=true)

    plot!(1:Nc, Id,
      linewidth=2,
      linestyle=:dash,
      label="Current (Id) with Zf = $(round(cZf, digits=3)), Z = $(round(cZ, digits=3)), X = $(round(cX, digits = 3))")

    savefig(plt, "current_vs_desired_image_defocus_control/iter_$(lpad(iter, 3, '0')).png")
    
    err = Id[di+1:end-di] .- Is[di+1:end-di]
    push!(cost, 0.5 * sum(err .^ 2))
    if (0.5 * sum(err .^ 2) <= 10^(-6))
        println("It took $iter before convergence")
        break;
    end
    ∇Id = gradient_h(Id, di)
    ΔId = laplacien_h(Id, di)

    J = zeros(NcP, 3)
    for (ind, ud) in enumerate(di+1:Nc-di)
        x = (ud - u0) / au
        Jgeom = Z_unknown ? [-au / csZ au * x / csZ 0; 0 -cst / csZ βc_zf] : [-au / cZ au * x / cZ 0; 0 -cst / cZ  β_zf]
        J[ind, :] = [-∇Id[ud] -ΔId[ud]] * Jgeom
    end
    #κ = cond(J)
    #println("matrice conditionnement:  ", κ)

    v = mu * pinv(J) * err
    global vX, vZ, δZf = v[1], v[2], v[3]
    p1 = plot(cost, lw=2, label  ="Cost (SSD)",xlabel="Iterations", ylabel="SSD", title="Cost vs Iteration", linecolor = "hotpink4")
    p2 = plot(tab_l, lw=2,label  ="Current λ(Z,Zf)", xlabel="Iterations", ylabel="λ(Z)", title="λ Evolution", linecolor = "hotpink4")
    p3 = plot(tab_cX, label="cX", lw=2,color= "red")
    plot!(p3, fill(csX, length(tab_cX)), label="X (desired)", linestyle=:dash,linecolor	= "red")
    plot!(p3, tab_cZ, label="cZ", lw=2, linecolor= "blue")
    plot!(p3, fill(csZ, length(tab_cX)), label="Z (desired)", linestyle=:dash, linecolor = "blue")
    plot!(p3, tab_czf, label="cZf", lw=2, linecolor = "orangered4")
    plot!(p3, fill(csZf, length(tab_cX)), label="Zf (desired)", linestyle=:dash,linecolor = "orangered4")
    plot!(p3, xlabel="Iterations", ylabel="DoF", title="Camera Path",legend=:right)
    p4 = plot(tab_uR, label="uR", lw=2,linecolor = "hotpink4")
    plot!(p4, fill(us, length(tab_uR)), label="us (desired)", linestyle=:dash,linecolor = "hotpink4")
    plot!(p4, xlabel="Iterations", ylabel="u", title="Pixel Position")
    plot_summary = plot(p1, p2, p3, p4, layout=(1,4), size=(1600,400))
    savefig(plot_summary, "DDVS_defocus_control_result/DVS_summary_iter_$(lpad(iter, 3, '0')).png")
end

println("error corrected in the $Nc pixels image = $(abs(tab_uR[end] - tab_uR[1])) pixels ($(100 * abs(tab_uR[end] - tab_uR[1]) / Nc)%)")
