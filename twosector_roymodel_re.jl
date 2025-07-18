using Pkg
using Random
using Distributions 
using Statistics
Random.seed!(1234) # Set a seed
using LinearAlgebra
using Integrals
#using DataFrames
using Plots, Distributions, Random, Optim
using NLSolversBase, ForwardDiff, FiniteDifferences
#using GLM
#Parameters
beta0a = 20
beta0b = 22
beta1a= 4 
beta1b = 2
gamma = 2
sigmab = 2
rho = -0.4
sigma_theta = 3
p_b = 3
# distribution = Normal(mean, std)
obs= 10000
mean = [0, 0]
Σ = [1 rho*sigmab ; rho*sigmab sigmab^2]
dist_e = MvNormal(mean, Σ) 
e = rand(dist_e, obs)

dist_x = Normal(2,1)
dist_z = Normal(0, 1.5)
dist_theta = Normal(0, sigma_theta)
x = rand(dist_x, obs)
z = rand(dist_z, obs)
theta = rand(dist_theta, obs)

identityv = ones(obs)

# Generate Wage in ind A and ind B in population without missing values.
ya = ones(obs)
yb = ones(obs)
ea = e[1,:]
eb = e[2,:]
# wage in industry A
ya = beta0a .+ beta1a .* x .+ theta .+ ea
# wage in industry B
yb = beta0b .+ beta1b .* x .+ p_b .* theta .+ eb

v = ea - eb
# Selection rule - Non random attrition a.k.a., "why some values are missing?" 
da = (beta0a - beta0b) .+ (beta1a - beta1b) .* x .+(1 - p_b) .* theta .- gamma .* z  .+ v

da_dummy = da .>= 0 #1 for who choose A and 0 for who choose B
# Indicator: 1 for who choose A and 0 for who choose B

obsa = length(da_dummy[da_dummy .> 0])
obsb = length(da_dummy[da_dummy .< 1])

function loglikelihood(θ)

    β0a = θ[1]
    β0b = θ[2]
    β1a = θ[3]
    β1b = θ[4]
    γ = θ[5]
    log_σb = θ[6]
    tanh_ρ = θ[7]
    log_σθ = θ[8]
    κ = θ[9]
    
    σb = exp(log_σb)
    ρ = atan(tanh_ρ)
    σθ = exp(log_σθ)

    σ_ζ = sqrt((1-κ)^2*σθ^2 + 1 + σb^2 - 2*ρ*σb)
    σA = sqrt(σθ^2 + 1)
    σB = sqrt(κ^2*σθ^2 + σb^2)
    
    ρ_Aζ = ((1-κ)*σθ^2 + 1 - ρ*σb) / (σ_ζ*σA)
    ρ_Bζ = (κ*(1-κ)*σθ^2 - σb^2 + ρ*σb) / (σ_ζ*σB)
         
    # To calculate the variance correctly, we should use obs-a and obs b sepearatley. 
    res_a = ya .- (β0a .+ β1a .* x )
    res_b = yb .- (β0b .+ β1b .* x ) 
    da_cond_a = (β0a .- β0b) .+ (β1a - β1b) .* x - γ .* z .+ res_a .* ((ρ_Aζ .* σ_ζ) / σA)
    da_cond_a_std = sqrt((1-ρ_Aζ^2))*σ_ζ
    da_cond_a_normal = da_cond_a ./ da_cond_a_std
    da_cond_b = (β0a .- β0b) .+ (β1a - β1b) .* x - γ .* z .+ res_b .* ((ρ_Bζ .* σ_ζ) / σB)
    da_cond_b_std = sqrt((1-ρ_Bζ^2))*σ_ζ
    da_cond_b_normal = da_cond_b ./ da_cond_b_std
    # cdf probabilty of (d|y)
    prob_cdfa_mar = cdf.(Normal(0, 1), da_cond_a_normal) #.+ 1e-6
    prob_cdfa_mar = clamp.(prob_cdfa_mar, 1e-6, 1 - 1e-6)
    prob_cdfb_mar = 1 .- cdf.(Normal(0, 1), da_cond_b_normal) #.+ 1e-6
    prob_cdfb_mar = clamp.(prob_cdfb_mar, 1e-6, 1 - 1e-6)
    # probability of y
    res_a_normal = res_a ./ σA
    res_b_normal = res_b ./ σB
    prob_pdfa = pdf.(Normal(0,1), res_a_normal)
    prob_pdfa = clamp.(prob_pdfa, 1e-6, Inf)
    prob_pdfb = pdf.(Normal(0,1), res_b_normal)
    prob_pdfb = clamp.(prob_pdfb, 1e-6, Inf)
    #prob of observed 
    sel_term = ((β0a - β0b)/σ_ζ .+ (β1a - β1b)/σ_ζ .* x .- γ/σ_ζ .* z) 

    sel_cdf = cdf.(Normal(0,1), sel_term)
    sel_cdf = clamp.(sel_cdf, 1e-6, 1 - 1e-6)
    #res_a_term = []
    #res_b_term = []
    sigmaoneB = ones(obs) .* log(σB)
    sigmaoneA = ones(obs) .* log(σA)
    ll = -(sum(log.(prob_cdfa_mar[da_dummy .== 1]) .+ log.(prob_pdfa[da_dummy .== 1]) .- sigmaoneA[da_dummy .== 1] .- log.(sel_cdf[da_dummy .== 1])) 
    + sum(log.(prob_cdfb_mar[da_dummy .== 0]) .+ log.(prob_pdfb[da_dummy .== 0]) .- sigmaoneB[da_dummy .== 0] .- log.(1 .- sel_cdf[da_dummy .== 0])))
    return -ll
end

guess = [beta0a, beta0b, beta1a, beta1b, gamma, log(sigmab), tanh(rho),  log(sigma_theta), p_b]
guess1 = [20.0000,20.000, 3.0000, 3.0000, 0.0000, log(1.0000), tanh(0.000), log(1.0000), 0.000]
opt = optimize(x -> -loglikelihood(x), guess, show_trace=true, iterations = 5000)
beta_hat = Optim.minimizer(opt)
[beta_hat guess]

println("\nEstimated Parameters:")
println("β0a = ", beta_hat[1])
println("β0b = ", beta_hat[2])
println("β1a = ", beta_hat[3])
println("β1b = ", beta_hat[4])
println("γ   = ", beta_hat[5])
println("σb  = ", exp(beta_hat[6]))
println("ρ   = ", atan(beta_hat[7]))
println("σθ   = ", exp(beta_hat[8]))
println("κ   = ", beta_hat[9])