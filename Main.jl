using Parameters, ForwardDiff, NLsolve, DifferentialEquations, Plots, ODEInterfaceDiffEq, LinearAlgebra, Interpolations, Optim, JuMP, GLPK, Ipopt, JLD, Printf

cd("C:\\Users\\ihsan\\Desktop\\Projects\\Julia\\replication") #Working directory. You need to change this
gr()

############ 1. DEFINE FUNDAMENTALS (PARAMETERS, FUNCTIONS, ETC)

@with_kw struct Params
    λ₁ᵒ::Float64 = 0.09
    λ₁ᵖ::Float64 = 0.9
    λ₂ᵒ::Float64 = 4.97
    λ₂ᵖ::Float64 = 0.49
    ρ::Float64 = 0.04
    A::Float64 = 1.0
    ηbar::Float64 = 0.97
    δbar::Float64 = 0.04
    δupperbar::Float64 = 0.087
    ϵ::Float64 = 20.0
    γ::Float64 = 1.0
    dt::Float64 = 0.001
    t₁::Float64 = 0.0 #Start time of boom state s=1
    t₂::Float64 = 0.4 #Start time of recession state s=2
    t₃::Float64 = 0.77 #Start time of recovery state s=3
    t = range(0.0, 1.0, step=dt);
end

param = Params();

δ(η) = param.δbar+(param.δupperbar-param.δbar)*(max(η[1]-param.ηbar,0.0)^(1+1/param.ϵ)/(1+1/param.ϵ));
δ′(η) = (param.δupperbar-param.δbar)*max(η[1]-param.ηbar,0.0)^(1/param.ϵ);
δ′′(η) = (param.δupperbar-param.δbar)*(1/param.ϵ)*max(η[1]-param.ηbar,0.0)^(1/param.ϵ);
ff(η) = δ′(η[1])*η[1]-param.ρ;
ηstar = nlsolve(ff, [1.0]).zero[1];
Qstar = param.A*ηstar/param.ρ;

g₁ = 0.1-(param.ρ-δ(ηstar));
g₃ = 0.1-(param.ρ-δ(ηstar));
g₂ = -0.05-(param.ρ-δ(ηstar));

u(c) = param.γ==1.0 ? log(c) : c^(1-param.γ)/(1-param.γ);
αmin = 0.0;
αmax = 1.0;
αspan = (αmin, αmax);

λ₁pl_def = (param.λ₁ᵒ+param.λ₁ᵖ)/2; #Planner beliefs
λ₂pl_def = (param.λ₂ᵒ+param.λ₂ᵖ)/2; #Planner beliefs

λ₁bar(α) = param.λ₁ᵒ*α+param.λ₁ᵖ*(1-α);
λ₂bar(α) = param.λ₂ᵒ*α+param.λ₂ᵖ*(1-α);

a = range(0.4, 0.9, length=100); #α grid
a2 = range(0.0, 1.0, length=101); #full α grid
dα = 0.01;

############ 2. CALCULATE RECESSION STATE FUNCTIONS: nQ₂base(α₂):normalized price of capital (Q₂/Q*) and w₂base(α₂):gap value function
#nQ₂base
ff(x) = param.ρ+g₂-δ(x*ηstar)+param.λ₂ᵖ*(1-x[1]);
Q2₀ = nlsolve(ff, [0.9]).zero;

ff(x) = param.ρ+g₂-δ(x*ηstar)+param.λ₂ᵒ*(1-x[1]);
Q2₁ = nlsolve(ff, [0.99]).zero;

function fQ2!(dQ2,Q2,p,α)
    param, g₂, δ, ηstar = p
    if α == 0.0
        dQ2[1] = 0.0;
    elseif α == 1.0
        dQ2[1] = 0.0;
    else
        dQ2[1] = (Q2[1]/((1-α)*α*(param.λ₂ᵒ-param.λ₂ᵖ)))*(param.ρ+g₂-δ(Q2*ηstar)+(α*param.λ₂ᵒ+(1-α)*param.λ₂ᵖ)*(1-Q2[1]));
    end
end

function fQ2!_jac(J,Q2,p,α)
    param, g₂, δ, ηstar = p
    if α == 0.0
        J[1,1] = 0.0;
    elseif α == 1.0
        J[1,1] = 0.0;
    else
        J[1,1] = (1.0/((1-α)*α*(param.λ₂ᵒ-param.λ₂ᵖ)))*(param.ρ+g₂-δ(Q2*ηstar)+(α*param.λ₂ᵒ+(1-α)*param.λ₂ᵖ)*(1-Q2[1]))-(δ′(Q2*ηstar)*ηstar+(α*param.λ₂ᵒ+(1-α)*param.λ₂ᵖ))*(Q2[1]/((1-α)*α*(param.λ₂ᵒ-param.λ₂ᵖ)));
    end
    nothing
end

f = ODEFunction(fQ2!, jac=fQ2!_jac);
p = (param, g₂, δ, ηstar);
problemQ2 = ODEProblem(f,Q2₀,αspan,p);

nQ₂base = DifferentialEquations.solve(problemQ2, Vern7(), saveat=0.001);
nQ₂(α₂) = nQ₂base(α₂)[1];

#w₂base
W(nQ) = log(nQ)-(1/param.ρ)*(δ(nQ*ηstar)-δ(ηstar));
function fw₂!(dw₂, w₂, p, α₂)
    param, λ₂pl, W, nQ₂base, dα = p
    if α₂==0.0
        dw₂[1] = 0.0;
    elseif α₂==1.0
        dw₂[1] = 0.0;
    else
        dw₂[1] = ((param.ρ+λ₂pl)*w₂[1]-W(nQ₂base(α₂)[1]))/((λ₂bar(α₂)-param.λ₂ᵒ)*α₂);
    end
end

function fw₂_jac!(J, w₂, p, α₂)
    param, λ₂pl, W, nQ₂base, dα = p
    if α₂==0.0
        J[1,1] = 0.0;
    elseif α₂==1.0
        J[1,1] = 0.0;
    else
        J[1,1] = (param.ρ+λ₂pl)/((λ₂bar(α₂)-param.λ₂ᵒ)*α₂);
    end
    nothing
end

w₂base = DifferentialEquations.solve(ODEProblem(ODEFunction(fw₂!, jac=fw₂_jac!), [W(nQ₂base(0.0)[1])/(param.ρ+λ₂pl_def)], αspan, (param, λ₂pl_def, W, nQ₂base, dα)), Vern7(), saveat=0.001);
w₂(α₂) = w₂base(α₂)[1];

############ 3. DEFINE TYPES TO CREATE MODELS AND SOLVER FUNCTIONS
#Baseline model is the one without PMP, that is monetary policy can implement efficient capital prices (zero lower bound is not hit)
@with_kw struct BaselineInitial
    ωbar::Float64   #Leverage limit
    α₀::Float64     #Initial value for the state variable α
    nQ₁::Function   #Normalized price of capital (Q₁/Q*)
end

#PMP is a monetary policy in which a given optimist welath share function α₂(α₁), that is obtained by a macroprudential policy, can be replicated
@with_kw struct PMPInitial
    ωbar::Float64
    nα₂::Function   #Normalized optimist wealth share α₂(α₁)/α₁
    α₀::Float64
end

#This is a generic type for a solved model that I keep equilibrium functions and other ingredients.
@with_kw mutable struct ModelSolved
    ωbar::Float64   #Leverage limit
    α₀::Float64     #Initial value for the state variable α
    nQ₁::Function   #Normalized price of capital (Q₁/Q*)

    nα₂star::Function   #Non-binding normalized optimist wealth share. TO BE SOLVED
    ωᵒstar::Function    #Non-binding leverage. TO BE SOLVED

    isnQ₁feasible::Bool     #Check if the monetary policy hits the zero lower bound or not. TO BE SOLVED
    isμpositive::Bool       #For some unusual values of given nQ₁ or ωbar, I sometimes had negative multiplier μ for the leverage limit. This is a check for that. TO BE SOLVED

    nα₂::Function       #Normalized optimist wealth share function: α₂(α₁)/α₁ when transitioning from the boom state to recession state. TO BE SOLVED
    μ::Function         #Lagrange multipler for the leverage limit. TO BE SOLVED
    ωᵒ::Function        #Optimal leverage in the boom state s=1. TO BE SOLVED
    α₁dot::Function     #Motion of the state variable α₁ through time in the boom state s=1. TO BE SOLVED
    r₁::Function        #Return on market portfolio (risky return) in the boom state s=1. TO BE SOLVED
    r₁ᶠ::Function       #Risk free interest rate in the boom state s=1 (We regard in the paper that the montary policy directly sets capital price rather than risk free rate. Thus, this is an equilibrium outcome). TO BE SOLVED

    αₜ::Vector{Float64} = [NaN]     #Simulated equilibrium state variable α. TO BE SOLVED
    rₜᶠ::Vector{Float64} = [NaN]    #Simulated risk free rate. TO BE SOLVED
    rₜ::Vector{Float64} = [NaN]     #Simulated risky return. TO BE SOLVED
    nQₜ::Vector{Float64} = [NaN]    #Simulated normalized price of capital. TO BE SOLVED

    w₁::Function = (α₁) -> NaN              #Gap value function for the boom state s=1. TO BE SOLVED
    w₂::Function = (α₂) -> w₂base(α₂)[1]    #Gap value function for the boom state s=2. ALREADY SOLVED ABOVE
    w₃::Function = (α₃) -> 0.0              #Gap value function for the boom state s=3. It is 0 for all α₃

    α₂::Function    = (α₁) -> nα₂(α₁)*α₁                #Optimist wealth share function when transitioning from boom state to recession state (not normalized)
    α₂dot::Function = (α₂) -> (λ₂bar(α₂)-param.λ₂ᵒ)*α₂  #Motion of the state variable α₂ through time in the recession state s=2
    α₃::Function    = (α₂) -> param.λ₂ᵒ/λ₂bar(α₂)*α₂    #Optimist wealth share function when transitioning from the recession state to recovery state (not normalized)
    α₃dot::Function = (α₃) -> 0.0                       #Motion of the state variable α₂ through time in the recovery state s=3. It is 0 by definition

    nQ₂::Function   = (α₂) -> nQ₂base(α₂)[1]    #Normalized price of capital in the recession state s=2. ALREADY SOLVED ABOVE
    nQ₃::Function   = (α₃) -> 1.0               #Normalized price of capital in the recovery state s=3. Monetary policy can implement the efficient level at the recovery state

    r₂ᶠ::Function   = (α₂) -> 0.0                   #Risk free rate in the recession state s=2. By parametric assumptions in the paper, the monetary policy hits the zero lower bound
    r₃ᶠ::Function   = (α₃) -> param.ρ+g₃-δ(ηstar)   #Risk free rate in the recovery state. It is equal to the risky return, since there is no uncertainty after the recovery

    r₂::Function    = (α₂) -> param.ρ+g₂-δ(nQ₂(α₂)*ηstar)+ForwardDiff.derivative(nQ₂,α₂)/nQ₂(α₂)*α₂dot(α₂)  #Risky return in the recession state s=2
    r₃::Function    = (α₃) -> param.ρ+g₃-δ(ηstar)                                                           #Risky return in the recovery state
end

#This function solves a baseline model, that is for a given monetary policy (nQ₁:normalized capital price), this function finds equilibrium optimist walth share function α₂(α₁)
function solver(model::BaselineInitial, param::Params)

    ωbar                = model.ωbar;
    α₀                  = model.α₀;
    nQ₁                 = model.nQ₁;
    dt, t₁, t₂, t₃, t   = param.dt, param.t₁, param.t₂, param.t₃, param.t;

    nα₂star(α₁) = param.λ₁ᵒ/λ₁bar(α₁);
    ωᵒstar(α₁)  = nQ₂(α₁*nα₂star(α₁))/(nQ₂(α₁*nα₂star(α₁))-nQ₁(α₁))*(nα₂star(α₁)-1)+1.0;

    function nα₂(α₁)
        if α₁ == 0.0
            return (ωbar-1)*((nQ₂(0.0)-nQ₁(0.0))/nQ₂(0.0))+1.0
        elseif α₁ == 1.0
            return 1.0
        elseif ωᵒstar(α₁) <= ωbar
            return nα₂star(α₁)
        else
            return nlsolve(nα₂ -> (nQ₂(nα₂[1]*α₁)/(nQ₂(nα₂[1]*α₁)-nQ₁(α₁)))*(nα₂[1]-1.0)+1.0-ωbar, [0.5*(nα₂star(α₁)+1/α₁)]).zero[1]
        end
    end

    function μ(α₁)
        if α₁ == 0.0
            return (1/param.ρ)*((1/nα₂(α₁))*param.λ₁ᵒ-param.λ₁ᵖ)*((nQ₂(0.0)-nQ₁(0.0))/nQ₂(0.0));
        elseif α₁ == 1.0
            return 0.0
        elseif ωᵒstar(α₁) <= ωbar
            return 0.0
        else
            return (1/param.ρ)*((1/nα₂(α₁))*param.λ₁ᵒ-(1-α₁)/(1-nα₂(α₁)*α₁)*param.λ₁ᵖ)*((nQ₂(nα₂(α₁)*α₁)-nQ₁(α₁))/nQ₂(nα₂(α₁)*α₁))
        end
    end

    function ωᵒ(α₁)
        if α₁ == 0.0
            return ωbar
        elseif α₁ == 1.0
            return 1.0
        elseif ωᵒstar(α₁) <= ωbar
            return ωᵒstar(α₁)
        else
            return ωbar
        end
    end

    function α₁dot(α₁)
        if α₁ == 0.0
            return 0.0
        elseif α₁ == 1.0
            return 0.0
        else
            return α₁*((1-α₁)/(1-nα₂(α₁)*α₁)*param.λ₁ᵖ)*(1-nα₂(α₁))
        end
    end

    function r₁(α₁)
        if α₁ == 0.0 || α₁ == 1.0
            return param.ρ+g₁-δ(nQ₁(α₁)*ηstar)
        else
            return param.ρ+g₁-δ(nQ₁(α₁)*ηstar)+ForwardDiff.derivative(nQ₁,α₁)/nQ₁(α₁)*α₁dot(α₁)
        end
    end

    function r₁ᶠ(α₁)
        if α₁ == 0.0 || α₁ == 1.0
            return r₁(α₁) + λ₁bar(α₁)*((nQ₂(nα₂(α₁)*α₁)-nQ₁(α₁))/nQ₂(nα₂(α₁)*α₁))
        else
            return r₁(α₁) + (1-α₁)/(1-nα₂(α₁)*α₁)*param.λ₁ᵖ*((nQ₂(nα₂(α₁)*α₁)-nQ₁(α₁))/nQ₂(nα₂(α₁)*α₁))
        end
    end

    #Restrictions. These restrictions are only checked for α₀ for computational reasons below
    isnQ₁feasible = r₁ᶠ(α₀) >= 0.0 && nα₂star(α₀) <= nα₂(α₀) && nα₂(α₀) <= 1.0/α₀;
    isμpositive = μ(α₀) >= 0.0;

    #= These are checks for a bunch of state varibles, but not calculated because what we will care about is α₀ below
    a = range(0.0,1.0,length=1000);
    isnQ₁feasible = sum(r₁ᶠ.(a) .< 0.0) == 0 && (sum(nα₂.(a) .< nα₂star.(a)) == 0 && sum(nα₂.(a) .> 1.0) == 0);
    isμpositive = sum(μ.(a) .< 0.0) == 0;
    =#

    return ModelSolved(ωbar=ωbar, α₀=α₀, nQ₁=nQ₁, nα₂star=nα₂star, ωᵒstar=ωᵒstar, isnQ₁feasible=isnQ₁feasible, isμpositive=isμpositive, nα₂=nα₂, μ=μ, ωᵒ=ωᵒ, α₁dot=α₁dot, r₁=r₁, r₁ᶠ=r₁ᶠ)
end

#This is dynamic solver for PMP. For a given α₂(α₁) function of macroprudential policy, this function finds the required equivalent monetary policy nQ₁(α₁).
function solver(model::PMPInitial, model_mac::ModelSolved, param::Params)

    ωbar                = model.ωbar;
    nα₂                 = model.nα₂;
    α₀                  = model.α₀;
    dt, t₁, t₂, t₃, t   = param.dt, param.t₁, param.t₂, param.t₃, param.t;

    nα₂star(α₁) = param.λ₁ᵒ/λ₁bar(α₁);
    ωᵒstar(α₁)  = nQ₂(α₁*nα₂star(α₁))/(nQ₂(α₁*nα₂star(α₁))-nQ₁(α₁))*(nα₂star(α₁)-1)+1.0;

    #nQ₁, μ and ωᵒ
    nQ₁(α₁) = model_mac.ωᵒ(α₁)<model_mac.ωbar ? model_mac.nQ₁(α₁) : nQ₂(nα₂(α₁)*α₁)*(1-(nα₂(α₁)-1)/(ωbar-1)); #Appendix page 46
    ωᵒ(α₁) = model_mac.ωᵒ(α₁)<model_mac.ωbar ? model_mac.ωᵒ(α₁) : ωbar; #If macroprudential model is not binding, so is not PMP model. If macroprudential model is binding, PMP model binds at its own leverage limit ωbar
    μ(α₁) = model_mac.μ(α₁)*(nQ₂(nα₂(α₁)*α₁)-nQ₁(α₁))/(nQ₂(nα₂(α₁)*α₁)-model_mac.nQ₁(α₁));

    #The rest
    α₁dot(α₁)   = model_mac.α₁dot(α₁);
    r₁(α₁)      = param.ρ+g₁-δ(nQ₁(α₁)*ηstar)+ForwardDiff.derivative(nQ₁,α₁)/nQ₁(α₁)*α₁dot(α₁);
    function r₁ᶠ(α₁)
        if α₁ == 0.0 || α₁ == 1.0
            return r₁(α₁) + λ₁bar(α₁)*((nQ₂(nα₂(α₁)*α₁)-nQ₁(α₁))/nQ₂(nα₂(α₁)*α₁))
        else
            return r₁(α₁) + (1-α₁)/(1-nα₂(α₁)*α₁)*param.λ₁ᵖ*((nQ₂(nα₂(α₁)*α₁)-nQ₁(α₁))/nQ₂(nα₂(α₁)*α₁))
        end
    end

    #Restrictions. These restrictions are only checked for α₀ for computational reasons below
    isnQ₁feasible = r₁ᶠ(α₀) >= 0.0 && nα₂star(α₀) <= nα₂(α₀) && nα₂(α₀) <= 1.0/α₀;
    isμpositive = μ(α₀) >= 0.0;

    #= These are checks for a bunch of state varibles, but not calculated because what we will care about is α₀ below
    a = range(0.0,1.0,length=1000);
    isnQ₁feasible = sum(r₁ᶠ.(a) .< 0.0) == 0 && (sum(nα₂.(a) .< nα₂star.(a)) == 0 && sum(nα₂.(a) .> 1.0) == 0);
    isμpositive = sum(μ.(a) .< 0.0) == 0;
    =#

    return ModelSolved(ωbar=ωbar, α₀=α₀, nQ₁=nQ₁, nα₂star=nα₂star, ωᵒstar=ωᵒstar, isnQ₁feasible=isnQ₁feasible, isμpositive=isμpositive, nα₂=nα₂, μ=μ, ωᵒ=ωᵒ, α₁dot=α₁dot, r₁=r₁, r₁ᶠ=r₁ᶠ)
end

#After solving for the euilibrium, this function simulates the series for a bunch of model variables through time
function sequenceSolver!(model::ModelSolved, param::Params)

    dt, t₁, t₂, t₃, t   = param.dt, param.t₁, param.t₂, param.t₃, param.t;

    #Simulate αₜ
    αₜ = fill(model.α₀, length(t));
    αₜ[Int(t₁*(1/dt)+1)]                    = model.α₀;                                                                                                        #Initial point
    αₜ[Int(t₁*(1/dt)+1):Int(t₂*(1/dt)+1)]   = DifferentialEquations.solve(ODEProblem((α₁,p,t) -> model.α₁dot(α₁), αₜ[Int(t₁*(1/dt)+1)], (t₁,t₂)), saveat=dt).u; #s=1
    αₜ[Int(t₂*(1/dt)+1)]                    = model.α₂(αₜ[Int(t₂*(1/dt))]);                                                                                     #s=2 initial point after transition
    αₜ[Int(t₂*(1/dt)+1):Int(t₃*(1/dt)+1)]   = DifferentialEquations.solve(ODEProblem((α₂,p,t) -> model.α₂dot(α₂), αₜ[Int(t₂*(1/dt)+1)], (t₂,t₃)), saveat=dt).u; #s=2
    αₜ[Int(t₃*(1/dt)+1):end]                .= model.α₃(αₜ[Int(t₃*(1/dt))]);                                                                                    #s=3

    #Simulate nQₜ
    nQₜ = similar(αₜ);
    nQₜ[Int(t₁*(1/dt)+1):Int(t₂*(1/dt)+1)] = model.nQ₁.(αₜ[Int(t₁*(1/dt)+1):Int(t₂*(1/dt)+1)]); #s=1
    nQₜ[Int(t₂*(1/dt)+1):Int(t₃*(1/dt)+1)] = model.nQ₂.(αₜ[Int(t₂*(1/dt)+1):Int(t₃*(1/dt)+1)]); #s=2
    nQₜ[Int(t₃*(1/dt)+1):end]              = model.nQ₃.(αₜ[Int(t₃*(1/dt)+1):end]);              #s=3

    #Simulate rₜᶠ
    rₜᶠ = similar(αₜ);
    rₜᶠ[Int(t₁*(1/dt)+1):Int(t₂*(1/dt)+1)] = model.r₁ᶠ.(αₜ[Int(t₁*(1/dt)+1):Int(t₂*(1/dt)+1)]); #s=1
    rₜᶠ[Int(t₂*(1/dt)+1):Int(t₃*(1/dt)+1)] = model.r₂ᶠ.(αₜ[Int(t₂*(1/dt)+1):Int(t₃*(1/dt)+1)]); #s=2
    rₜᶠ[Int(t₃*(1/dt)+1):end]              = model.r₃ᶠ.(αₜ[Int(t₃*(1/dt)+1):end]);              #s=3

    #Simulate rₜ
    rₜ = similar(αₜ);
    rₜ[Int(t₁*(1/dt)+1):Int(t₂*(1/dt)+1)] = model.r₁.(αₜ[Int(t₁*(1/dt)+1):Int(t₂*(1/dt)+1)]);   #s=1
    rₜ[Int(t₂*(1/dt)+1):Int(t₃*(1/dt)+1)] = model.r₂.(αₜ[Int(t₂*(1/dt)+1):Int(t₃*(1/dt)+1)]);   #s=2
    rₜ[Int(t₃*(1/dt)+1):end]              = model.r₃.(αₜ[Int(t₃*(1/dt)+1):end]);                #s=3

    #Update
    model.αₜ = αₜ;
    model.nQₜ = nQₜ;
    model.rₜᶠ = rₜᶠ;
    model.rₜ = rₜ;
end

#This function solves for the gap value function in s=1, w₁(α₁)
function welfareSolver!(model::ModelSolved, param::Params)

    #See my notes on how I construct the following linear system.
    a = first(a2);
    b = last(a2);
    n = length(a2)-1; #Number of intervals
    h = (b-a)/n;
    α = range(a, b, length=n+1); #This is actually equal to a2 grid
    w₀ = (W(model.nQ₁(α[1]))+λ₁pl_def*model.w₂(model.α₂(α[1])))/(param.ρ+λ₁pl_def);
    wₙ = (W(model.nQ₁(α[end]))+λ₁pl_def*model.w₂(model.α₂(α[end])))/(param.ρ+λ₁pl_def);

    bb = zeros(n-1);
    for (i,α₁) in enumerate(α[2:end-1])
        bb[i] = -(param.ρ+λ₁pl_def)/model.α₁dot(α₁);
    end

    aa = (1/(2*h))*ones(n-2);
    minusaa = -aa;

    A = Tridiagonal(minusaa, bb, aa);
    B = zeros(n-1);
    for (i,α₁) in enumerate(α[2:end-1])
        if i == 1
            B[i] = -(W(model.nQ₁(α₁))+λ₁pl_def*model.w₂(model.α₂(α₁)))/model.α₁dot(α₁)+(1/(2*h))*w₀;
        elseif i == n-1
            B[i] = -(W(model.nQ₁(α₁))+λ₁pl_def*model.w₂(model.α₂(α₁)))/model.α₁dot(α₁)-(1/(2*h))*wₙ;
        else
            B[i] = -(W(model.nQ₁(α₁))+λ₁pl_def*model.w₂(model.α₂(α₁)))/model.α₁dot(α₁);
        end
    end

    w = A\B;
    w₁ = [w₀; w; wₙ];

    function w₁fun(α₁)
        if α₁==a
            return w₀
        elseif α₁==b
            return wₙ
        else
            i = (α₁-a)/h+1;
            itp = interpolate(w,BSpline(Linear()));
            return itp(i)
        end
    end
    model.w₁ = w₁fun;
end

############ 4. CREATE MODELS AND SOLVE THEM
function ωbarSolver!(F, ωbar)
    model = BaselineInitial(ωbar=ωbar[1], α₀=0.85, nQ₁=(α₁)->1.0);
    model = solver(model, param);
    F[1] = model.ωᵒstar(0.5)-ωbar[1];
end
ωbar_def = nlsolve(ωbarSolver!, [9.0]).zero[1]; #ωbar_def is the default leverage limit that binds α₁=0.5. In the paper this value is 9.03, but I find 9.09. This difference must be due to the numerical solution that I implement for nQ₂. To find nQ₂, I solve an initial value problem by the package DifferentialEquations. Actually, that was a two boundary problem. I could have solved the exact two boundary problem without using the package.

#Baseline model
model1 = BaselineInitial(ωbar=ωbar_def, α₀=0.85, nQ₁=(α₁)->1.0);
model1 = solver(model1, param);
sequenceSolver!(model1, param);
welfareSolver!(model1, param);

#Macroprudential policy
model2 = BaselineInitial(ωbar=0.75*ωbar_def, α₀=0.85, nQ₁=(α₁)->1.0);
model2 = solver(model2, param);
sequenceSolver!(model2, param);
welfareSolver!(model2, param);

#PMP wrt model2
model3 = PMPInitial(ωbar=ωbar_def, nα₂=(α₁)->model2.nα₂(α₁), α₀=0.85);
model3 = solver(model3, model2, param);
sequenceSolver!(model3, param);
welfareSolver!(model3, param);

############ 5. FIGURES 3, 4, 5 AND 6
function plot_f3_f4_f5()
    #5.0. Figure 3 (page 18): normalized capital price in high-risk-premium state
    f3 = plot(legend=false, title="normalized capital price in high-risk-premium state", xlims=(0,1), xticks=0:0.1:1, ylims=(0.9,1), yticks=0.9:0.01:1, xlabel="α", ylabel="Q₂(α)/Q*")
    plot!(f3, (x)->1.0, range(0.0, 1.0, length=100), linecolor=:blue, linestyle=:dash, linewidth=2)
    plot!(f3, nQ₂base, linecolor=:blue, linestyle=:solid)
    savefig(f3, "figures\\figure3\\normalized capital price in high-risk-premium state.png")
    display(f3)

    #5.1. Figure 4: Optimal leverage
    f41 = plot(legend=false, title="optimist's leverage ratio", xlims=(0.4,0.9), xticks=0.4:0.1:0.9, ylims=(6.5,9.5), yticks=6.5:0.5:9.5, xlabel="α", ylabel="ωᵒ₁(α)")
    plot!(f41, model1.ωᵒ, a, linecolor=:red, linestyle=:dash)
    plot!(f41, model2.ωᵒ, a, linecolor=:black, linestyle=:dashdot, linewidth=3)
    plot!(f41, model3.ωᵒ, a, linecolor=:blue, linestyle=:solid)
    savefig(f41, "figures\\figure4\\optimist's leverage ratio.png")
    display(f41)

    #5.2. Figure 4: Normalized wealth after transition
    f42 = plot(legend=false, title="normalized wealth after transition", xlims=(0.4,0.9), xticks=0.4:0.1:0.9, ylims=(0.1,0.8), yticks=0.1:0.1:0.8, xlabel="α", ylabel="α₂(α)/α")
    plot!(f42, (α₁) -> model1.nα₂(α₁), a, linecolor=:red, linestyle=:dash)
    plot!(f42, (α₁) -> model2.nα₂(α₁), a, linecolor=:black, linestyle=:dashdot, linewidth=3)
    plot!(f42, (α₁) -> model3.nα₂(α₁), a, linecolor=:blue, linestyle=:solid)
    savefig(f42, "figures\\figure4\\normalized wealth after transition.png")
    display(f42)

    #5.3. Figure 4: Wealth growth
    f43 = plot(legend=false, title="wealth growth", xlims=(0.4,0.9), xticks=0.4:0.1:0.9, ylims=(0.05,0.5), yticks=0.05:0.05:0.5, xlabel="α", ylabel="α̇/α")
    plot!(f43, (α₁) -> model1.α₁dot(α₁)/α₁, a, linecolor=:red, linestyle=:dash)
    plot!(f43, (α₁) -> model2.α₁dot(α₁)/α₁, a, linecolor=:black, linestyle=:dashdot, linewidth=3)
    plot!(f43, (α₁) -> model3.α₁dot(α₁)/α₁, a, linecolor=:blue, linestyle=:solid)
    savefig(f43, "figures\\figure4\\wealth growth.png")
    display(f43)

    #5.4. Figure 4: Normalized price
    f44 = plot(legend=false, title="normalized price", xlims=(0.4,0.9), xticks=0.4:0.1:0.9, ylims=(0.9,1.0), yticks=0.9:0.02:1.0, xlabel="α", ylabel="Q₁(α)/Q*")
    plot!(f44, model1.nQ₁, a, linecolor=:red, linestyle=:dash)
    plot!(f44, model2.nQ₁, a, linecolor=:black, linestyle=:dashdot, linewidth=3)
    plot!(f44, model3.nQ₁, a, linecolor=:blue, linestyle=:solid)
    savefig(f44, "figures\\figure4\\normalized price.png")
    display(f44)

    #5.5. Figure 4: Normalized price after transition
    f45 = plot(legend=false, title="normalized price after transition", xlims=(0.4,0.9), xticks=0.4:0.1:0.9, ylims=(0.9,1.0), yticks=0.9:0.02:1.0, xlabel="α", ylabel="Q₂(α₂(α))/Q*")
    plot!(f45, (α₁) -> model1.nQ₂(model1.α₂(α₁)), a, linecolor=:red, linestyle=:dash)
    plot!(f45, (α₁) -> model2.nQ₂(model2.α₂(α₁)), a, linecolor=:black, linestyle=:dashdot, linewidth=3)
    plot!(f45, (α₁) -> model3.nQ₂(model3.α₂(α₁)), a, linecolor=:blue, linestyle=:solid)
    savefig(f45, "figures\\figure4\\normalized price after transition.png")
    display(f45)

    #5.6. Figure 4: Policy interest rate
    f46 = plot(legend=false, title="policy interest rate", xlims=(0.4,0.9), xticks=0.4:0.1:0.9, ylims=(0.03,0.1), yticks=0.03:0.01:0.1, xlabel="α", ylabel="r₁ᶠ(α)")
    plot!(f46, model1.r₁ᶠ, a, linecolor=:red, linestyle=:dash)
    plot!(f46, model2.r₁ᶠ, a, linecolor=:black, linestyle=:dashdot, linewidth=3)
    plot!(f46, model3.r₁ᶠ, a, linecolor=:blue, linestyle=:solid)
    savefig(f46, "figures\\figure4\\policy interest rate.png")
    display(f46)

    #Figure 5: optimist's wealth share
    f51 = plot(legend=false, title="optimist's wealth share", xlims=(0,1), xticks=0:0.1:1, ylims=(0,1), yticks=0:0.2:1, xlabel="time", ylabel="αₜ", size=(800,250))
    plot!(f51, param.t, model1.αₜ, linecolor=:red, linestyle=:dash)
    plot!(f51, param.t, model2.αₜ, linecolor=:black, linestyle=:dashdot, linewidth=3)
    plot!(f51, param.t, model3.αₜ, linecolor=:blue, linestyle=:solid)
    savefig(f51, "figures\\figure5\\optimist's wealth share.png")
    display(f51)

    #Figure 5: normalized price of capital
    f52 = plot(legend=false, title="normalized price of capital", xlims=(0,1), xticks=0:0.1:1, ylims=(0.9,1), yticks=0.9:0.02:1, xlabel="time", ylabel="Qₜ/Q*", size=(800,250))
    plot!(f52, param.t, model1.nQₜ, linecolor=:red, linestyle=:dash)
    plot!(f52, param.t, model2.nQₜ, linecolor=:black, linestyle=:dashdot, linewidth=3)
    plot!(f52, param.t, model3.nQₜ, linecolor=:blue, linestyle=:solid)
    savefig(f52, "figures\\figure5\\normalized price of capital.png")
    display(f52)

    #Figure 5: policy interest rate
    f53 = plot(legend=false, title="policy interest rate", xlims=(0,1), xticks=0:0.1:1, ylims=(0.0,0.1), yticks=0.0:0.02:0.1, xlabel="time", ylabel="rₜᶠ", size=(800,250))
    plot!(f53, param.t, model1.rₜᶠ, linecolor=:red, linestyle=:dash)
    plot!(f53, param.t, model2.rₜᶠ, linecolor=:black, linestyle=:dashdot, linewidth=3)
    plot!(f53, param.t, model3.rₜᶠ, linecolor=:blue, linestyle=:solid)
    savefig(f53, "figures\\figure5\\policy interest rate.png")
    display(f53)
end
plot_f3_f4_f5()

function plot_f6(α₀=0.85, N=15)
    α₀ = 0.85; #Initial α in the paper
    N = 15;
    mac_models = [BaselineInitial(ωbar=l, α₀=α₀, nQ₁=(α₁)->1.0) for l in range(2.7, ωbar_def, length=N)];
    mac_models_solved = Vector{ModelSolved}(undef, N);
    for (i,model) in enumerate(mac_models)
        x = solver(model, param);
        welfareSolver!(x, param);
        mac_models_solved[i] = x;
    end

    pmp_models = [PMPInitial(ωbar=ωbar_def, nα₂=(α₁)->model.nα₂(α₁), α₀=α₀) for model in mac_models_solved];
    pmp_models_solved = similar(mac_models_solved);
    for (i,model) in enumerate(pmp_models)
        x = solver(model, mac_models_solved[i], param);
        welfareSolver!(x, param);
        pmp_models_solved[i] = x;
    end

    mac_w1 = [model.w₁(α₀) for model in mac_models_solved];
    pmp_w1 = [model.w₁(α₀) for model in pmp_models_solved];

    f61 = plot(legend=:bottomright, title="gap value according to planner's beliefs", xlims=(2.7, ωbar_def), xticks=3:1:9, xflip=true, xlabel="leverage limit (inverted scale)", ylabel="w₁ᵖˡ(α₀)")
    plot!(f61, range(2.7, ωbar_def, length=N), mac_w1, linecolor=:black, linestyle=:dash, linewidth=2.0, label="macroprudential")
    plot!(f61, range(2.7, ωbar_def, length=N), pmp_w1, linecolor=:blue, linestyle=:solid, linewidth=2.0, label="prudential monetary")
    savefig(f61, "figures\\figure6\\gap value according to planner's beliefs.png")
    display(f61)

    return (mac_models_solved, pmp_models_solved)
end
mac_models_solved, pmp_models_solved = plot_f6()



############ 6. OPTIMAL PMP
W(nQ) = log(nQ)-(1/param.ρ)*(δ(nQ*ηstar)-δ(ηstar));

#Bellman operator with respect to explicit method when updating the value function
function T(w₁::AbstractArray; ωbar::Float64, λ₁pl::Float64, a::AbstractArray, b::AbstractArray, param::Params=param, Δ, compute_policy=false)

    ρ = param.ρ;
    λ₁ᵒ = param.λ₁ᵒ;
    λ₁ᵖ = param.λ₁ᵖ;
    λ₂ᵒ = param.λ₂ᵒ;
    λ₂ᵖ = param.λ₂ᵖ;

    #Costruct the derivative vector for the middle points i=1,...,n-1. For details, see notes. w₁ and a has n+1 points, but w₁′ has n-1 points. It corresponds to a[2:end-1]
    w₁′ = (w₁[3:end]-w₁[1:end-2])/(2*step(a));
    #w₁′ = (w₁[3:end]-w₁[2:end-1])/(step(a));

    w₁ = w₁[2:end-1];
    function objective(α,nQ₁,i)
        model = BaselineInitial(ωbar=ωbar, α₀=α, nQ₁=(α₁) -> α₁==α ? nQ₁ : 1.0);
        model = solver(model, param);

        if nQ₂base(model.α₂(α))[1] > nQ₁
            return -Inf
        elseif !model.isnQ₁feasible || !model.isμpositive
            return -Inf
        else
            return w₁[i]+Δ*(W(nQ₁) + w₁′[i]*model.α₁dot(α) + λ₁pl*w₂base(model.α₂(α))[1]-(ρ+λ₁pl)*w₁[i])
        end
    end

    OBJ = [objective(α,nQ₁,i) for (i,α) in enumerate(a[2:end-1]), nQ₁ in b];
    OBJmax, nQ₁_ind = findmax(OBJ, dims=2);
    nQ₁_ind = [ind[2] for ind in nQ₁_ind];

    w₁_first = λ₁pl*w₂(0.0)/(ρ+λ₁pl);
    w₁_end = λ₁pl*w₂(1.0)/(ρ+λ₁pl);

    Tw₁ = [w₁_first; OBJmax; w₁_end]
    nQ₁vec = [1.0; b[nQ₁_ind]; 1.0];

    if compute_policy
        return Tw₁, nQ₁vec, nQ₁_ind
    else
        return Tw₁
    end
end

#Main optimal PMP solver
function optPMPSolver(; ωbar::Float64=ωbar_def, λ₁pl::Float64=λ₁pl_def, param::Params=param, a::AbstractArray, b::AbstractArray, Δ::Float64=0.009, tol::Float64=1e-5, maxiter::Int64=200)

    ωbar_label = round(ωbar, digits=2);
    λ₁pl_label = round(λ₁pl, digits=2);

    w₁_first = λ₁pl*w₂(0.0)/(param.ρ+λ₁pl);
    w₁_end = λ₁pl*w₂(1.0)/(param.ρ+λ₁pl);

    local w₁ #Initial guess for w₁ gap value function

    try
        w₁_opt = load(string("tmp\\opt_pmp_", @sprintf("%.2f",ωbar_label), "_", @sprintf("%.2f",λ₁pl_label), ".jld"), "w₁_opt");

        function w₁_opt_fun(α₁)
            if α₁==0.0
                return w₁_first
            elseif α₁==1.0
                return w₁_end
            else
                i = (α₁-0.0)/(1.0/(length(w₁_opt)-1))+1;
                itp = interpolate(w₁_opt[:,1], BSpline(Linear()));
                return itp(i)
            end
        end

        w₁ = w₁_opt_fun.(a);
    catch
        w₁ = range(w₁_first, w₁_end, length=length(a));
    end

    diff = tol+1;
    iter = 0;

    while diff>tol && iter<maxiter
        iter = iter+1;
        println("Iteration = $iter")
        w₁next = T(w₁, ωbar=ωbar, λ₁pl=λ₁pl, a=a, b=b, param=param, Δ=Δ, compute_policy=false);
        diff = maximum(abs.(w₁-w₁next));
        w₁ = w₁next;
    end

    w₁_opt , nQ₁_opt, ~ = T(w₁, ωbar=ωbar, λ₁pl=λ₁pl, a=a, b=b, param=param, Δ=Δ, compute_policy=true);
    save(string("tmp\\opt_pmp_", @sprintf("%.2f",ωbar_label), "_", @sprintf("%.2f",λ₁pl_label), ".jld"), "w₁_opt", w₁_opt, "nQ₁_opt", nQ₁_opt)

    nQ₁_green = similar(a);
    for (i,α) in enumerate(a)
        nα₂star = param.λ₁ᵒ/λ₁bar(α);
        nQ₁_green[i] = nQ₂(nα₂star*α)*(1-(nα₂star-1.0)/(ωbar-1.0));
    end

    return w₁_opt, nQ₁_opt, nQ₁_green, iter, diff
end

#I also tried Howard's algorithm to improve computation time, but did not achieve much.
function T_howard(w₁::AbstractArray, nQ₁_ind, n_howard; ωbar::Float64, λ₁pl::Float64, a::AbstractArray, b::AbstractArray, param::Params=param, compute_policy=false)

    ρ = param.ρ;
    λ₁ᵒ = param.λ₁ᵒ;
    λ₁ᵖ = param.λ₁ᵖ;
    λ₂ᵒ = param.λ₂ᵒ;
    λ₂ᵖ = param.λ₂ᵖ;

    #Costruct the derivative vector for the middle points i=1,...,n-1. For details, see notes. w₁ and a has n+1 points, but w₁′ has n-1 points. It corresponds to a[2:end-1]
    w₁′ = (w₁[3:end]-w₁[1:end-2])/(2*step(a));
    #w₁′ = (w₁[3:end]-w₁[2:end-1])/(step(a));

    w₁ = w₁[2:end-1];
    function objective(α,nQ₁,i)
        model = BaselineInitial(ωbar=ωbar, α₀=α, nQ₁=(α₁) -> α₁==α ? nQ₁ : 1.0);
        model = solver(model, param);

        if nQ₂base(model.α₂(α))[1] > nQ₁
            return -Inf
        elseif !model.isnQ₁feasible || !model.isμpositive
            return -Inf
        else
            return w₁[i]+Δ*(W(nQ₁) + w₁′[i]*model.α₁dot(α) + λ₁pl*w₂base(model.α₂(α))[1]-(ρ+λ₁pl)*w₁[i])
        end
    end

    local w_updated
    for j in 1:n_howard
        w_updated = [objective(α,b[nQ₁_ind[i]],i) for (i,α) in enumerate(a[2:end-1])];
        w₁ = w_updated;
    end

    w₁_first = λ₁pl*w₂(0.0)/(ρ+λ₁pl);
    w₁_end = λ₁pl*w₂(1.0)/(ρ+λ₁pl);

    Tw₁ = [w₁_first; w_updated; w₁_end];

    return Tw₁
end

#Optimal PMP solver for Howard's algorithm. In the code I actually do not use it
function optPMPSolver_howard(; ωbar::Float64=ωbar_def, λ₁pl::Float64=λ₁pl, param::Params=param, a::AbstractArray, b::AbstractArray, Δ::Float64=0.009, tol::Float64=1e-5, maxiter::Int64=200, n_howard::Int64=100)

    ωbar_label = round(ωbar, digits=2);
    λ₁pl_label = round(λ₁pl, digits=2);

    w₁_first = λ₁pl*w₂(0.0)/(param.ρ+λ₁pl);
    w₁_end = λ₁pl*w₂(1.0)/(param.ρ+λ₁pl);

    local w₁

    try
        w₁_opt = load(string("tmp\\opt_pmp_", @sprintf("%.2f",ωbar_label), "_", @sprintf("%.2f",λ₁pl_label), ".jld"), "w₁_opt");

        function w₁_opt_fun(α₁)
            if α₁==0.0
                return w₁_first
            elseif α₁==1.0
                return w₁_end
            else
                i = (α₁-0.0)/(1.0/(length(w₁_opt)-1))+1;
                itp = interpolate(w₁_opt[:,1], BSpline(Linear()));
                return itp(i)
            end
        end

        w₁ = w₁_opt_fun.(a);
    catch
        w₁ = range(w₁_first, w₁_end, length=length(a));
    end

    diff = tol+1;
    iter = 0;

    while diff>tol && iter<maxiter
        iter = iter+1;
        println("Iteration = $iter")
        w₁next1, ~, ind = T(w₁, ωbar=ωbar, λ₁pl=λ₁pl, a=a, b=b, param=param, compute_policy=true);
        w₁next = T_howard(w₁next1, ind, n_howard, ωbar=ωbar, λ₁pl=λ₁pl, a=a, b=b, param=param);
        diff = maximum(abs.(w₁-w₁next));
        w₁ = w₁next;
    end

    w₁_opt , nQ₁_opt, ~ = T(w₁, ωbar=ωbar, λ₁pl=λ₁pl, a=a, b=b, param=param, compute_policy=true);
    save(string("tmp\\opt_pmp_", @sprintf("%.2f",ωbar_label), "_", @sprintf("%.2f",λ₁pl_label), ".jld"), "w₁_opt", w₁_opt, "nQ₁_opt", nQ₁_opt)

    nQ₁_green = similar(a);
    for (i,α) in enumerate(a)
        nα₂star = param.λ₁ᵒ/λ₁bar(α);
        nQ₁_green[i] = nQ₂(nα₂star*α)*(1-(nα₂star-1.0)/(ωbar-1.0));
    end

    return w₁_opt, nQ₁_opt, nQ₁_green, iter, diff
end

#Instead of relying on a finite difference algorithm, I have also tried Julia's JuMP package to find the optimal nQ₁. But because this is a non-linear system, I needed to take derivatives of objective and constraint functions which is algebraically very complicated. Below I also present that solver for the records
W(nQ) = log(nQ)-(1/param.ρ)*(δ(nQ*ηstar)-δ(ηstar));
W′(nQ) = 1/nQ-(1/param.ρ)*δ′(nQ*ηstar)*ηstar;
W′′(nQ) = (-1)*nQ^(-2)-(1/param.ρ)*δ′′(nQ*ηstar)*ηstar^2;

function T_test(w₁::AbstractArray; ωbar::Float64, λ₁pl::Float64, a::AbstractArray, b::AbstractArray, param::Params=param, compute_policy=false)

    ρ = param.ρ;
    λ₁ᵒ = param.λ₁ᵒ;
    λ₁ᵖ = param.λ₁ᵖ;
    λ₂ᵒ = param.λ₂ᵒ;
    λ₂ᵖ = param.λ₂ᵖ;

    w₁′ = (w₁[3:end]-w₁[1:end-2])/(2*step(a));
    w₁ = w₁[2:end-1];

    Tw₁′ = zeros(length(a[2:end-1]));
    nQ₁vec′ = zeros(length(a[2:end-1]));

    #For loop over the middle points
    for (i,α) in enumerate(a[2:end-1])

        #Objective functions
        function objective(nQ₁)
            model = BaselineInitial(ωbar=ωbar, α₀=α, nQ₁=(α₁) -> α₁==α ? nQ₁ : 1.0);
            model = solver(model, param);

            if !model.isnQ₁feasible || !model.isμpositive
                return -Inf
            else
                return w₁[i]+Δ*(W(nQ₁) + w₁′[i]*model.α₁dot(α) + λ₁pl*w₂base(model.α₂(α))[1]-(ρ+λ₁pl)*w₁[i])
            end
        end

        function objective′(nQ₁)
            model = BaselineInitial(ωbar=ωbar, α₀=α, nQ₁=(α₁) -> α₁==α ? nQ₁ : 1.0);
            model = solver(model, param);

            α₂ = model.α₂(α);

            α₁dot′ = -λ₁ᵖ*(1-α)*(1+α-2*α₂)/(1-α₂)^2;

            nQ₂ = nQ₂base(α₂)[1];
            nQ₂′ = nQ₂/((1-α₂)*α₂*(λ₂ᵒ-λ₂ᵖ))*(ρ+g₂-δ(nQ₂*ηstar)+λ₂bar(α₂)*(1-nQ₂));

            pay = nQ₂*(α₂/α-1.0);
            payda = (nQ₂′*nQ₁*(α₂/α-1.0)-nQ₂/α*(nQ₂-nQ₁));
            α₂′ = model.ωᵒstar(α)>ωbar ? pay/payda : 0.0;

            w₂ = w₂base(α₂)[1];
            w₂′ = ((ρ+λ₂pl)*w₂-W(nQ₂))/((λ₂bar(α₂)-λ₂ᵒ)*α₂);

            return Δ*(W′(nQ₁) + w₁′[i]*α₁dot′*α₂′ + λ₁pl*w₂′*α₂′)
        end

        function objective′′(nQ₁)
            model = BaselineInitial(ωbar=ωbar, α₀=α, nQ₁=(α₁) -> α₁==α ? nQ₁ : 1.0);
            model = solver(model, param);

            α₂ = model.α₂(α);

            α₁dot′ = -λ₁ᵖ*(1-α)*(1+α-2*α₂)/(1-α₂)^2;
            α₁dot′′ = 2*λ₁pl*(1-α)*(α₂-α)/(1-α₂)^3;

            nQ₂ = nQ₂base(α₂)[1];
            nQ₂′ = nQ₂/((1-α₂)*α₂*(λ₂ᵒ-λ₂ᵖ))*(ρ+g₂-δ(nQ₂*ηstar)+λ₂bar(α₂)*(1-nQ₂));

            pay = nQ₂*(α₂/α-1.0);
            payda = (nQ₂′*nQ₁*(α₂/α-1.0)-nQ₂/α*(nQ₂-nQ₁));
            α₂′ = model.ωᵒstar(α)>ωbar ? pay/payda : 0.0;

            nQ₂′′ = ((nQ₂′*(1-α₂)*α₂*(λ₂ᵒ-λ₂ᵖ) - (1.0-2.0*α₂)*(λ₂ᵒ-λ₂ᵖ)*nQ₂)/((1-α₂)*α₂*(λ₂ᵒ-λ₂ᵖ))^2) * (ρ+g₂-δ(nQ₂*ηstar)+λ₂bar(α₂)*(1-nQ₂)) + (-δ′(nQ₂*ηstar)*nQ₂′*ηstar + (λ₂ᵒ-λ₂ᵖ)*(1-nQ₂) - nQ₂′*λ₂bar(α₂)) * (nQ₂/((1-α₂)*α₂*(λ₂ᵒ-λ₂ᵖ)));

            pay_turev = nQ₂′*α₂′*(α₂/α-1.0) + α₂′/α*nQ₂;
            payda_turev = nQ₁*(nQ₂′′*α₂′*(α₂/α-1.0) + α₂′/α*nQ₂′) + nQ₂′*(α₂/α-1.0) - 1/α*(nQ₂′*α₂′*(nQ₂-nQ₁) + (nQ₂′*α₂′-1.0)*nQ₂);
            α₂′′ = model.ωᵒstar(α)>ωbar ? (pay_turev*payda - payda_turev*pay)/payda^2 : 0.0;

            w₂ = w₂base(α₂)[1];
            w₂′ = ((ρ+λ₂pl)*w₂-W(nQ₂))/((λ₂bar(α₂)-λ₂ᵒ)*α₂);
            w₂′′ = (((ρ+λ₂pl)*w₂′ - W′(nQ₂)*nQ₂′)*(λ₂bar(α₂)-λ₂ᵒ)*α₂ - (λ₂ᵒ-λ₂ᵖ)*(2.0*α₂-1.0)*((ρ+λ₂pl)*w₂ - W(nQ₂)))/((λ₂bar(α₂)-λ₂ᵒ)*α₂)^2;

            return Δ*(W′′(nQ₁) + w₁′[i]*(α₁dot′′*(α₂′)^2 + α₂′′*α₁dot′) + λ₁pl*(w₂′′*(α₂′)^2 + α₂′′*w₂′))
        end

        #Constraint-1 functions
        function constraint1(nQ₁)
            model = BaselineInitial(ωbar=ωbar, α₀=α, nQ₁=(α₁) -> α₁==α ? nQ₁ : 1.0);
            model = solver(model, param);

            α₂ = model.α₂(α);

            nQ₂ = nQ₂base(α₂)[1];

            return nQ₂-nQ₁
        end

        function constraint1′(nQ₁)
            model = BaselineInitial(ωbar=ωbar, α₀=α, nQ₁=(α₁) -> α₁==α ? nQ₁ : 1.0);
            model = solver(model, param);

            α₂ = model.α₂(α);

            nQ₂ = nQ₂base(α₂)[1];
            nQ₂′ = nQ₂/((1-α₂)*α₂*(λ₂ᵒ-λ₂ᵖ))*(ρ+g₂-δ(nQ₂*ηstar)+λ₂bar(α₂)*(1-nQ₂));

            pay = nQ₂*(α₂/α-1.0);
            payda = (nQ₂′*nQ₁*(α₂/α-1.0)-nQ₂/α*(nQ₂-nQ₁));
            α₂′ = model.ωᵒstar(α)>ωbar ? pay/payda : 0.0;

            return nQ₂′*α₂′-1.0
        end

        function constraint1′′(nQ₁)
            model = BaselineInitial(ωbar=ωbar, α₀=α, nQ₁=(α₁) -> α₁==α ? nQ₁ : 1.0);
            model = solver(model, param);

            α₂ = model.α₂(α);

            nQ₂ = nQ₂base(α₂)[1];
            nQ₂′ = nQ₂/((1-α₂)*α₂*(λ₂ᵒ-λ₂ᵖ))*(ρ+g₂-δ(nQ₂*ηstar)+λ₂bar(α₂)*(1-nQ₂));

            pay = nQ₂*(α₂/α-1.0);
            payda = (nQ₂′*nQ₁*(α₂/α-1.0)-nQ₂/α*(nQ₂-nQ₁));
            α₂′ = model.ωᵒstar(α)>ωbar ? pay/payda : 0.0;

            nQ₂′′ = ((nQ₂′*(1-α₂)*α₂*(λ₂ᵒ-λ₂ᵖ) - (1.0-2.0*α₂)*(λ₂ᵒ-λ₂ᵖ)*nQ₂)/((1-α₂)*α₂*(λ₂ᵒ-λ₂ᵖ))^2) * (ρ+g₂-δ(nQ₂*ηstar)+λ₂bar(α₂)*(1-nQ₂)) + (-δ′(nQ₂*ηstar)*nQ₂′*ηstar + (λ₂ᵒ-λ₂ᵖ)*(1-nQ₂) - nQ₂′*λ₂bar(α₂)) * (nQ₂/((1-α₂)*α₂*(λ₂ᵒ-λ₂ᵖ)));

            pay_turev = nQ₂′*α₂′*(α₂/α-1.0) + α₂′/α*nQ₂;
            payda_turev = nQ₁*(nQ₂′′*α₂′*(α₂/α-1.0) + α₂′/α*nQ₂′) + nQ₂′*(α₂/α-1.0) - 1/α*(nQ₂′*α₂′*(nQ₂-nQ₁) + (nQ₂′*α₂′-1.0)*nQ₂);
            α₂′′ = model.ωᵒstar(α)>ωbar ? (pay_turev*payda - payda_turev*pay)/payda^2 : 0.0;

            return nQ₂′′*(α₂′)^2 + α₂′′*nQ₂′
        end

        #Constraint-2 functions
        function constraint2(nQ₁)
            return nQ₁-1.0
        end

        function constraint2′(nQ₁)
            return 1.0
        end

        function constraint2′′(nQ₁)
            return 0.0
        end

        #Optimize
        optmodel = Model(Ipopt.Optimizer);
        @variable(optmodel, nQ₂base(0.0)[1] <= nQ₁ <=1.0, start=0.995);
        register(optmodel, :objective, 1, objective, objective′, objective′′);
        register(optmodel, :constraint1, 1, constraint1, constraint1′, constraint1′′);
        register(optmodel, :constraint2, 1, constraint2, constraint2′, constraint2′′);

        @NLobjective(optmodel, Max, objective(nQ₁));
        @NLconstraint(optmodel, constraint1, constraint1(nQ₁) <= 0.0);
        @NLconstraint(optmodel, constraint2, constraint2(nQ₁) <= 0.0);
        optimize!(optmodel);

        #Finishing
        if has_values(optmodel)
            nQ₁vec′[i] = value(nQ₁);
            Tw₁′[i] = objective_value(optmodel);
        else
            return nothing
        end
    end

    #Update
    w₁_first = λ₁pl*w₂(0.0)/(ρ+λ₁pl);
    w₁_end = λ₁pl*w₂(1.0)/(ρ+λ₁pl);

    Tw₁ = [w₁_first; Tw₁′; w₁_end]
    nQ₁vec = [1.0; nQ₁vec′; 1.0];

    #Finish
    if compute_policy
        return Tw₁, nQ₁vec
    else
        return Tw₁
    end
end

function optPMPSolver_test(; ωbar::Float64=ωbar_def, λ₁pl::Float64=λ₁pl, param::Params=param, a::AbstractArray, b::AbstractArray, Δ::Float64=0.009, tol::Float64=1e-5, maxiter::Int64=200)

    ωbar_label = round(ωbar, digits=2);
    λ₁pl_label = round(λ₁pl, digits=2);

    w₁_first = λ₁pl*w₂(0.0)/(param.ρ+λ₁pl);
    w₁_end = λ₁pl*w₂(1.0)/(param.ρ+λ₁pl);

    local w₁
    try
        #w₁_opt = load(string("tmp\\opt_pmp_", @sprintf("%.2f",ωbar_label), "_", @sprintf("%.2f",λ₁pl_label), ".jld"), "w₁_opt");
        w₁_opt = load("tmp\\opt_pmp_12.50_0.50.jld", "w₁_opt");

        function w₁_opt_fun(α₁)
            if α₁==0.0
                return w₁_first
            elseif α₁==1.0
                return w₁_end
            else
                i = (α₁-0.0)/(1.0/(length(w₁_opt)-1))+1;
                itp = interpolate(w₁_opt[:,1], BSpline(Linear()));
                return itp(i)
            end
        end

        w₁ = w₁_opt_fun.(a3);
    catch
        w₁ = range(w₁_first, w₁_end, length=length(a3));
    end

    diff = tol+1;
    iter = 0;

    while diff>tol && iter<maxiter
        iter = iter+1;
        println("Iteration = $iter")
        w₁next = T_test(w₁, ωbar=ωbar, λ₁pl=λ₁pl, a=a, b=b, param=param, compute_policy=false);
        diff = maximum(abs.(w₁-w₁next));
        w₁ = w₁next;
    end

    w₁_opt , nQ₁_opt = T_test(w₁, ωbar=ωbar, λ₁pl=λ₁pl, a=a, b=b, param=param, compute_policy=true);

    #save(string("tmp\\opt_pmp_", @sprintf("%.2f",ωbar_label), "_", @sprintf("%.2f",λ₁pl_label), ".jld"), "w₁_opt", w₁_opt, "nQ₁_opt", nQ₁_opt)
    save("tmp\\opt_pmp_13.50_0.50.jld", "w₁_opt", w₁_opt, "nQ₁_opt", nQ₁_opt)

    nQ₁_green = similar(a);
    for (i,α) in enumerate(a)
        nα₂star = param.λ₁ᵒ/λ₁bar(α);
        nQ₁_green[i] = nQ₂(nα₂star*α)*(1-(nα₂star-1.0)/(ωbar-1.0));
    end

    return w₁_opt, nQ₁_opt, nQ₁_green, iter, diff
end



############ 7. FIGURE 7. In this section I compute optimal PMP with ωbar_def=9.09 (benchmark leverage limit) and λ₁pl_def=0.495 (benchmark planner's belief)
function plot_f7(nQ₁_opt, nQ₁_green, a; ωbar, λ₁pl, param)
    f71 = plot(legend=:bottomleft, title="normalized price of capital", xlims=(0.0, 1.0), xticks=0.0:0.1:1.0, ylims = (0.975, 1), yticks=0.975:0.005:1, xlabel="wealth share of optimists, α", ylabel="Q₁(α)/Q*")
    plot!(f71, a, fill(1.0, length(a)), linecolor=:red, linestyle=:dash, linewidth=2.0, label="limit 9.09")
    plot!(f71, a, nQ₁_opt, linecolor=:blue, linestyle=:solid, linewidth=1.5, label="optimal price")
    plot!(f71, a, nQ₁_green, linecolor=:green, linestyle=:dot, linewidth=1.5, label="price that makes 9.09 bind")
    savefig(f71, "figures\\figure7\\normalized price of capital.png")
    display(f71)

    M = zeros(length(a), 3);
    for (i,α) in enumerate(a)
        model_opt = BaselineInitial(ωbar=ωbar, α₀=α, nQ₁=(α₁) -> α₁==α ? nQ₁_opt[i] : 1.0);
        model_opt = solver(model_opt, param);
        if !model_opt.isnQ₁feasible
            println(["nQ₁ at α=$α is not feasible. Breaking!!!"])
            break
        end
        M[i,1] = model_opt.ωᵒ(α);
        M[i,2] = model_opt.nα₂(α);
        M[i,3] = model_opt.r₁ᶠ(α)-model1.r₁ᶠ(α);
    end

    f72 = plot(legend=:false, title="optimists' leverage ratio", xlims=(0.0, 1.0), xticks=0.0:0.2:1.0, ylims = (0, 10), yticks=0:2:10, ylabel="ωᵒ₁(α)")
    plot!(f72, model1.ωᵒ, a, linecolor=:red, linestyle=:dash, linewidth=1.5)
    plot!(f72, a, M[:,1], linecolor=:blue, linestyle=:solid, linewidth=1.5)
    savefig(f72, "figures\\figure7\\optimists' leverage ratio.png")
    display(f72)

    f73 = plot(legend=:false, title="normalized wealth after transition", xlims=(0.0, 1.0), xticks=0.0:0.2:1.0, ylims = (0, 1), yticks=0:0.2:1, ylabel="α₂(α)/α")
    plot!(f73, model1.nα₂, a, linecolor=:red, linestyle=:dash, linewidth=1.5)
    plot!(f73, a, M[:,2], linecolor=:blue, linestyle=:solid, linewidth=1.5)
    savefig(f73, "figures\\figure7\\normalized wealth after transition.png")
    display(f73)

    f74 = plot(legend=:false, title="relative policy interest rate", xlims=(0.0, 1.0), xticks=0.0:0.2:1.0, ylims = (0, 0.012), yticks=0:0.002:0.012, ylabel="r₁ᶠ(α)-r₁ᶠ*(α)")
    plot!(f74, (α₁) -> model1.r₁ᶠ(α₁)-model1.r₁ᶠ(α₁), a, linecolor=:red, linestyle=:dash, linewidth=3.0)
    plot!(f74, a, M[:,3], linecolor=:blue, linestyle=:solid, linewidth=1.5)
    savefig(f74, "figures\\figure7\\relative policy interest rate.png")
    display(f74)
end

a3 = range(0.0, 1.0, step=0.02); #Grid for α
b3 = range(nQ₂base(0.0)[1], 1.0, length=401); #Grid for nQ₁

w₁_opt_f7, nQ₁_opt_f7, nQ₁_green_f7, iter_f7, diff_f7 = optPMPSolver(ωbar=ωbar_def, λ₁pl=λ₁pl_def, param=param, a=a3, b=b3, Δ=0.009);
plot_f7(nQ₁_opt_f7, nQ₁_green_f7, a3, ωbar=ωbar_def, λ₁pl=λ₁pl_def, param=param)




############ 8. FIGURE 8. In this section I do comparative statics for different values of ωbar (leverage limit) and λ₁pl (planner's belief)

#8.1. For different ωbars
ωbars = range(7.5, 14.5, length=8);
nQ₁_opt_at85 = zeros(length(ωbars));
rel_int_at85 = zeros(length(ωbars));;

a3 = range(0.0, 1.0, step=0.02);
b3 = range(nQ₂base(0.0)[1], 1.0, length=401);

for (i,ωbarr) in enumerate(ωbars)
    w₁_opt_s, nQ₁_opt_s, nQ₁_green_s, iter_s, diff_s = optPMPSolver(ωbar=ωbarr, λ₁pl=λ₁pl_def, param=param, a=a3, b=b3, Δ=0.009);

    function nQ₁_opt_fun(α₁)
        if α₁==0.0
            return 1.0
        elseif α₁==1.0
            return 1.0
        else
            j = (α₁-0.0)/(1.0/(length(nQ₁_opt_s)-1))+1;
            itp = interpolate(nQ₁_opt_s[:,1], BSpline(Linear()));
            return itp(j)
        end
    end

    nQ₁_opt_at85[i] = nQ₁_opt_fun(0.85);

    model_opt_s = BaselineInitial(ωbar=ωbarr, α₀=0.85, nQ₁=(α₁) -> α₁==0.85 ? nQ₁_opt_at85[i] : 1.0);
    model_opt_s = solver(model_opt_s, param);
    if !model_opt_s.isnQ₁feasible
        println(["nQ₁ at α=0.85 for $ωbarr is not feasible. Breaking!!!"])
        break
    end
    rel_int_at85[i] = model_opt_s.r₁ᶠ(0.85)-model1.r₁ᶠ(0.85);
end

f81 = plot(legend=:false, title="optimal normalized price", xlims=(ωbars[1], ωbars[end]), xticks=14:-1:8, xflip=true, xlabel="leverage limit (inverted scale)", ylims = (minimum(nQ₁_opt_at85)-0.00005, 1+0.00005), yticks=0.98:0.005:1, ylabel="Q₁(α)/Q*")
plot!(f81, ωbars, nQ₁_opt_at85, linecolor=:blue, linestyle=:solid, linewidth=1.5)
savefig(f81, "figures\\figure8\\optimal normalized price 1.png")
display(f81)

f83 = plot(legend=:false, title="optimal relative interest rate", xlims=(ωbars[1], ωbars[end]), xticks=14:-1:8, xflip=true, xlabel="leverage limit (inverted scale)", ylims = (-0.00005, maximum(rel_int_at85)+0.00005), yticks=0:0.002:0.014, ylabel="r₁ᶠ(α)-r₁ᶠ*(α)")
plot!(f83, ωbars, rel_int_at85, linecolor=:blue, linestyle=:solid, linewidth=1.5)
savefig(f83, "figures\\figure8\\optimal relative interest rate 1.png")
display(f83)



#8.2. For different λ₁pls
λ₁pls = range(0.37, 0.62, length=8);
nQ₁_opt_at85_2 = zeros(length(λ₁pls));
rel_int_at85_2 = zeros(length(λ₁pls));

a3 = range(0.0, 1.0, step=0.02);
b3 = range(nQ₂base(0.0)[1], 1.0, length=401);

for (i,λ₁pll) in enumerate(λ₁pls)
    w₁_opt_s, nQ₁_opt_s, nQ₁_green_s, iter_s, diff_s = optPMPSolver(ωbar=ωbar_def, λ₁pl=λ₁pll, param=param, a=a3, b=b3, Δ=0.009);

    function nQ₁_opt_fun(α₁)
        if α₁==0.0
            return 1.0
        elseif α₁==1.0
            return 1.0
        else
            j = (α₁-0.0)/(1.0/(length(nQ₁_opt_s)-1))+1;
            itp = interpolate(nQ₁_opt_s[:,1], BSpline(Linear()));
            return itp(j)
        end
    end

    nQ₁_opt_at85_2[i] = nQ₁_opt_fun(0.85);

    model_opt_s = BaselineInitial(ωbar=ωbar_def, α₀=0.85, nQ₁=(α₁) -> α₁==0.85 ? nQ₁_opt_at85_2[i] : 1.0);
    model_opt_s = solver(model_opt_s, param);
    if !model_opt_s.isnQ₁feasible
        println(["nQ₁ at α=0.85 for $λ₁pll is not feasible. Breaking!!!"])
        break
    end
    rel_int_at85_2[i] = model_opt_s.r₁ᶠ(0.85)-model1.r₁ᶠ(0.85);
end

f82 = plot(legend=:false, title="optimal normalized price", xlims=(λ₁pls[1], λ₁pls[end]), xticks=0.4:0.05:0.6, xlabel="planner's belief, λ₁pl", ylims=(minimum(nQ₁_opt_at85_2), maximum(nQ₁_opt_at85_2)), yticks=range(0.9785, 0.9820, step=0.0003), ylabel="Q₁(α)/Q*")
plot!(f82, λ₁pls, nQ₁_opt_at85_2, linecolor=:blue, linestyle=:solid, linewidth=1.5)
savefig(f82, "figures\\figure8\\optimal normalized price 2.png")
display(f82)

f84 = plot(legend=:false, title="optimal relative interest rate", xlims=(λ₁pls[1], λ₁pls[end]), xticks=0.4:0.05:0.6, xlabel="planner's belief, λ₁pl", ylims = (minimum(rel_int_at85_2), maximum(rel_int_at85_2)), yticks=0.0070:0.0003:0.0110, ylabel="r₁ᶠ(α)-r₁ᶠ*(α)")
plot!(f84, λ₁pls, rel_int_at85_2, linecolor=:blue, linestyle=:solid, linewidth=1.5)
savefig(f84, "figures\\figure8\\optimal relative interest rate 2.png")
display(f84)

save("tmp\\figure8_data.jld", "nQ₁_opt_at85", nQ₁_opt_at85, "rel_int_at85", rel_int_at85, "nQ₁_opt_at85_2", nQ₁_opt_at85_2, "rel_int_at85_2", rel_int_at85_2)
