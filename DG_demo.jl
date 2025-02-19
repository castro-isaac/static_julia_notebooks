### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 05b50f9b-6c4e-4d78-889c-c899f66993b5
begin
	using Plots
	using Images
	using PlutoUI
	using RungeKutta
	using Polynomials
	using LaTeXStrings
	using LinearAlgebra
	using BlockDiagonals
	using GaussQuadrature
end

# ╔═╡ c4b3c454-d50b-42fe-9cac-88338b68c6cf


# ╔═╡ 53dbab7a-16e6-42c2-bd32-50888bbdc261


# ╔═╡ 80e71a8a-64d6-454d-8b8e-98125a91acf1


# ╔═╡ 603c4500-b173-4906-af2b-f646a2b80130


# ╔═╡ 9dd16993-48e6-4ab7-9b0f-0e8038c9bd49


# ╔═╡ cc5e88b0-60ea-4e07-8138-627c97d9d316
md"""
# Demo on the Discontinuous Galerkin(DG) method
by Isaac Castro 
"""

# ╔═╡ baa6c094-be3f-48a8-9789-2e2ded962ca2


# ╔═╡ ebf02f64-2fc5-49e3-9c44-416d053981ab


# ╔═╡ e84fe0a8-c7fc-4968-a511-8ff18dec9137


# ╔═╡ 477e05b7-7a79-42a6-940b-fa65cc2f6c2a


# ╔═╡ 5d673879-23e2-465a-97e4-e585ddb86540


# ╔═╡ e995fcfc-aa02-4210-a507-73d8fcafdf84


# ╔═╡ 50fefdea-cc50-11ef-26f2-e91cd4188755
md"""
## Contents
1. Learn about the DG appropximation space
   * demo the DG approximation for the function $sin(2 \pi x)$
   * visualize $h$ and $p$ adaptivity



2. Learn about numerical flux
   * upwind, Lax-freidrichs, central



3. Solve the advection equation $u_t + (f(u))_x = 0$
   * derive weak formulation
   * example with inflow boundary condition 



4. Solve the diffusion equation $u_t = u_{xx}$
   * derive weak formulation using LDG
   * example with periodic boundary condition



"""

# ╔═╡ 0217b9b5-736b-4db3-889c-8958b8260bb2


# ╔═╡ 56857bb6-023b-4afb-9fb1-20df8a0db495


# ╔═╡ b0c0de2d-3aee-46ba-909d-fdf989e7a842


# ╔═╡ 042409a0-9ab2-4d89-8ef0-7cb89f850bfa


# ╔═╡ 6c6609cb-3376-479e-a5aa-793e007ac118


# ╔═╡ f0c2c4d5-bd73-461a-a80b-d4c70cf28087


# ╔═╡ 88ae3924-4171-40c7-9111-c2259a873b12
md"""
## 1. Learn about the DG approximation space
Let's approximate a function on an interval $[a,b]$.

Start by discretizing $[a,b]$ into $N$ elements $I_j$. There are two DG approaches:"""

# ╔═╡ efbd26fc-f3be-4959-acbf-a479ad333e87


# ╔═╡ 786b7acc-676e-4b90-a7e9-dec548a1cfb9
md"###### Modal
On each element $I_j$ the *modal* DG supports a local function $u_j = \sum^{k_j}_{i=1} u^j_i \phi^j_i$, for some *modal* basis functions $\phi^j_i(x)$."

# ╔═╡ acedd07e-a022-4e6d-9151-bc73dc772689


# ╔═╡ f4764c10-b075-4b6b-87b9-a5798fa24e3a
md"###### Nodal
Suppose were given $k$ data points on $I_j$. A *nodal* DG defines $u_j$ as the interpolant through the $k$ nodal values of $u$ on element $I_j$. We will use Gaussian points for our nodal values to take advantage of exact integration of our local polynomial interpolant."

# ╔═╡ 415eaeab-8235-4599-be04-bfa5a186bd2c


# ╔═╡ eb82e080-51e7-405f-964c-e5b69003c551
# Lagrange interpolation basis function
function phi(x,nodes,pos)
    number_of_nodes = length(nodes); prod = 1
    for i in 1:number_of_nodes
        if i!=pos 
            prod = prod * (x-nodes[i]) / (nodes[pos] - nodes[i])    
        end
    end
    return prod
end

# ╔═╡ b0f68bee-118d-43a4-850d-4c8988d0eafc
function phi_x(x,nodes,pos)
    len = length(nodes); sum = 0
    for j in 1:len
        if j != pos
            prod = 1
            for i in 1:len
                if i!=pos && i!=j
                    prod = prod * (x-nodes[i]) / (nodes[pos] - nodes[i])    
                end
            end
            sum = sum + prod / (nodes[pos]-nodes[j])
        end
    end
    return sum
end

# ╔═╡ ac9dead8-df39-4a75-b77e-762088438203
function sendInterval(I,a,b)
    l = length(I); new = zeros(l)
    for i in 1:l
        new[i] = 0.5*(b-a)*(I[i]+1)+a
    end
    return new
end;

# ╔═╡ 8a0cb286-f90e-4593-ae0c-a2dfd2c30a2a
function getNodes(N,K,x)
    nodes = zeros(N*K); h = x[2] - x[1]
    weights_temp = zeros(N*K)
    node0,w = legendre(K)
    for i in 1:N
        indx = i*K-(K-1)
        node = sendInterval(node0,x[i],x[i+1])
        for j in 1:K
            nodes[j+indx-1] = node[j]
            weights_temp[j+indx-1] = h * w[j] / 2
        end
    end
    return nodes,weights_temp;
end

# ╔═╡ 88b92906-146a-4870-9231-f445599b59ed


# ╔═╡ 930f2201-961a-47c9-8888-0f69267e843a


# ╔═╡ 069a0c9c-a4e0-44b7-a6da-2a519a18a260


# ╔═╡ 371dd606-598f-4276-b181-dfb0cb68f2ad
load("def.png")

# ╔═╡ 88748b44-89d2-4f90-b989-4a108e977f77
md"Figure 1 - DG discretization"

# ╔═╡ e4b877e6-bd16-4b38-9061-e175ea26bcac


# ╔═╡ a7bb1389-8ee4-4c39-9279-d0f06fd52541


# ╔═╡ a0ddb8fe-49d1-48ce-bf57-7f5a22f4cfca


# ╔═╡ d0788be1-8e9a-48f3-a8ba-59dd31237072


# ╔═╡ ca02bf4a-8638-4672-8d6c-bea94acc291a


# ╔═╡ c8a7f4a8-c97d-4cab-86a5-f81a089b05c7


# ╔═╡ bd3894a5-cf5f-4db0-94f6-14dd83cdb9a7


# ╔═╡ aca8a6cc-e3c9-4a58-b5b0-8b5c266b826e


# ╔═╡ 62f45f4b-39f4-4ad7-a639-87bf55d14682
md"""
Let's use the *nodal* approach to approximate $sin(2 \pi x)$ on $[0,1]$.

Use the sliders to adjust the values for $p$ and $N$ ($h$-adaptivity).
"""

# ╔═╡ eeb27776-0b80-4bda-a37c-2a6572610de7
@bind Nx_sin Slider(3:50)

# ╔═╡ a0f83b8f-b37e-4a9b-96ee-ec3f74dd3d33
println("N = $Nx_sin")

# ╔═╡ 6a7dfa09-bb9d-42d3-ad2d-36ad444a864a
@bind K_sin Slider(2:10)

# ╔═╡ 7851f053-3a6f-409e-acb8-78ae94e0e8b9
println("K = $K_sin")

# ╔═╡ 8e19f892-7c90-4cb7-b634-1f6e1d5b2bd0
begin
# Approximate sin(2 pi x) using nodal DG
interval_discretization = 0:(1/Nx_sin):1;
grid,_ = getNodes(Nx_sin, K_sin, interval_discretization);
sin_2pix = sin.(2*pi*grid);
end;md"Show lines:"

# ╔═╡ 770c6c5a-a9ca-4991-82f5-2f6d2b1b5526
@bind lines CheckBox()

# ╔═╡ 72a89205-09d4-423e-8971-bafd599f9491
md"Figure 2 - The colored lines are polynomial interpolants of Gaussian nodal values represented by bloack dots."

# ╔═╡ d7fc2587-6827-4651-a8b1-44b3f8c1fd63


# ╔═╡ 62b95082-2c8f-41b1-abc2-58415081022f


# ╔═╡ ac77a95e-0cfe-4534-a61d-1d20c55f43f3


# ╔═╡ 5c87a9d0-6759-494f-98e5-9f12a8247662


# ╔═╡ e71eaed9-7aa6-4c1a-9479-0b1716f4bbf3
md"""
## 2. Learn about numerical flux
How does our quantity of interest (heat, energy etc... ) move between elements?

The numerical flux faciliates the movement of information between cell interfaces. At each cell interface we define a numerical flux 

$$$\begin{align} 
\hat{f}_{j + \frac{1}{2}} &= \hat{f}_{j + \frac{1}{2}}(u^-,u^+) \ \text{, where} \\ 
u^- &= \lim_{x \to x^-_{j + \frac{1}{2}}} u_{j}, \\ 
u^+ &= \lim_{x \to x^+_{j + \frac{1}{2}}} u_{j+1}. \end{align}$$$
"""

# ╔═╡ d05cf74c-4cb2-4769-9697-bb5d2de591d8


# ╔═╡ 0a3358a9-4518-42ee-9e37-2aee892cbd25


# ╔═╡ 1bfb62b8-f5cd-43d5-a734-efa5e8c4e315


# ╔═╡ be440a2e-8c91-4c01-8d43-e0e8e3e537fc


# ╔═╡ 1911802a-9e6c-459a-ba83-ae5002a4cb7e


# ╔═╡ 1e352a89-75a5-47dd-bda9-e7ca25f68444


# ╔═╡ f8405a57-66e6-49a2-aea0-91798695ef19


# ╔═╡ 13e137c8-8b42-40cb-b56e-c48972fd2402


# ╔═╡ 07d1dbaa-ec59-4150-b8be-cc6bc47832ba


# ╔═╡ 1005b5ae-7463-4216-a450-301d38c6bad2
md"#### Upwind flux in $I_j$ for $f(u)=u$"

# ╔═╡ e477482a-2230-426b-8edb-0bedbd710fce
load("upwind.png")

# ╔═╡ ac17c662-d129-4580-9cd8-5ed5c2176ac8
md"Figure 2 - Upwind flux"

# ╔═╡ 6836168c-21c5-47cb-bd56-340c41ee5078


# ╔═╡ 10760691-285a-42d4-9796-31ffa64204ec
md"""
If $f(u)=u$ information flows right. Then the numerical flux should depend only on $f(u^-)$. This is called the upwind flux:

$$$\hat{f}(u^-,u^+) = u^-.$$$
"""

# ╔═╡ ad38ba15-6f9e-4801-b2b2-9e8a2ad1cd59


# ╔═╡ 6bf2fe7c-c152-43d5-b564-bb6fe53375d1


# ╔═╡ 2f3dab67-0eae-47ed-bb74-aa49abe176c4


# ╔═╡ 4585ad4c-bcc0-4e1c-a53f-9dc40eac73e6


# ╔═╡ d963f1b7-c68f-4f34-a03d-e7768fb71aea


# ╔═╡ 63429240-b990-45cf-a3b7-047c1aa263cc


# ╔═╡ 78330b69-7f34-4b97-bf49-b7c3ce3044f4
md"#### Lax-Friedrich flux in $I_j$ for $f(u)=u$"

# ╔═╡ e520a10a-f62f-4ad9-b621-2a6d301b89ac
load("lax.png")

# ╔═╡ 60485810-bd81-40d1-a6c6-e255da4e435e
md"Figure 3 - Lax-Friedrich flux"

# ╔═╡ d787bd51-03af-4444-8117-edd4c5bbaae9


# ╔═╡ fec387d7-a0ed-4d97-9611-d4df05dce646
md"""The popular Lax-Fridreich flux considers the size of the jump discontinuity and is defined by 

$$$\hat{f}^{LF}(u^-,u^+) = \frac{1}{2} (f(u^+) + f(u^-) - \alpha(u^+ - u^-) ),$$$

where $\alpha = \max_u|f'(u)|$."""

# ╔═╡ 553bf2ca-5011-4d81-b8ae-fe1d43de1368


# ╔═╡ 6aea6194-6a9a-49d6-b9eb-7fc411d96be8


# ╔═╡ b7c71baf-b7c9-44f7-9367-464e49fbb140


# ╔═╡ b915f191-bc21-48a3-a909-e5efb3ff4c3d
md"""
## 3. Transport equation
Consider

$$$\begin{cases}
u_t + (f(u))_x = 0 \ \ \text{for} \  \ (x,t) \in [a,b] \times [0,T] \hspace{2cm} (1.1)\\
u(x,0) = u_0 \\
\text{boundary conditions}.
\end{cases}$$$
"""

# ╔═╡ 9d0b597f-f41d-4f37-bd59-49c29771d570


# ╔═╡ 82c4ab34-a8de-4534-84cc-8e423a5cdead


# ╔═╡ 7ae76709-0bb6-49e5-9ebe-29febd16dc98


# ╔═╡ 67a1d5a8-a038-42aa-94d6-5a21ddfc6a1e
md"""
### Derive the weak formulation
The *weak form* and *strong form* are two directions in starting to solve for our nodal values. Today we will talk about the *weak* form. 


To start, mulitply (1.1) by $v \in P_{k-1}$ and integrate over $I_j$.

$$$\begin{equation}
\int_{I_j} u_{j,t} v + (f(u_j))_x v \ dx = 0
\end{equation}$$$

where $u_{j,t} = \frac{\partial}{\partial t} u_j$.
"""

# ╔═╡ d982346c-f68c-4fc2-bde2-5ccbe1863473


# ╔═╡ 774a3666-9ae1-4e92-9a15-ab92f237dcb0


# ╔═╡ b9299299-4b81-4438-98ea-9fb9f9e79e39


# ╔═╡ a10b92f0-a766-4c4f-a213-e6ef82772894


# ╔═╡ 7f106397-7b43-4eea-aeaf-ab68245c9558


# ╔═╡ d4646353-9e61-448f-8479-a471529e4d4d
md"""
Then integrate by parts

$$$\begin{equation}
\int_{I_j} u_{j,t} v - f(u_j) v_x \ dx + \hat{f}_{j+\frac{1}{2}} v^-(x_{j+\frac{1}{2}}) + \hat{f}_{j-\frac{1}{2}} v^+(x_{j-\frac{1}{2}}) \ dx = 0. \hspace{1.5cm} (1.2)
\end{equation}$$$
"""

# ╔═╡ 95fd5987-9d7e-4995-a740-0d8434fe0dcb


# ╔═╡ 5e5bb57b-d5e3-4009-bc7d-9d75db659d7a


# ╔═╡ 1d62dd6d-8ccd-4fc7-bcc8-bd7020a2574a


# ╔═╡ 7d2c9313-38a5-4771-bf18-1c8ceaae7e0d
md"""
The *semi-discrete* problem is to find $u_h$ such that (1.2) holds for all $v \in P_{k-1}$ (really we just need to enforce (1.2) on a basis of $P^{k-1}$),  with initial condition $u_{0,h} =  \text{Proj}_{P_{k-1}}(u_0)$ and corresponding boundary conditions.""" 

# ╔═╡ 95c9517c-e29c-4621-b26b-34b35d4a7a6d


# ╔═╡ 772a9d75-2e23-4b19-b21c-8a4d10e7252d


# ╔═╡ 92228c4c-4fa2-4bd3-9ce5-204c27e57cd3
md"""Then for $\mathbf{U^j}=(u^j_1, \dots, u^j_k)^T$ we get the local matrix equation

$$$\mathbf{M} \mathbf{U}^j_t = \mathbf{S} \mathbf{U}^j - \vec{f} (\mathbf{U}^{j-1},\mathbf{U}^j,\mathbf{U}^{j+1}). \hspace{2cm}      $$$

where
$$$(\mathbf{M})_{ik} =\int_{I_j} \phi^j_i(x) \phi^j_k(x) \ dx, \hspace{0.5cm} (\mathbf{S})_{ik} = \int_{I_j} \phi_i'(x) \phi_k(x) \ dx,$$$
and $\vec{f}$ is the flux contribution which depends on $I_j$ and its two neighbors."""

# ╔═╡ 83f8d27f-4350-427f-9ac5-d3f033051777


# ╔═╡ 85cbd274-c240-4a61-8171-6474fb2dbc36
function getMass(N,K,nodes,weights)
    temp = zeros(K,K)
    for i in 1:K
        for j in 1:K
            nodalValues = [phi(nodes[s],nodes,i)*phi(nodes[s],nodes,j) for s in 1:K]
            temp[i,j] = dot(weights,nodalValues')
        end
    end
    return  kron(Diagonal(ones(N)),temp)
end

# ╔═╡ 722523f6-2d0b-48c5-86a8-ab40e27154d9
function getStiff(N,K,nodes,weights)
	temp = zeros(K,K)
	for i in 1:K
		for j in 1:K
			nodalValues = [phi(nodes[s],nodes,j)*phi_x(nodes[s],nodes,i) for s in 1:K]
			temp[i,j] = dot(nodalValues,weights)
		end
	end
	return kron(Diagonal(ones(N)),temp)
end

# ╔═╡ c80ca58f-5c8b-4811-9b86-7ab7f722ad4a
function getFlux(nodes,x,N,K)
	temp = zeros(N*K,N*K)
	for e in 1:N
		for i in 1:K
			for j in 1:K
				# Element e flux contribution on right 
				xR = x[e+1]
				temp[(e-1)*K+i,(e-1)*K+j] = phi(xR,nodes[(e-1)*K+1:(e-1)*K+K],i) * phi(xR,nodes[(e-1)*K+1:(e-1)*K+K],j)
				# Element e+1 flux contribution on left
				if e!=N
					xL = x[e+1]
					temp[e*K+i,(e-1)*K+j] = -phi(xL,nodes[(e)*K+1:(e)*K+K],i) * phi(xL,nodes[(e-1)*K+1:(e-1)*K+K],j) 
				end
			end
		end
	end
	return temp
end

# ╔═╡ 2d39f605-52fd-4102-9f7f-43744071bdc3


# ╔═╡ ccc8dacf-1401-4576-98bc-f9521b6f0cc3


# ╔═╡ d518bb30-6be2-4458-8972-d5cba32bbffb


# ╔═╡ 0fbc427f-c4db-41b0-9d3d-487c6420380b
md"""
Then we get the initial value problem:

$$$\mathbf{U}_t = \mathbf{M}^{-1} (\mathbf{S} \mathbf{U} - \vec{f}) \hspace{2cm} (1.3)$$$"""

# ╔═╡ 6091e1c6-ae5d-4c35-9ade-544e6aaa8129


# ╔═╡ b44532cd-497d-403d-a7e1-6e3d131b702d
md"""Now we can choose a time integrator such as Backward Euler or a Runge-Kutta scheme. 
"""

# ╔═╡ ba0658f2-5cad-4394-be17-3e4187b8a30d


# ╔═╡ d1d5e574-650f-4e2c-a166-1d46db2cdc01


# ╔═╡ 8e1310b4-a733-4a11-a64d-54a3f1e058b5
md"""
## Example 1
Consider $f(u) = u$ and $u_0 = sin(2pix)$ on $[0,1]$ with a $0$ inflow boundary condition.
"""

# ╔═╡ a1bd73cf-832d-4e9b-a231-690df092a72a
@bind K Slider(2:10)

# ╔═╡ 19374134-b4da-463c-a43d-213b0eb299cc
print("k=$K")

# ╔═╡ e1438840-4a98-4063-bfcb-bca2a13637c2
@bind Nx Slider(3:50)

# ╔═╡ df44ca25-80ee-4c47-b791-7b75911a76a2
print("N = $(Nx)")

# ╔═╡ fef4288d-70e2-431a-b1d5-53ea865a6acf
begin
# Parameters
L = 1; T = 1; 
dx = L / Nx; Nt = 50; dt = T / Nt;
x = 0:dx:1;
xG,weights = getNodes(Nx,K,x);
end;

# ╔═╡ 9b5fda3c-e862-48e2-9883-4be9c6ec0650
M = getMass(Nx,K,xG[1:K],weights[1:K]);

# ╔═╡ 02f85754-cb8c-41cc-a093-a00d1f90aec9
S = getStiff(Nx,K,xG[1:K],weights[1:K]);

# ╔═╡ 0dc093b6-83f8-4e37-9a80-72a04f868b6d
F = getFlux(xG,x,Nx,K); 

# ╔═╡ a041a450-e2e5-43e2-b7cf-ac20c9bb0c65
D = M\(S-F);

# ╔═╡ f98ba9e8-70c8-4aad-a923-79c9e5f1a78e
u0 = sin.(2*pi*xG);

# ╔═╡ 890490f3-9e8a-4f2e-8562-cc0de161d7fe
function func_eval(mesh,nodes,u,num,K)
	vec = zeros(num)
	for iter in 1:num
		sum = 0
		x = mesh[iter]
		for i in 1:K
			#@show size(u)
			sum = sum + u[i]*phi(x,nodes,i)
		end
		vec[iter] = sum
	end
	return vec
end

# ╔═╡ 031ef1ae-4434-4fd6-ab17-640cf03bdd98
begin
p = plot()
for e in 1:Nx_sin
	local_mesh = range(interval_discretization[e],interval_discretization[e+1],20)
	local_func = func_eval(local_mesh,grid[K_sin*(e-1)+1:K_sin*(e-1)+K_sin],sin_2pix[K_sin*(e-1)+1:K_sin*(e-1)+K_sin],20,K_sin)
	for f in 1:K_sin
		scatter!((grid[(e-1)*K_sin+f],sin_2pix[(e-1)*K_sin+f]), color=:black,ms=1.6)
	end
	plot!(local_mesh,local_func,ylims=[-1.1,1.1],legend=false)
	if lines && e!= 1
		plot!([x[e]; x[e]],[2; -2],color=:black,linestyle=:dash
		)
	end
end
title!("DG approximation of sin(2 pi x)")
p
end

# ╔═╡ e50b3ba1-2306-45e0-91cc-cea36156bd6a
begin
gr()
u = u0;
@gif for t in 1:Nt
	global u = (I-dt*D)\u # Backward Euler
	plot()
	for e in 1:Nx
		local_mesh = range(x[e],x[e+1],20)
		local_func = func_eval(local_mesh,xG[K*(e-1)+1:K*(e-1)+K],u[K*(e-1)+1:K*(e-1)+K],20,K)
		plot!(local_mesh,local_func,ylims=[-1.1,1.1],legend=false)
	end
	title!("DG Solution to transport equation")
end
end

# ╔═╡ 620a7de6-290c-4d70-8f92-b3cc64fbe082


# ╔═╡ 9291efa6-0fa4-4617-adb0-4ccf3b79ccb3


# ╔═╡ a9766eb1-c26e-4803-bea0-b84d7e35cc03


# ╔═╡ a5f9d5db-38a0-497d-8d10-67f6a311cc23


# ╔═╡ 184af42f-8291-4375-aa63-71124a051154


# ╔═╡ f4ef1722-6519-476b-b4b9-3980c947e908


# ╔═╡ c3f2ae27-e862-4b5d-9b4d-c940215f0001


# ╔═╡ 0cf77a75-815c-48f5-9492-dd0f43e61c8c


# ╔═╡ f3a2eeb1-c53d-457f-bda9-6dadf2aca509
md"""
# Diffusion

Consider

$$$\begin{cases}
	& u_t = u_{xx} \\
	& u(x,0) = u_0 \\
	& \text{periodic B.C}
\end{cases}$$$

The wrong thing to do is solve

$$$\begin{cases}
	& u_t = (u_x)_x \\
	& u(x,0) = u_0 \\
	& \text{periodic B.C}
\end{cases}$$$

as a transport equation treating $(u_x)$ as the flux function. The numerical scheme is consistent but there is a mild unstability hence the scheme is not convergent. 
"""

# ╔═╡ 2936f609-641a-4c88-82f7-48d5885b19d8


# ╔═╡ fb046bd7-c686-4444-affa-f4dca676fa69


# ╔═╡ 0984c49b-4be6-4aa6-8370-73096e3854df


# ╔═╡ ceb3fc6f-7aa6-4515-ad18-cb8adb658a5f


# ╔═╡ 63ab41c6-0780-4386-bff3-c8b6c77fff4e


# ╔═╡ 9e1cae15-e27f-42af-befa-85e3d2737696


# ╔═╡ 9810a3ff-c51e-4a50-abcd-2aad8fa52d14


# ╔═╡ 5ce21b5d-12ad-4221-9546-889090b6dded
md"""
Instead rewrite the PDE as the first order system

$$$\begin{cases}
	& u_t = p_x \\
	& p = u_x \\
	& \text{boundary and initial datum}
\end{cases}$$$

Now we can discretize as before. Let's jump to the matrix system:
"""

# ╔═╡ c88abbb1-b35b-4382-b8d1-56d3bae1dfd6


# ╔═╡ 84fc7364-7a7f-4353-8c21-3c9fa1192f02


# ╔═╡ 372e9497-4490-4ead-b0c2-3d8ef7127eeb


# ╔═╡ 96144de7-b9eb-4f7a-92d6-13454a1e0f0b
md"""
$$$\begin{cases}
	& M_u \vec{u}_t = (F_p-S_p)\vec{p} \\
	& M_p p = (F_u-S_u) \vec{u} \\
	& \text{boundary and initial datum}
\end{cases}$$$"""

# ╔═╡ 1a6488c8-4e1f-4e8e-8c21-dcba36a40337


# ╔═╡ 26400467-f034-4946-be90-b6ca7e82afe7


# ╔═╡ 025c265b-2cb5-4c66-affd-3b0822b727b4
md"""
Solving the second equation for $\vec{p}$ and pluggin back into the first equation we get:

$$$M_u \vec{u}_t = (F_p-S_p) M_p^{-1} (F_u-S_u) \vec{u},$$$

then we get the initial 
value problem 

$$$\vec{u}_t = M^{-1}_u (F_p-S_p) M_p^{-1} (F_u-S_u) \vec{u}.$$$
"""

# ╔═╡ 4a4ce7fa-2f8b-43ca-8ace-dfd385aef6bf


# ╔═╡ c403f097-e663-4f0a-b84c-906e7663ea67
md"""
#### Central flux
Information is flowing both ways. The simplest thing is to define the numerical flux as the average of across the discontinuity at the cell interface. This is normally denoted with curly braces by

$$$\vec{f}^{central} = \{u\} = \frac{u^+ + u^-}{2}.$$$
"""

# ╔═╡ dd73b50d-17d6-43c1-91fe-4927c1a581c6
# assuming a uniform mesh, M_j,S_j,F_j will be identical 
function getCentralFlux(nodes,xL,xR,N,K)
	temp = zeros(K,K)
	tempR = zeros(K,K)
	tempL = zeros(K,K)
	temp_full = zeros(N*K,N*K)
	for i in 1:K
		for j in 1:K
			temp[i,j] = ( (phi(xR,nodes,i)*phi(xR,nodes,j) ) - ( phi(xL,nodes,i)*phi(xL,nodes,j) ) ) / 2;
			tempR[i,j] = phi(xR,nodes,i)*phi(xL,nodes,j) / 2;
			tempL[i,j] = phi(xL,nodes,i)*phi(xR,nodes,j) / 2;
		end
	end
	for e in 1:N
		for i in 1:K
			for j in 1:K
				i1 = (e-1)*K
				i2 = (e-2)*K
				i3 = (e)*K
				temp_full[i1+i,i1+j] = temp[i,j]
				if e != 1
					temp_full[i1+i,i2+j] = -tempL[i,j]
				end
				if e != N
					temp_full[i1+i,i3+j] = tempR[i,j]
				end
			end
		end
	end
	for i in 1:K
		for j in 1:K
			temp_full[i,N*K-K+j] = -tempL[i,j]
			temp_full[N*K-K+i,j] = tempR[i,j]
		end
	end
	return temp_full
end;

# ╔═╡ 7ecea581-dadc-4b8a-8bac-681803fbf6f1


# ╔═╡ e4f496cd-3b8c-4366-826b-58eda4b5a3c8


# ╔═╡ 6e8b5ee1-bfe5-48b2-a08f-a45094c56ba5


# ╔═╡ e47f562c-7d37-41cd-8b83-552abdd7f911
md""" #### Example
Take (1.3) with $u_0=\sin(2 \pi x)$ with periodic boundary conditions.
"""

# ╔═╡ 240164e9-4697-4bc8-baa4-266b270b43ab
@bind K_d Slider(2:10)

# ╔═╡ 72e5acc7-8253-494b-b543-15b5dcf2c189
println("K = $K_d")

# ╔═╡ 6e124e4e-c6a0-4840-a532-06461078cb19
@bind N_d Slider(3:50)

# ╔═╡ c1024f33-5d1f-48e5-8063-ce67045ad07d
x_d = 0:(1/N_d):1;

# ╔═╡ 5e84076b-15c1-4615-836d-ad1ec0c96050
diff_nodes, diff_weights = getNodes(N_d,K_d,x_d);

# ╔═╡ a385ff61-749f-4a34-8f66-38c599e28a1e
M_diff = getMass(N_d,K_d,diff_nodes[1:K_d],diff_weights[1:K_d]);

# ╔═╡ ef44005b-9843-44ab-b296-8bc5da2cbaab
S_diff = getStiff(N_d,K_d,diff_nodes[1:K_d],diff_weights[1:K_d]);

# ╔═╡ 720ff7aa-0174-4aba-a8f1-1c96ec9b4b2b
flux_diff = getCentralFlux(diff_nodes[1:K_d],x_d[1],x_d[2],N_d,K_d);

# ╔═╡ a63c232d-baa8-43b9-963c-372b48572877
Diffusion = inv(M_diff)*(flux_diff-S_diff)*inv(M_diff)*(flux_diff-S_diff);

# ╔═╡ c220fd2b-a7b1-440d-921c-2236e63770ca
println("N = $N_d")

# ╔═╡ 286cb32a-660c-4675-91b7-601ef84fcdfc
begin
gr()
uD = sin.(2*pi*diff_nodes);
@gif for t in 1:Nt
	global uD = (I-0.0001*Diffusion)\uD # Backward Euler
	plot()
	for e in 1:N_d
		local_mesh = range(x_d[e],x_d[e+1],20)
		local_func = func_eval(local_mesh,diff_nodes[K_d*(e-1)+1:K_d*(e-1)+K_d],uD[K_d*(e-1)+1:K_d*(e-1)+K_d],20,K_d)
		plot!(local_mesh,local_func,ylims=[-1.1,1.1],legend=false)
	end
	title!("DG solution to heat equation")
end
end

# ╔═╡ 4b83426f-4b1f-4e4c-8bc7-fdec6770ef74


# ╔═╡ 316f7ba5-47a8-4a83-9e47-d396292df8b9


# ╔═╡ b5ebaba7-2881-49fd-995b-7b857487ec88


# ╔═╡ ba3bf92a-d0e8-4a6f-b1a1-865cd2f8e854


# ╔═╡ 7de90b27-b8c6-4a60-9a8a-0c9b8905a673


# ╔═╡ 988e4b17-7b27-40f4-b5a2-7d2c9fc2e564


# ╔═╡ 764f99c4-4007-433b-8ac4-0378a7452874


# ╔═╡ a2844202-3696-44b2-9110-4e076213c74d


# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BlockDiagonals = "0a1fb500-61f7-11e9-3c65-f5ef3456f9f0"
GaussQuadrature = "d54b0c1a-921d-58e0-8e36-89d8069c0969"
Images = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Polynomials = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
RungeKutta = "fb486d5c-30a0-4a8a-8415-a8b4ace5a6f7"

[compat]
BlockDiagonals = "~0.1.42"
GaussQuadrature = "~0.5.8"
Images = "~0.26.1"
LaTeXStrings = "~1.4.0"
Plots = "~1.40.7"
PlutoUI = "~0.7.23"
Polynomials = "~4.0.12"
RungeKutta = "~0.5.15"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.2"
manifest_format = "2.0"
project_hash = "de9a2c1e7568a5de1d90101df922bf10d5b92858"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "50c3c56a52972d78e8be9fd135bfb91c9574c140"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.1.1"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "d57bd3762d308bded22c3b82d033bff85f6195c6"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.4.0"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra"]
git-tree-sha1 = "017fcb757f8e921fb44ee063a7aafe5f89b86dd1"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.18.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = "CUDSS"
    ArrayInterfaceChainRulesCoreExt = "ChainRulesCore"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceReverseDiffExt = "ReverseDiff"
    ArrayInterfaceSparseArraysExt = "SparseArrays"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.ArrayLayouts]]
deps = ["FillArrays", "LinearAlgebra"]
git-tree-sha1 = "2bf6e01f453284cb61c312836b4680331ddfc44b"
uuid = "4c555306-a7a7-4459-81d9-ec55ddd5c99a"
version = "1.11.0"
weakdeps = ["SparseArrays"]

    [deps.ArrayLayouts.extensions]
    ArrayLayoutsSparseArraysExt = "SparseArrays"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "16351be62963a67ac4083f748fdb3cca58bfd52f"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.7"

[[deps.BandedMatrices]]
deps = ["ArrayLayouts", "FillArrays", "LinearAlgebra", "PrecompileTools"]
git-tree-sha1 = "fc8d8197de6c69ad6fd01c255b6b386ca8199331"
uuid = "aae01518-5342-5314-be14-df237901396f"
version = "1.9.0"
weakdeps = ["SparseArrays"]

    [deps.BandedMatrices.extensions]
    BandedMatricesSparseArraysExt = "SparseArrays"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.Bessels]]
git-tree-sha1 = "4435559dc39793d53a9e3d278e185e920b4619ef"
uuid = "0e736298-9ec6-45e8-9647-e4fc86a2fe38"
version = "0.2.8"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.BitTwiddlingConvenienceFunctions]]
deps = ["Static"]
git-tree-sha1 = "f21cfd4950cb9f0587d5067e69405ad2acd27b87"
uuid = "62783981-4cbd-42fc-bca8-16325de8dc4b"
version = "0.1.6"

[[deps.BlockArrays]]
deps = ["ArrayLayouts", "FillArrays", "LinearAlgebra"]
git-tree-sha1 = "b406207917260364a2e0287b42e4c6772cb9db88"
uuid = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
version = "1.3.0"
weakdeps = ["BandedMatrices"]

    [deps.BlockArrays.extensions]
    BlockArraysBandedMatricesExt = "BandedMatrices"

[[deps.BlockDiagonals]]
deps = ["ChainRulesCore", "FillArrays", "FiniteDifferences", "LinearAlgebra"]
git-tree-sha1 = "920d3775e35c519a2aced9e7bbe9ac61218eeead"
uuid = "0a1fb500-61f7-11e9-3c65-f5ef3456f9f0"
version = "0.1.42"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "8873e196c2eb87962a2048b3b8e08946535864a1"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+4"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CPUSummary]]
deps = ["CpuId", "IfElse", "PrecompileTools", "Static"]
git-tree-sha1 = "5a97e67919535d6841172016c9530fd69494e5ec"
uuid = "2a0fbf3d-bb9c-48f3-b0a9-814d99fd7ab9"
version = "0.2.6"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "009060c9a6168704143100f36ab08f06c2af4642"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.2+1"

[[deps.CatIndices]]
deps = ["CustomUnitRanges", "OffsetArrays"]
git-tree-sha1 = "a0f80a09780eed9b1d106a1bf62041c2efc995bc"
uuid = "aafaddc9-749c-510e-ac4f-586e18779b91"
version = "0.2.2"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "3e4b134270b372f2ed4d4d0e936aabaefc1802bc"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.25.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.CloseOpenIntervals]]
deps = ["Static", "StaticArrayInterface"]
git-tree-sha1 = "05ba0d07cd4fd8b7a39541e31a7b0254704ea581"
uuid = "fb6a15b2-703c-40df-9091-08a04967cfa9"
version = "0.1.13"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "Random", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "9ebb045901e9bbf58767a9f34ff89831ed711aae"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.15.7"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "bce6804e5e6044c6daab27bb533d1295e4a2e759"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.6"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "c785dfb1b3bfddd1da557e861b919819b82bbe5b"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.27.1"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "c7acce7a7e1078a20a285211dd73cd3941a871d6"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.0"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "64e15186f0aa277e174aa81798f7eb8598e0157e"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.0"

[[deps.CommonWorldInvalidations]]
git-tree-sha1 = "ae52d1c52048455e85a387fbee9be553ec2b68d0"
uuid = "f70d9fcc-98c5-4d4a-abd7-e4cdeebd8ca8"
version = "1.0.0"

[[deps.CompactBasisFunctions]]
deps = ["ContinuumArrays", "FastTransforms", "LinearAlgebra", "OffsetArrays", "QuadratureRules", "StaticArrays"]
git-tree-sha1 = "988eb64bb3cd0e4230f7ef0a5de95dace6e9f5f5"
uuid = "a09551c4-f815-4143-809e-dd1986753ba7"
version = "0.2.12"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "8ae8d32e09f0dcf42a36b90d4e17f5dd2e4c4215"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.16.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.CompositeTypes]]
git-tree-sha1 = "bce26c3dab336582805503bed209faab1c279768"
uuid = "b152e2b5-7a66-4b01-a709-34e65c35f657"
version = "0.1.4"

[[deps.ComputationalResources]]
git-tree-sha1 = "52cb3ec90e8a8bea0e62e275ba577ad0f74821f7"
uuid = "ed09eef8-17a6-5b46-8889-db040fac31e3"
version = "0.3.2"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "f36e5e8fdffcb5646ea5da81495a5a7566005127"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.4.3"

[[deps.ConstructionBase]]
git-tree-sha1 = "76219f1ed5771adbb096743bff43fb5fdd4c1157"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.8"
weakdeps = ["IntervalSets", "LinearAlgebra", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.ContinuumArrays]]
deps = ["AbstractFFTs", "ArrayLayouts", "BandedMatrices", "BlockArrays", "DomainSets", "FillArrays", "InfiniteArrays", "Infinities", "IntervalSets", "LazyArrays", "LinearAlgebra", "QuasiArrays", "StaticArrays"]
git-tree-sha1 = "9c5f95903dbce1f72cb626d198da529e1980045f"
uuid = "7ae1f121-cc2c-504b-ac30-9b923412ae5c"
version = "0.18.6"

    [deps.ContinuumArrays.extensions]
    ContinuumArraysMakieExt = "Makie"
    ContinuumArraysRecipesBaseExt = "RecipesBase"

    [deps.ContinuumArrays.weakdeps]
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.CoordinateTransformations]]
deps = ["LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "f9d7112bfff8a19a3a4ea4e03a8e6a91fe8456bf"
uuid = "150eb455-5306-5404-9cee-2592286d6298"
version = "0.6.3"

[[deps.CpuId]]
deps = ["Markdown"]
git-tree-sha1 = "fcbb72b032692610bfbdb15018ac16a36cf2e406"
uuid = "adafc99b-e345-5852-983c-f28acb93d879"
version = "0.3.1"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.CustomUnitRanges]]
git-tree-sha1 = "1a3f97f907e6dd8983b744d2642651bb162a3f7a"
uuid = "dc8bdbbb-1ca9-579f-8c36-e416f6a65cce"
version = "1.0.2"

[[deps.DSP]]
deps = ["Bessels", "FFTW", "IterTools", "LinearAlgebra", "Polynomials", "Random", "Reexport", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "489db9d78b53e44fb753d225c58832632d74ab10"
uuid = "717857b8-e6f2-59f4-9121-6e50c889abd2"
version = "0.8.0"
weakdeps = ["OffsetArrays"]

    [deps.DSP.extensions]
    OffsetArraysExt = "OffsetArrays"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1d0a14036acb104d9e89698bd408f63ab58cdc82"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.20"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fc173b380865f70627d7dd1190dc2fce6cc105af"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.14.10+0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "c7e3a542b999843086e2f29dac96a618c105be1d"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.12"
weakdeps = ["ChainRulesCore", "SparseArrays"]

    [deps.Distances.extensions]
    DistancesChainRulesCoreExt = "ChainRulesCore"
    DistancesSparseArraysExt = "SparseArrays"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.DomainSets]]
deps = ["CompositeTypes", "IntervalSets", "LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "490392af2c7d63183bfa2c8aaa6ab981c5ba7561"
uuid = "5b8099bc-c8ec-5219-889f-1d9e522a28bf"
version = "0.7.14"

    [deps.DomainSets.extensions]
    DomainSetsMakieExt = "Makie"

    [deps.DomainSets.weakdeps]
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a4be429317c42cfae6a7fc03c31bad1970c310d"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+1"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "d36f682e590a83d63d1c7dbd287573764682d12a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.11"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e51db81749b0777b2147fbe7b783ee79045b8e99"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.6.4+3"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "53ebe7511fa11d33bec688a9178fac4e49eeee00"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.2"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "74faea50c1d007c85837327f6775bea60b5492dd"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+2"

[[deps.FFTViews]]
deps = ["CustomUnitRanges", "FFTW"]
git-tree-sha1 = "cbdf14d1e8c7c8aacbe8b19862e0179fd08321c2"
uuid = "4f61f5a4-77b1-5117-aa51-3ab5ef4ef0cd"
version = "0.3.2"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "4820348781ae578893311153d69049a93d05f39d"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.8.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4d81ed14783ec49ce9f2e168208a12ce1815aa25"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+3"

[[deps.FastGaussQuadrature]]
deps = ["LinearAlgebra", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "fd923962364b645f3719855c88f7074413a6ad92"
uuid = "442a2c76-b920-505d-bb47-c5924d526838"
version = "1.0.2"

[[deps.FastTransforms]]
deps = ["AbstractFFTs", "ArrayLayouts", "BandedMatrices", "FFTW", "FastGaussQuadrature", "FastTransforms_jll", "FillArrays", "GenericFFT", "LazyArrays", "Libdl", "LinearAlgebra", "RecurrenceRelationships", "Reexport", "SpecialFunctions", "ToeplitzMatrices"]
git-tree-sha1 = "e914dd1c91d1909f6584d61d767fd0c4f64fcba3"
uuid = "057dd010-8810-581a-b7be-e3fc3b93f78c"
version = "0.16.8"

[[deps.FastTransforms_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "FFTW_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl", "MPFR_jll", "OpenBLAS_jll"]
git-tree-sha1 = "efb41482692019ed03e0de67b9e48e88c0504e7d"
uuid = "34b6f7d7-08f9-5794-9e10-3819e4c7e49a"
version = "0.6.3+0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "2dd20384bf8c6d411b5c7370865b1e9b26cb2ea3"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.6"
weakdeps = ["HTTP"]

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "6a70198746448456524cb442b8af316927ff3e1a"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.13.0"

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

    [deps.FillArrays.weakdeps]
    PDMats = "90014a1f-27ba-587c-ab20-58faa44d9150"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.FiniteDifferences]]
deps = ["ChainRulesCore", "LinearAlgebra", "Printf", "Random", "Richardson", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "06d76c780d657729cf20821fb5832c6cc4dfd0b5"
uuid = "26cc04aa-876d-5657-8c51-4c34ba976000"
version = "0.12.32"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "21fac3c77d7b5a9fc03b0ec503aa1a6392c34d2b"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.15.0+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "786e968a8d2fb167f2e4880baba62e0e26bd8e4e"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.3+1"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "846f7026a9decf3679419122b49f8a1fdb48d2d5"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.16+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "fcb0584ff34e25155876418979d4c8971243bb89"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.0+2"

[[deps.GMP_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "781609d7-10c4-51f6-84f2-b8444358ff6d"
version = "6.3.0+0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "8e2d86e06ceb4580110d9e716be26658effc5bfd"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.72.8"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "da121cbdc95b065da07fbb93638367737969693f"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.72.8+0"

[[deps.GaussQuadrature]]
deps = ["SpecialFunctions"]
git-tree-sha1 = "eb6f1f48aa994f3018cbd029a17863c6535a266d"
uuid = "d54b0c1a-921d-58e0-8e36-89d8069c0969"
version = "0.5.8"

[[deps.GenericFFT]]
deps = ["AbstractFFTs", "FFTW", "LinearAlgebra", "Reexport"]
git-tree-sha1 = "1bc01f2ea9a0226a60723794ff86b8017739f5d9"
uuid = "a8297547-1b15-4a5a-a998-a2ac5f1cef28"
version = "0.1.6"

[[deps.GenericLinearAlgebra]]
deps = ["LinearAlgebra", "Printf", "Random", "libblastrampoline_jll"]
git-tree-sha1 = "c4f9c87b74aedf20920034bd4db81d0bffc527d2"
uuid = "14197337-ba66-59df-a3e3-ca00e7dcff7a"
version = "0.3.14"

[[deps.GeometricBase]]
git-tree-sha1 = "a3f7967d9be5da465de9e8536951f46d1ef14f0d"
uuid = "9a0b12b7-583b-4f04-aa1f-d8551b6addc9"
version = "0.10.11"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "43ba3d3c82c18d88471cfd2924931658838c9d8f"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.0+4"

[[deps.Giflib_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6570366d757b50fabae9f4315ad74d2e40c0560a"
uuid = "59f7168a-df46-5410-90c8-f2779963d0ec"
version = "5.2.3+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "b0036b392358c80d2d2124746c2bf3d48d457938"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.82.4+0"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "a641238db938fff9b2f60d08ed9030387daf428c"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.3"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "01979f9b37367603e2848ea225918a3b3861b606"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+1"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "Compat", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "1dc470db8b1131cfc7fb4c115de89fe391b9e780"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.12.0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "PrecompileTools", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "c67b33b085f6e2faf8bf79a61962e7339a81129c"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.15"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "55c53be97790242c29031e5cd45e8ac296dadda3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.0+0"

[[deps.HistogramThresholding]]
deps = ["ImageBase", "LinearAlgebra", "MappedArrays"]
git-tree-sha1 = "7194dfbb2f8d945abdaf68fa9480a965d6661e69"
uuid = "2c695a8d-9458-5d45-9878-1b8a99cf7853"
version = "0.3.1"

[[deps.HostCPUFeatures]]
deps = ["BitTwiddlingConvenienceFunctions", "IfElse", "Libdl", "Static"]
git-tree-sha1 = "8e070b599339d622e9a081d17230d74a5c473293"
uuid = "3e5b6fbb-0976-4d2c-9146-d79de83f2fb0"
version = "0.1.17"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "e12629406c6c4442539436581041d372d69c55ba"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.12"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "eb49b82c172811fd2c86759fa0553a2221feb909"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.7"

[[deps.ImageBinarization]]
deps = ["HistogramThresholding", "ImageCore", "LinearAlgebra", "Polynomials", "Reexport", "Statistics"]
git-tree-sha1 = "33485b4e40d1df46c806498c73ea32dc17475c59"
uuid = "cbc4b850-ae4b-5111-9e64-df94c024a13d"
version = "0.3.1"

[[deps.ImageContrastAdjustment]]
deps = ["ImageBase", "ImageCore", "ImageTransformations", "Parameters"]
git-tree-sha1 = "eb3d4365a10e3f3ecb3b115e9d12db131d28a386"
uuid = "f332f351-ec65-5f6a-b3d1-319c6670881a"
version = "0.3.12"

[[deps.ImageCore]]
deps = ["ColorVectorSpace", "Colors", "FixedPointNumbers", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "PrecompileTools", "Reexport"]
git-tree-sha1 = "8c193230235bbcee22c8066b0374f63b5683c2d3"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.10.5"

[[deps.ImageCorners]]
deps = ["ImageCore", "ImageFiltering", "PrecompileTools", "StaticArrays", "StatsBase"]
git-tree-sha1 = "24c52de051293745a9bad7d73497708954562b79"
uuid = "89d5987c-236e-4e32-acd0-25bd6bd87b70"
version = "0.1.3"

[[deps.ImageDistances]]
deps = ["Distances", "ImageCore", "ImageMorphology", "LinearAlgebra", "Statistics"]
git-tree-sha1 = "08b0e6354b21ef5dd5e49026028e41831401aca8"
uuid = "51556ac3-7006-55f5-8cb3-34580c88182d"
version = "0.2.17"

[[deps.ImageFiltering]]
deps = ["CatIndices", "ComputationalResources", "DataStructures", "FFTViews", "FFTW", "ImageBase", "ImageCore", "LinearAlgebra", "OffsetArrays", "PrecompileTools", "Reexport", "SparseArrays", "StaticArrays", "Statistics", "TiledIteration"]
git-tree-sha1 = "33cb509839cc4011beb45bde2316e64344b0f92b"
uuid = "6a3955dd-da59-5b1f-98d4-e7296123deb5"
version = "0.7.9"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs", "WebP"]
git-tree-sha1 = "696144904b76e1ca433b886b4e7edd067d76cbf7"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.9"

[[deps.ImageMagick]]
deps = ["FileIO", "ImageCore", "ImageMagick_jll", "InteractiveUtils"]
git-tree-sha1 = "c5c5478ae8d944c63d6de961b19e6d3324812c35"
uuid = "6218d12a-5da1-5696-b52f-db25d2ecc6d1"
version = "1.4.0"

[[deps.ImageMagick_jll]]
deps = ["Artifacts", "Ghostscript_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "OpenJpeg_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "fa01c98985be12e5d75301c4527fff2c46fa3e0e"
uuid = "c73af94c-d91f-53ed-93a7-00f77d67a9d7"
version = "7.1.1+1"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "2a81c3897be6fbcde0802a0ebe6796d0562f63ec"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.10"

[[deps.ImageMorphology]]
deps = ["DataStructures", "ImageCore", "LinearAlgebra", "LoopVectorization", "OffsetArrays", "Requires", "TiledIteration"]
git-tree-sha1 = "6f0a801136cb9c229aebea0df296cdcd471dbcd1"
uuid = "787d08f9-d448-5407-9aad-5290dd7ab264"
version = "0.4.5"

[[deps.ImageQualityIndexes]]
deps = ["ImageContrastAdjustment", "ImageCore", "ImageDistances", "ImageFiltering", "LazyModules", "OffsetArrays", "PrecompileTools", "Statistics"]
git-tree-sha1 = "783b70725ed326340adf225be4889906c96b8fd1"
uuid = "2996bd0c-7a13-11e9-2da2-2f5ce47296a9"
version = "0.3.7"

[[deps.ImageSegmentation]]
deps = ["Clustering", "DataStructures", "Distances", "Graphs", "ImageCore", "ImageFiltering", "ImageMorphology", "LinearAlgebra", "MetaGraphs", "RegionTrees", "SimpleWeightedGraphs", "StaticArrays", "Statistics"]
git-tree-sha1 = "b217d9ded4a95052ffc09acc41ab781f7f72c7ba"
uuid = "80713f31-8817-5129-9cf8-209ff8fb23e1"
version = "1.8.3"

[[deps.ImageShow]]
deps = ["Base64", "ColorSchemes", "FileIO", "ImageBase", "ImageCore", "OffsetArrays", "StackViews"]
git-tree-sha1 = "3b5344bcdbdc11ad58f3b1956709b5b9345355de"
uuid = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
version = "0.3.8"

[[deps.ImageTransformations]]
deps = ["AxisAlgorithms", "CoordinateTransformations", "ImageBase", "ImageCore", "Interpolations", "OffsetArrays", "Rotations", "StaticArrays"]
git-tree-sha1 = "e0884bdf01bbbb111aea77c348368a86fb4b5ab6"
uuid = "02fcd773-0e25-5acc-982a-7f6622650795"
version = "0.10.1"

[[deps.Images]]
deps = ["Base64", "FileIO", "Graphics", "ImageAxes", "ImageBase", "ImageBinarization", "ImageContrastAdjustment", "ImageCore", "ImageCorners", "ImageDistances", "ImageFiltering", "ImageIO", "ImageMagick", "ImageMetadata", "ImageMorphology", "ImageQualityIndexes", "ImageSegmentation", "ImageShow", "ImageTransformations", "IndirectArrays", "IntegralArrays", "Random", "Reexport", "SparseArrays", "StaticArrays", "Statistics", "StatsBase", "TiledIteration"]
git-tree-sha1 = "12fdd617c7fe25dc4a6cc804d657cc4b2230302b"
uuid = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
version = "0.26.1"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0936ba688c6d201805a83da835b55c61a180db52"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.11+0"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.InfiniteArrays]]
deps = ["ArrayLayouts", "FillArrays", "Infinities", "LazyArrays", "LinearAlgebra"]
git-tree-sha1 = "6785ca14cc6390e5ffb8c10be107c435ad6032e3"
uuid = "4858937d-0d70-526a-a4dd-2d5cb5dd786c"
version = "0.15.3"

    [deps.InfiniteArrays.extensions]
    InfiniteArraysBandedMatricesExt = "BandedMatrices"
    InfiniteArraysBlockArraysExt = "BlockArrays"
    InfiniteArraysBlockBandedMatricesExt = "BlockBandedMatrices"
    InfiniteArraysDSPExt = "DSP"
    InfiniteArraysStatisticsExt = "Statistics"

    [deps.InfiniteArrays.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    DSP = "717857b8-e6f2-59f4-9121-6e50c889abd2"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.Infinities]]
git-tree-sha1 = "acfbb29ee3796e5147cd9144a9c2ce8410044d8d"
uuid = "e1ba4f0e-776d-440f-acd9-e1d2e9742647"
version = "0.1.9"

[[deps.Inflate]]
git-tree-sha1 = "d1b1b796e47d94588b3757fe84fbf65a5ec4a80d"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.5"

[[deps.IntegralArrays]]
deps = ["ColorTypes", "FixedPointNumbers", "IntervalSets"]
git-tree-sha1 = "b842cbff3f44804a84fda409745cc8f04c029a20"
uuid = "1d092043-8f09-5a30-832f-7509e371ab51"
version = "0.1.6"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "10bd689145d2c3b2a9844005d01087cc1194e79e"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2024.2.1+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "88a101217d7cb38a7b481ccd50d21876e1d1b0e0"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.15.1"
weakdeps = ["Unitful"]

    [deps.Interpolations.extensions]
    InterpolationsUnitfulExt = "Unitful"

[[deps.IntervalSets]]
git-tree-sha1 = "dba9ddf07f77f60450fe5d2e2beb9854d9a49bd0"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.10"
weakdeps = ["Random", "RecipesBase", "Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLD2]]
deps = ["FileIO", "MacroTools", "Mmap", "OrderedCollections", "PrecompileTools", "Requires", "TranscodingStreams"]
git-tree-sha1 = "f1a1c1037af2a4541ea186b26b0c0e7eeaad232b"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.5.10"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "71b48d857e86bf7a1838c4736545699974ce79a2"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.9"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "a007feb38b422fbdab534406aeca1b86823cb4d6"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "fa6d0bcff8583bac20f1ffa708c3913ca605c611"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.5"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eac1206917768cb54957c65a615460d87b455fc1"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.1+0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "170b660facf5df5de098d866564877e119141cbd"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.2+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "78211fb6cbc872f77cad3fc0b6cf647d923f4929"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.7+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "854a9c268c43b77b0a27f22d7fab8d33cdb3a731"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.2+3"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "ce5f5621cac23a86011836badfedf664a612cee4"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.5"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LayoutPointers]]
deps = ["ArrayInterface", "LinearAlgebra", "ManualMemory", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "a9eaadb366f5493a5654e843864c13d8b107548c"
uuid = "10f19ff3-798f-405d-979b-55457f8fc047"
version = "0.1.17"

[[deps.LazyArrays]]
deps = ["ArrayLayouts", "FillArrays", "LinearAlgebra", "MacroTools", "SparseArrays"]
git-tree-sha1 = "f289bee714e11708df257c57514585863aa02b33"
uuid = "5078a376-72f3-5289-bfd5-ec5146d43c02"
version = "2.3.1"

    [deps.LazyArrays.extensions]
    LazyArraysBandedMatricesExt = "BandedMatrices"
    LazyArraysBlockArraysExt = "BlockArrays"
    LazyArraysBlockBandedMatricesExt = "BlockBandedMatrices"
    LazyArraysStaticArraysExt = "StaticArrays"

    [deps.LazyArrays.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "27ecae93dd25ee0909666e6835051dd684cc035e"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+2"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll"]
git-tree-sha1 = "8be878062e0ffa2c3f67bb58a595375eda5de80b"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.11.0+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "ff3b4b9d35de638936a525ecd36e86a8bb919d11"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a7f43994b47130e4f491c3b2dbe78fe9e2aed2b3"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.51.0+2"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "61dfdba58e585066d8bce214c5a51eaa0539f269"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+1"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "84eef7acd508ee5b3e956a2ae51b05024181dee0"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.40.2+2"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "edbf5309f9ddf1cab25afc344b1e8150b7c832f9"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.40.2+2"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.LittleCMS_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pkg"]
git-tree-sha1 = "110897e7db2d6836be22c18bffd9422218ee6284"
uuid = "d3a379c0-f9a3-5b72-a4c0-6bf4d2e8af0f"
version = "2.12.0+0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f02b56007b064fbfddb4c9cd60161b6dd0f40df3"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.1.0"

[[deps.LoopVectorization]]
deps = ["ArrayInterface", "CPUSummary", "CloseOpenIntervals", "DocStringExtensions", "HostCPUFeatures", "IfElse", "LayoutPointers", "LinearAlgebra", "OffsetArrays", "PolyesterWeave", "PrecompileTools", "SIMDTypes", "SLEEFPirates", "Static", "StaticArrayInterface", "ThreadingUtilities", "UnPack", "VectorizationBase"]
git-tree-sha1 = "8084c25a250e00ae427a379a5b607e7aed96a2dd"
uuid = "bdcacae8-1622-11e9-2a5c-532679323890"
version = "0.12.171"

    [deps.LoopVectorization.extensions]
    ForwardDiffExt = ["ChainRulesCore", "ForwardDiff"]
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.LoopVectorization.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "f046ccd0c6db2832a9f639e2c669c6fe867e5f4f"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2024.2.0+0"

[[deps.MPFR_jll]]
deps = ["Artifacts", "GMP_jll", "Libdl"]
uuid = "3a97d323-0669-5f0c-9066-3539efd106a3"
version = "4.2.1+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.ManualMemory]]
git-tree-sha1 = "bcaef4fc7a0cfe2cba636d84cda54b5e4e4ca3cd"
uuid = "d125e4d3-2237-4719-b19c-fa641b8a4667"
version = "0.1.8"

[[deps.MappedArrays]]
git-tree-sha1 = "2dab0221fe2b0f2cb6754eaa743cc266339f527e"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.2"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.MetaGraphs]]
deps = ["Graphs", "JLD2", "Random"]
git-tree-sha1 = "e9650bea7f91c3397eb9ae6377343963a22bf5b8"
uuid = "626554b9-1ddb-594c-aa3c-2596fe9399a5"
version = "0.8.0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "8a3271d8309285f4db73b4f662b1b290c715e85e"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.21"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "d92b107dbb887293622df7697a2223f9f8176fcd"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OffsetArrays]]
git-tree-sha1 = "5e1897147d1ff8d98883cda2be2187dcf57d8f0c"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.15.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "97db9e07fe2091882c765380ef58ec553074e9c7"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.3"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "8292dd5c8a38257111ada2174000a33745b06d4e"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.2.4+0"

[[deps.OpenJpeg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libtiff_jll", "LittleCMS_jll", "Pkg", "libpng_jll"]
git-tree-sha1 = "76374b6e7f632c130e78100b166e5a48464256f8"
uuid = "643b3616-a352-519d-856d-80112ee9badc"
version = "2.4.0+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "38cb508d080d21dc1128f7fb04f20387ed4c0af4"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.3"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ad31332567b189f508a3ea8957a2640b1147ab00"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.23+1"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6703a85cb3781bd5909d48730a67205f3f31a575"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.3+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "12f1439c4f986bb868acda6ea33ebc78e19b95ad"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.7.0"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "67186a2bc9a90f9f85ff3cc8277868961fb57cbd"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.4.3"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ed6834e95bd326c52d5675b4181386dfbe885afb"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.55.5+0"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "35621f10a7531bc8fa58f74610b1bfb70a3cfc6b"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.43.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f9501cc0430a26bc3d156ae1b5b0c1b47af4d6da"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.3"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "41031ef3a1be6f5bbbf3e8073f210556daeae5ca"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.3.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "3ca9a356cd2e113c420f2c13bea19f8d3fb1cb18"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.3"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "f202a1ca4f6e165238d8175df63a7e26a51e04dc"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.40.7"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "5152abbdab6488d5eec6a01029ca6697dff4ec8f"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.23"

[[deps.PolyesterWeave]]
deps = ["BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "Static", "ThreadingUtilities"]
git-tree-sha1 = "645bed98cd47f72f67316fd42fc47dee771aefcd"
uuid = "1d0040c9-8b98-4ee7-8388-3f51789ca0ad"
version = "0.2.2"

[[deps.Polynomials]]
deps = ["LinearAlgebra", "OrderedCollections", "RecipesBase", "Requires", "Setfield", "SparseArrays"]
git-tree-sha1 = "adc25dbd4d13f148f3256b6d4743fe7e63a71c4a"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "4.0.12"

    [deps.Polynomials.extensions]
    PolynomialsChainRulesCoreExt = "ChainRulesCore"
    PolynomialsFFTWExt = "FFTW"
    PolynomialsMakieCoreExt = "MakieCore"
    PolynomialsMutableArithmeticsExt = "MutableArithmetics"

    [deps.Polynomials.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    MakieCore = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
    MutableArithmetics = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "1101cd475833706e4d0e7b122218257178f48f34"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.4.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "8f6bc219586aef8baf0ff9a5fe16ee9c70cb65e4"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.10.2"

[[deps.PtrArrays]]
git-tree-sha1 = "77a42d78b6a92df47ab37e177b2deac405e1c88f"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.2.1"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "8b3fc30bc0390abdce15f8822c889f669baed73d"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.1"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "0c03844e2231e12fda4d0086fd7cbe4098ee8dc5"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+2"

[[deps.QuadratureRules]]
deps = ["FastGaussQuadrature", "GenericLinearAlgebra", "Polynomials"]
git-tree-sha1 = "17c0b16306232b309c4cb9270afba8e84dd36954"
uuid = "a08977f5-a20d-4b99-b11f-f5ea535e0575"
version = "0.1.6"

[[deps.QuasiArrays]]
deps = ["ArrayLayouts", "DomainSets", "FillArrays", "LazyArrays", "LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "c6a85acf492dacdbe64cabcc2ac03761d4c2472a"
uuid = "c4ea9172-b204-11e9-377d-29865faadc5c"
version = "0.11.9"
weakdeps = ["SparseArrays"]

    [deps.QuasiArrays.extensions]
    QuasiArraysSparseArraysExt = "SparseArrays"

[[deps.Quaternions]]
deps = ["LinearAlgebra", "Random", "RealDot"]
git-tree-sha1 = "994cc27cdacca10e68feb291673ec3a76aa2fae9"
uuid = "94ee1d12-ae83-5a48-8b1c-48b8ff168ae0"
version = "0.7.6"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.RecurrenceRelationships]]
git-tree-sha1 = "1a268b6c7f2eacf86dc345d85a88d7120c4f8103"
uuid = "807425ed-42ea-44d6-a357-6771516d7b2c"
version = "0.1.1"
weakdeps = ["FillArrays", "LazyArrays", "LinearAlgebra"]

    [deps.RecurrenceRelationships.extensions]
    RecurrenceRelationshipsFillArraysExt = "FillArrays"
    RecurrenceRelationshipsLazyArraysExt = "LazyArrays"
    RecurrenceRelationshipsLinearAlgebraExt = "LinearAlgebra"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RegionTrees]]
deps = ["IterTools", "LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "4618ed0da7a251c7f92e869ae1a19c74a7d2a7f9"
uuid = "dee08c22-ab7f-5625-9660-a9af2021b33f"
version = "0.3.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Richardson]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "48f038bfd83344065434089c2a79417f38715c41"
uuid = "708f8203-808e-40c0-ba2d-98a6953ed40d"
version = "1.4.2"

[[deps.Rotations]]
deps = ["LinearAlgebra", "Quaternions", "Random", "StaticArrays"]
git-tree-sha1 = "5680a9276685d392c87407df00d57c9924d9f11e"
uuid = "6038ab10-8711-5258-84ad-4b1120ba62dc"
version = "1.7.1"
weakdeps = ["RecipesBase"]

    [deps.Rotations.extensions]
    RotationsRecipesBaseExt = "RecipesBase"

[[deps.RungeKutta]]
deps = ["CompactBasisFunctions", "DelimitedFiles", "GenericLinearAlgebra", "GeometricBase", "LinearAlgebra", "Markdown", "Polynomials", "PrettyTables", "Reexport", "StaticArrays"]
git-tree-sha1 = "923d940ba952ee40859a262e934103e03e48f602"
uuid = "fb486d5c-30a0-4a8a-8415-a8b4ace5a6f7"
version = "0.5.15"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMD]]
deps = ["PrecompileTools"]
git-tree-sha1 = "52af86e35dd1b177d051b12681e1c581f53c281b"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.7.0"

[[deps.SIMDTypes]]
git-tree-sha1 = "330289636fb8107c5f32088d2741e9fd7a061a5c"
uuid = "94e857df-77ce-4151-89e5-788b33177be4"
version = "0.1.0"

[[deps.SLEEFPirates]]
deps = ["IfElse", "Static", "VectorizationBase"]
git-tree-sha1 = "456f610ca2fbd1c14f5fcf31c6bfadc55e7d66e0"
uuid = "476501e8-09a2-5ece-8869-fb82de89a1fa"
version = "0.6.43"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"
version = "1.11.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.SimpleWeightedGraphs]]
deps = ["Graphs", "LinearAlgebra", "Markdown", "SparseArrays"]
git-tree-sha1 = "4b33e0e081a825dbfaf314decf58fa47e53d6acb"
uuid = "47aef6b3-ad0c-573a-a1e2-d07658019622"
version = "1.4.0"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "2da10356e31327c7096832eb9cd86307a50b1eb6"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "64cca0c26b4f31ba18f13f6c12af7c85f478cfde"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.5.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "83e6cce8324d49dfaf9ef059227f91ed4441a8e5"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.2"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[deps.Static]]
deps = ["CommonWorldInvalidations", "IfElse", "PrecompileTools"]
git-tree-sha1 = "87d51a3ee9a4b0d2fe054bdd3fc2436258db2603"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "1.1.1"

[[deps.StaticArrayInterface]]
deps = ["ArrayInterface", "Compat", "IfElse", "LinearAlgebra", "PrecompileTools", "Static"]
git-tree-sha1 = "96381d50f1ce85f2663584c8e886a6ca97e60554"
uuid = "0d7ed370-da01-4f52-bd93-41d350b8b718"
version = "1.8.0"
weakdeps = ["OffsetArrays", "StaticArrays"]

    [deps.StaticArrayInterface.extensions]
    StaticArrayInterfaceOffsetArraysExt = "OffsetArrays"
    StaticArrayInterfaceStaticArraysExt = "StaticArrays"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "47091a0340a675c738b1304b58161f3b0839d454"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.10"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "29321314c920c26684834965ec2ce0dacc9cf8e5"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.4"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "a6b1675a536c5ad1a60e5a5153e1fee12eb146e3"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.0"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "598cd7c1f68d1e205689b1c2fe65a9f85846f297"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.ThreadingUtilities]]
deps = ["ManualMemory"]
git-tree-sha1 = "eda08f7e9818eb53661b3deb74e3159460dfbc27"
uuid = "8290d209-cae3-49c0-8002-c8c24d57dab5"
version = "0.5.2"

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "ProgressMeter", "SIMD", "UUIDs"]
git-tree-sha1 = "3c0faa42f2bd3c6d994b06286bba2328eae34027"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.11.2"

[[deps.TiledIteration]]
deps = ["OffsetArrays", "StaticArrayInterface"]
git-tree-sha1 = "1176cc31e867217b06928e2f140c90bd1bc88283"
uuid = "06e1c1a7-607b-532d-9fad-de7d9aa2abac"
version = "0.5.0"

[[deps.ToeplitzMatrices]]
deps = ["AbstractFFTs", "DSP", "FillArrays", "LinearAlgebra"]
git-tree-sha1 = "338d725bd62115be4ba7ffa891d85654e0bfb1a1"
uuid = "c751599d-da0a-543b-9d20-d0a503d91d24"
version = "0.8.5"
weakdeps = ["StatsBase"]

    [deps.ToeplitzMatrices.extensions]
    ToeplitzMatricesStatsBaseExt = "StatsBase"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "7822b97e99a1672bfb1b49b668a6d46d58d8cbcb"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.9"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "01915bfcd62be15329c9a07235447a89d588327c"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.21.1"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "975c354fcd5f7e1ddcc1f1a23e6e091d99e99bc8"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.4"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.VectorizationBase]]
deps = ["ArrayInterface", "CPUSummary", "HostCPUFeatures", "IfElse", "LayoutPointers", "Libdl", "LinearAlgebra", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "4ab62a49f1d8d9548a1c8d1a75e5f55cf196f64e"
uuid = "3d5dd08c-fd9d-11e8-17fa-ed2836048c2f"
version = "0.21.71"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "85c7811eddec9e7f22615371c3cc81a504c508ee"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+2"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "5db3e9d307d32baba7067b13fc7b5aa6edd4a19a"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.36.0+0"

[[deps.WebP]]
deps = ["CEnum", "ColorTypes", "FileIO", "FixedPointNumbers", "ImageCore", "libwebp_jll"]
git-tree-sha1 = "aa1ca3c47f119fbdae8770c29820e5e6119b83f2"
uuid = "e3aaa7dc-3e4b-44e0-be63-ffb868ccd7c1"
version = "0.1.3"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c1a7aa6219628fcd757dede0ca95e245c5cd9511"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.0.0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "a2fccc6559132927d4c5dc183e3e01048c6dcbd6"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.5+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "7d1671acbe47ac88e981868a078bd6b4e27c5191"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.42+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "9dafcee1d24c4f024e7edc92603cedba72118283"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+3"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "2b0e27d52ec9d8d483e2ca0b72b3cb1a8df5c27a"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+3"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "807c226eaf3651e7b2c468f687ac788291f9a89b"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.3+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "02054ee01980c90297412e4c809c8694d7323af3"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+3"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "d7155fea91a4123ef59f42c4afb5ab3b4ca95058"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.6+3"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "6fcc21d5aea1a0b7cce6cab3e62246abd1949b86"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.0+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "984b313b049c89739075b8e2a94407076de17449"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.2+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "a1a7eaf6c3b5b05cb903e35e8372049b107ac729"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.5+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "b6f664b7b2f6a39689d822a6300b14df4668f0f4"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.4+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "a490c6212a0e90d2d55111ac956f7c4fa9c277a6"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.11+1"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fee57a273563e273f0f53275101cd41a8153517a"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "1a74296303b6524a0472a8cb12d3d87a78eb3612"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "dbc53e4cf7701c6c7047c51e17d6e64df55dca94"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+1"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "ab2221d309eda71020cdda67a973aa582aa85d69"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+1"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b9ead2d2bdb27330545eb14234a2e300da61232e"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "622cf78670d067c738667aaa96c553430b65e269"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6e50f145003024df4f5cb96c7fce79466741d601"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.56.3+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1827acba325fdcdf1d2647fc8d5301dd9ba43a9d"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.9.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e17c115d55c5fbb7e52ebedb427a0dca79d4484e"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.2+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a22cf860a7d27e4f3498a0fe0811a7957badb38"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.3+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "b7bfd3ab9d2c58c3829684142f5804e4c6499abc"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.45+0"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "libpng_jll"]
git-tree-sha1 = "7dfa0fd9c783d3d0cc43ea1af53d69ba45c447df"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.3+3"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "490376214c4721cdaca654041f635213c6165cb3"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+2"

[[deps.libwebp_jll]]
deps = ["Artifacts", "Giflib_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libglvnd_jll", "Libtiff_jll", "libpng_jll"]
git-tree-sha1 = "ccbb625a89ec6195856a50aa2b668a5c08712c94"
uuid = "c5f90fcd-3b7e-5836-afba-fc50a0988cb2"
version = "1.4.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7d0ea0f4895ef2f5cb83645fa689e52cb55cf493"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2021.12.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "63406453ed9b33a0df95d570816d5366c92b7809"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+2"
"""

# ╔═╡ Cell order:
# ╟─c4b3c454-d50b-42fe-9cac-88338b68c6cf
# ╟─53dbab7a-16e6-42c2-bd32-50888bbdc261
# ╟─05b50f9b-6c4e-4d78-889c-c899f66993b5
# ╟─80e71a8a-64d6-454d-8b8e-98125a91acf1
# ╟─603c4500-b173-4906-af2b-f646a2b80130
# ╟─9dd16993-48e6-4ab7-9b0f-0e8038c9bd49
# ╟─cc5e88b0-60ea-4e07-8138-627c97d9d316
# ╟─baa6c094-be3f-48a8-9789-2e2ded962ca2
# ╟─ebf02f64-2fc5-49e3-9c44-416d053981ab
# ╟─e84fe0a8-c7fc-4968-a511-8ff18dec9137
# ╟─477e05b7-7a79-42a6-940b-fa65cc2f6c2a
# ╟─5d673879-23e2-465a-97e4-e585ddb86540
# ╟─e995fcfc-aa02-4210-a507-73d8fcafdf84
# ╟─50fefdea-cc50-11ef-26f2-e91cd4188755
# ╟─0217b9b5-736b-4db3-889c-8958b8260bb2
# ╟─56857bb6-023b-4afb-9fb1-20df8a0db495
# ╟─b0c0de2d-3aee-46ba-909d-fdf989e7a842
# ╟─042409a0-9ab2-4d89-8ef0-7cb89f850bfa
# ╟─6c6609cb-3376-479e-a5aa-793e007ac118
# ╟─f0c2c4d5-bd73-461a-a80b-d4c70cf28087
# ╟─88ae3924-4171-40c7-9111-c2259a873b12
# ╟─efbd26fc-f3be-4959-acbf-a479ad333e87
# ╟─786b7acc-676e-4b90-a7e9-dec548a1cfb9
# ╟─acedd07e-a022-4e6d-9151-bc73dc772689
# ╟─f4764c10-b075-4b6b-87b9-a5798fa24e3a
# ╟─415eaeab-8235-4599-be04-bfa5a186bd2c
# ╟─eb82e080-51e7-405f-964c-e5b69003c551
# ╟─b0f68bee-118d-43a4-850d-4c8988d0eafc
# ╟─8a0cb286-f90e-4593-ae0c-a2dfd2c30a2a
# ╟─ac9dead8-df39-4a75-b77e-762088438203
# ╟─88b92906-146a-4870-9231-f445599b59ed
# ╟─930f2201-961a-47c9-8888-0f69267e843a
# ╟─069a0c9c-a4e0-44b7-a6da-2a519a18a260
# ╟─371dd606-598f-4276-b181-dfb0cb68f2ad
# ╟─88748b44-89d2-4f90-b989-4a108e977f77
# ╟─e4b877e6-bd16-4b38-9061-e175ea26bcac
# ╟─a7bb1389-8ee4-4c39-9279-d0f06fd52541
# ╟─a0ddb8fe-49d1-48ce-bf57-7f5a22f4cfca
# ╟─d0788be1-8e9a-48f3-a8ba-59dd31237072
# ╟─ca02bf4a-8638-4672-8d6c-bea94acc291a
# ╟─c8a7f4a8-c97d-4cab-86a5-f81a089b05c7
# ╟─bd3894a5-cf5f-4db0-94f6-14dd83cdb9a7
# ╟─aca8a6cc-e3c9-4a58-b5b0-8b5c266b826e
# ╟─62f45f4b-39f4-4ad7-a639-87bf55d14682
# ╟─a0f83b8f-b37e-4a9b-96ee-ec3f74dd3d33
# ╟─eeb27776-0b80-4bda-a37c-2a6572610de7
# ╟─7851f053-3a6f-409e-acb8-78ae94e0e8b9
# ╟─6a7dfa09-bb9d-42d3-ad2d-36ad444a864a
# ╟─8e19f892-7c90-4cb7-b634-1f6e1d5b2bd0
# ╟─770c6c5a-a9ca-4991-82f5-2f6d2b1b5526
# ╟─031ef1ae-4434-4fd6-ab17-640cf03bdd98
# ╟─72a89205-09d4-423e-8971-bafd599f9491
# ╟─d7fc2587-6827-4651-a8b1-44b3f8c1fd63
# ╟─62b95082-2c8f-41b1-abc2-58415081022f
# ╟─ac77a95e-0cfe-4534-a61d-1d20c55f43f3
# ╟─5c87a9d0-6759-494f-98e5-9f12a8247662
# ╟─e71eaed9-7aa6-4c1a-9479-0b1716f4bbf3
# ╟─d05cf74c-4cb2-4769-9697-bb5d2de591d8
# ╟─0a3358a9-4518-42ee-9e37-2aee892cbd25
# ╟─1bfb62b8-f5cd-43d5-a734-efa5e8c4e315
# ╟─be440a2e-8c91-4c01-8d43-e0e8e3e537fc
# ╟─1911802a-9e6c-459a-ba83-ae5002a4cb7e
# ╟─1e352a89-75a5-47dd-bda9-e7ca25f68444
# ╟─f8405a57-66e6-49a2-aea0-91798695ef19
# ╟─13e137c8-8b42-40cb-b56e-c48972fd2402
# ╟─07d1dbaa-ec59-4150-b8be-cc6bc47832ba
# ╟─1005b5ae-7463-4216-a450-301d38c6bad2
# ╟─e477482a-2230-426b-8edb-0bedbd710fce
# ╟─ac17c662-d129-4580-9cd8-5ed5c2176ac8
# ╟─6836168c-21c5-47cb-bd56-340c41ee5078
# ╟─10760691-285a-42d4-9796-31ffa64204ec
# ╟─ad38ba15-6f9e-4801-b2b2-9e8a2ad1cd59
# ╟─6bf2fe7c-c152-43d5-b564-bb6fe53375d1
# ╟─2f3dab67-0eae-47ed-bb74-aa49abe176c4
# ╟─4585ad4c-bcc0-4e1c-a53f-9dc40eac73e6
# ╟─d963f1b7-c68f-4f34-a03d-e7768fb71aea
# ╟─63429240-b990-45cf-a3b7-047c1aa263cc
# ╟─78330b69-7f34-4b97-bf49-b7c3ce3044f4
# ╟─e520a10a-f62f-4ad9-b621-2a6d301b89ac
# ╟─60485810-bd81-40d1-a6c6-e255da4e435e
# ╟─d787bd51-03af-4444-8117-edd4c5bbaae9
# ╟─fec387d7-a0ed-4d97-9611-d4df05dce646
# ╟─553bf2ca-5011-4d81-b8ae-fe1d43de1368
# ╟─6aea6194-6a9a-49d6-b9eb-7fc411d96be8
# ╟─b7c71baf-b7c9-44f7-9367-464e49fbb140
# ╟─b915f191-bc21-48a3-a909-e5efb3ff4c3d
# ╟─9d0b597f-f41d-4f37-bd59-49c29771d570
# ╟─82c4ab34-a8de-4534-84cc-8e423a5cdead
# ╟─7ae76709-0bb6-49e5-9ebe-29febd16dc98
# ╟─67a1d5a8-a038-42aa-94d6-5a21ddfc6a1e
# ╟─d982346c-f68c-4fc2-bde2-5ccbe1863473
# ╟─774a3666-9ae1-4e92-9a15-ab92f237dcb0
# ╟─b9299299-4b81-4438-98ea-9fb9f9e79e39
# ╟─a10b92f0-a766-4c4f-a213-e6ef82772894
# ╟─7f106397-7b43-4eea-aeaf-ab68245c9558
# ╟─d4646353-9e61-448f-8479-a471529e4d4d
# ╟─95fd5987-9d7e-4995-a740-0d8434fe0dcb
# ╟─5e5bb57b-d5e3-4009-bc7d-9d75db659d7a
# ╟─1d62dd6d-8ccd-4fc7-bcc8-bd7020a2574a
# ╟─7d2c9313-38a5-4771-bf18-1c8ceaae7e0d
# ╟─95c9517c-e29c-4621-b26b-34b35d4a7a6d
# ╟─772a9d75-2e23-4b19-b21c-8a4d10e7252d
# ╟─92228c4c-4fa2-4bd3-9ce5-204c27e57cd3
# ╟─83f8d27f-4350-427f-9ac5-d3f033051777
# ╟─85cbd274-c240-4a61-8171-6474fb2dbc36
# ╟─722523f6-2d0b-48c5-86a8-ab40e27154d9
# ╟─c80ca58f-5c8b-4811-9b86-7ab7f722ad4a
# ╠═9b5fda3c-e862-48e2-9883-4be9c6ec0650
# ╠═02f85754-cb8c-41cc-a093-a00d1f90aec9
# ╠═0dc093b6-83f8-4e37-9a80-72a04f868b6d
# ╟─2d39f605-52fd-4102-9f7f-43744071bdc3
# ╟─ccc8dacf-1401-4576-98bc-f9521b6f0cc3
# ╟─d518bb30-6be2-4458-8972-d5cba32bbffb
# ╟─0fbc427f-c4db-41b0-9d3d-487c6420380b
# ╠═a041a450-e2e5-43e2-b7cf-ac20c9bb0c65
# ╟─6091e1c6-ae5d-4c35-9ade-544e6aaa8129
# ╟─b44532cd-497d-403d-a7e1-6e3d131b702d
# ╟─ba0658f2-5cad-4394-be17-3e4187b8a30d
# ╟─d1d5e574-650f-4e2c-a166-1d46db2cdc01
# ╟─8e1310b4-a733-4a11-a64d-54a3f1e058b5
# ╠═f98ba9e8-70c8-4aad-a923-79c9e5f1a78e
# ╟─19374134-b4da-463c-a43d-213b0eb299cc
# ╟─a1bd73cf-832d-4e9b-a231-690df092a72a
# ╟─df44ca25-80ee-4c47-b791-7b75911a76a2
# ╟─e1438840-4a98-4063-bfcb-bca2a13637c2
# ╟─fef4288d-70e2-431a-b1d5-53ea865a6acf
# ╟─e50b3ba1-2306-45e0-91cc-cea36156bd6a
# ╟─890490f3-9e8a-4f2e-8562-cc0de161d7fe
# ╟─620a7de6-290c-4d70-8f92-b3cc64fbe082
# ╟─9291efa6-0fa4-4617-adb0-4ccf3b79ccb3
# ╟─a9766eb1-c26e-4803-bea0-b84d7e35cc03
# ╟─a5f9d5db-38a0-497d-8d10-67f6a311cc23
# ╟─184af42f-8291-4375-aa63-71124a051154
# ╟─f4ef1722-6519-476b-b4b9-3980c947e908
# ╟─c3f2ae27-e862-4b5d-9b4d-c940215f0001
# ╟─0cf77a75-815c-48f5-9492-dd0f43e61c8c
# ╟─f3a2eeb1-c53d-457f-bda9-6dadf2aca509
# ╟─2936f609-641a-4c88-82f7-48d5885b19d8
# ╟─fb046bd7-c686-4444-affa-f4dca676fa69
# ╟─0984c49b-4be6-4aa6-8370-73096e3854df
# ╟─ceb3fc6f-7aa6-4515-ad18-cb8adb658a5f
# ╟─63ab41c6-0780-4386-bff3-c8b6c77fff4e
# ╟─9e1cae15-e27f-42af-befa-85e3d2737696
# ╟─9810a3ff-c51e-4a50-abcd-2aad8fa52d14
# ╟─5ce21b5d-12ad-4221-9546-889090b6dded
# ╟─c88abbb1-b35b-4382-b8d1-56d3bae1dfd6
# ╟─84fc7364-7a7f-4353-8c21-3c9fa1192f02
# ╟─372e9497-4490-4ead-b0c2-3d8ef7127eeb
# ╟─96144de7-b9eb-4f7a-92d6-13454a1e0f0b
# ╟─1a6488c8-4e1f-4e8e-8c21-dcba36a40337
# ╟─26400467-f034-4946-be90-b6ca7e82afe7
# ╟─025c265b-2cb5-4c66-affd-3b0822b727b4
# ╟─a385ff61-749f-4a34-8f66-38c599e28a1e
# ╟─ef44005b-9843-44ab-b296-8bc5da2cbaab
# ╟─4a4ce7fa-2f8b-43ca-8ace-dfd385aef6bf
# ╟─a63c232d-baa8-43b9-963c-372b48572877
# ╟─c403f097-e663-4f0a-b84c-906e7663ea67
# ╟─dd73b50d-17d6-43c1-91fe-4927c1a581c6
# ╟─720ff7aa-0174-4aba-a8f1-1c96ec9b4b2b
# ╟─7ecea581-dadc-4b8a-8bac-681803fbf6f1
# ╟─c1024f33-5d1f-48e5-8063-ce67045ad07d
# ╟─5e84076b-15c1-4615-836d-ad1ec0c96050
# ╟─e4f496cd-3b8c-4366-826b-58eda4b5a3c8
# ╟─6e8b5ee1-bfe5-48b2-a08f-a45094c56ba5
# ╟─e47f562c-7d37-41cd-8b83-552abdd7f911
# ╟─72e5acc7-8253-494b-b543-15b5dcf2c189
# ╟─240164e9-4697-4bc8-baa4-266b270b43ab
# ╟─c220fd2b-a7b1-440d-921c-2236e63770ca
# ╟─6e124e4e-c6a0-4840-a532-06461078cb19
# ╟─286cb32a-660c-4675-91b7-601ef84fcdfc
# ╟─4b83426f-4b1f-4e4c-8bc7-fdec6770ef74
# ╟─316f7ba5-47a8-4a83-9e47-d396292df8b9
# ╟─b5ebaba7-2881-49fd-995b-7b857487ec88
# ╟─ba3bf92a-d0e8-4a6f-b1a1-865cd2f8e854
# ╟─7de90b27-b8c6-4a60-9a8a-0c9b8905a673
# ╟─988e4b17-7b27-40f4-b5a2-7d2c9fc2e564
# ╟─764f99c4-4007-433b-8ac4-0378a7452874
# ╟─a2844202-3696-44b2-9110-4e076213c74d
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
