### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# ╔═╡ 660227ee-c714-11ef-0f53-f53572195380
md"""
In this notebook we solve the 1D variable coefficient transport problem given by
$$\begin{cases}
	u_t + (\sin(x)u)_x = 0 , (t,x) \in \mathbb{R}^+ \times [0,2\pi] \\
	u(0) = 1
\end{cases}$$

using a low rank Discontinuous Galerkin space-time method. Note the exact solution is $u(x,t) = \frac{2\arctan(\exp(-t) \tan(\frac{x}{2}))}{\sin(x)}$."""

# ╔═╡ e6b7ba9d-0b79-4199-ad80-533312b6d071
md"""
We will derive the global DG matrix system and formulate it as a Sylvester matrix equation which we will solve using the low rank krylov solver in [1]. 
### Discontinous Galerkin Formulation
Discretize the spatial domain with $N$ uniform elements.
"""

# ╔═╡ 55ed59ec-dc08-43bf-8421-ce3e6d474cba


# ╔═╡ d2369cfa-3f9d-4ae6-b9c1-35f82abc0e08


# ╔═╡ ecc3f033-9e4e-4929-ab88-401d6258eaf0


# ╔═╡ d2b6bd39-df19-484b-8f6c-4eed1079522f
md"On each element we prescribe $K$ Gaussian quadrature nodes $x_i \ (i=1,..,K)$."

# ╔═╡ 1e1eac77-df99-4cf6-b224-b142e13f0d18


# ╔═╡ 8fc7e19f-43ff-42a7-8eab-5e2509a52701
md"""On each element we will support a degree $K$ polynomial defined as the nodal interpolant of our function $u(x,t)$ through the Gaussian nodes on that element i.e $u_h(x,t) = \sum_{i=1}^K u_i(t) l_i(x)$

where $l_i(x)$ are the Lagrange polynomial interpolation basis functions corresponding to the Gaussian nodes on an element. 

Multiplying the PDE by a test function $v$ and integrating over an element we get

$$\begin{equation}
\int_{x_i}^{x_{i+1}} u_t v + (\sin(x)u)_x v \ dx = 0.
\end{equation}$$

Using integration by parts on the spatial derivative we get 

$$\begin{equation}
\int_{x_i}^{x_{i+1}} u_t v \ dx = (\sin(x)u) v_x - \hat{f}(x_{j+\frac{1}{2}}) v(x_{j+\frac{1}{2}}) + \hat{f}(x_{j-\frac{1}{2}}) v(x_{j-\frac{1}{2}})
\end{equation}$$

where $\hat{f}$ is the Lax-Freidrichs flux. 

The semi-discrete formulation is to find $u$ that satisfies the above equation for all $v$ in a finite-dimenstional test space. Substituting in our ansatz we can test over a finite length basis. This leads to the matrix equation

$$\begin{equation}
\mathbf{M} \vec{u}_t = \mathbf{S} \vec{u} - \mathbf{F} \vec{u} - \mathbf{F} \vec{u_{ext}}
\end{equation}$$

for mass, stiffness and flux matrices $\mathbf{M}, \mathbf{S}, \mathbf{F}$ respectively, defined by

$$\mathbf{M}_{ij} = \int_{x_i}^{x_{i+1}} l_i(x) l_j(x) \ dx, \hspace{1cm} \mathbf{S}_{ij} = \int_{x_i}^{x_{x+1}} l_i'(x) l_j(x) \ dx$$

and $\mathbf{F}$ accounts for flux contributions defined by the Lax-Friedrichs flux. Note that the flux depends on nodal values from neigboring elements and is represnted by the symbolic $\vec{u}_{ext}$."""

# ╔═╡ fdd8df3f-b5f3-4286-83f7-0bc3648299cb


# ╔═╡ 3d08db47-ff99-4552-a1e6-ba53d2dd5a58
md"""Extending this for every element we can form a global closed system to solve (note that we can solve local systems indiviudally if we develop a sweeping strategy).

Define the global system(recylcing notation for $\mathbf{M,S,F}) by

$$\begin{equation}
\mathbf{M} \vec{U}_t = \mathbf{S} \vec{U} - \mathbf{F} \vec{U}.
\end{equation}$$

This leads us to the initial value problem

$$\begin{gather}
\mathbf{M} \vec{U}_t = \mathbf{S} \vec{U} - \mathbf{F} \vec{U} \\
\Rightarrow \vec{U}_t = \mathbf{M}^{-1} (\mathbf{S} - \mathbf{F}) \vec{U} := \mathbf{B} \vec{U}
\end{gather}$$
"""

# ╔═╡ 610d9408-6af7-4c66-8b93-8a8f0cc07e8b


# ╔═╡ db6b4db6-6502-4bad-9cf5-e697ff406252
md"""
### Radau iia

Using the $s$ stage Radau iia RK scheme with Butcher tableau $(A,b,c)$ we can approximate $s$ stages of our solution at collocation points in time $t_n + c_1, \  \dots \ , \ t_n + c_s$ by with the RK step

$$\begin{equation}
U = U_n + \Delta t A U B^T
\end{equation}$$

where $U$ is the matrix whose $i$-th row is the approximated $i$-th stage at time $t_n+c_i$."""

# ╔═╡ 0531870d-0771-41f9-b7ff-658eb306f8c3


# ╔═╡ 0fd169c6-b145-463d-b955-febaf3d4b127
md"""
### Sylvester matrix equation
Multiplying the last eqatuion on the left by $A^{-1}$ and rearranging terms we can formulate the Sylvester equation

$$\begin{equation}
A^{-1} U - h U B^T = A^{-1} U_n.
\end{equation}$$

### Low rank Krylov solver
We solve the Sylvester equation with the low rank Krlyov solver in [1].
"""

# ╔═╡ 06851832-6079-4e0e-8ad9-059a13ee9586


# ╔═╡ 1cbc3308-e2eb-49a6-aa84-8902e83386ad
md"""
### Time stepping scheme
"""

# ╔═╡ 41231afb-ded4-4d37-ac04-454de8279ed0


# ╔═╡ e5d0e917-ec19-4ade-9e65-b1fa0f0f5d95
md"""### P-refinement""" 

# ╔═╡ Cell order:
# ╟─660227ee-c714-11ef-0f53-f53572195380
# ╟─e6b7ba9d-0b79-4199-ad80-533312b6d071
# ╠═55ed59ec-dc08-43bf-8421-ce3e6d474cba
# ╠═d2369cfa-3f9d-4ae6-b9c1-35f82abc0e08
# ╠═ecc3f033-9e4e-4929-ab88-401d6258eaf0
# ╟─d2b6bd39-df19-484b-8f6c-4eed1079522f
# ╠═1e1eac77-df99-4cf6-b224-b142e13f0d18
# ╟─8fc7e19f-43ff-42a7-8eab-5e2509a52701
# ╠═fdd8df3f-b5f3-4286-83f7-0bc3648299cb
# ╟─3d08db47-ff99-4552-a1e6-ba53d2dd5a58
# ╠═610d9408-6af7-4c66-8b93-8a8f0cc07e8b
# ╟─db6b4db6-6502-4bad-9cf5-e697ff406252
# ╠═0531870d-0771-41f9-b7ff-658eb306f8c3
# ╟─0fd169c6-b145-463d-b955-febaf3d4b127
# ╠═06851832-6079-4e0e-8ad9-059a13ee9586
# ╟─1cbc3308-e2eb-49a6-aa84-8902e83386ad
# ╠═41231afb-ded4-4d37-ac04-454de8279ed0
# ╟─e5d0e917-ec19-4ade-9e65-b1fa0f0f5d95
