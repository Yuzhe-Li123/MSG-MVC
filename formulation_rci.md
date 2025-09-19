# RCI formulation

## Condition

$$
\begin{aligned}
&D_{KL}\left(P(\mathbf{Z}^v|\mathbf{Z}) ||R(\mathbf{Z}^v|\mathbf{Z})  \right) \ge 0 
\end{aligned}
$$


## Formulation
$$
\begin{aligned}
H(\mathbf{Z}^v|\mathbf{Z}) & = - \int_{\mathbf{Z}}\int_{\mathbf{Z}^v} p(\mathbf{z}^v,\mathbf{z})\log p(\mathbf{z}^v|\mathbf{z}) d\mathbf{z}^vd\mathbf{z} \\
& = -\mathbb{E}_{p(\mathbf{z}^v, \mathbf{z})} [\log {p}({\mathbf{z}^v}|\mathbf{z})]\\
&= -\mathbb{E}_{p(\mathbf{z}^v, \mathbf{z})} [\log R({\mathbf{z}^v}|\mathbf{z})] + D_{KL}({\mathcal{P}}({\mathbf{Z}^v}|\mathbf{Z})||R({\mathbf{Z}^v}|\mathbf{Z})) \\ 
& \le -\mathbb{E}_{P(\mathbf{z}^v, \mathbf{z})} [\log R({\mathbf{z}^v}|\mathbf{z})] \\
&= -\mathbb{E}_{P(\mathbf{z}^v, \mathbf{z})} \left[ \log \left( \frac{1}{(2\pi\sigma^2)^{d_z^v/2}} e^{-\frac{\left\|{\mathbf{z}^v} - G^v(\mathbf{z})\right\|^2}{2\sigma^2}} \right) \right]
\end{aligned}
$$

assume that  \( R({\mathbf{Z}^v}|\mathbf{Z}) \) follows a Gaussian distribution  \( \mathcal{N}({\mathbf{Z}^v} |  G^v(\mathbf{Z}), \sigma^2 \mathbf{I}) \)