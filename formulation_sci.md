# SIC formulation

## Condition
$$
\begin{aligned}
&D_{KL}\left(P(\mathbf{X}^v|\mathbf{Z}^v) ||D(\mathbf{X}^v|\mathbf{Z}^v)  \right) \\
& = \int_{\mathbf{X}^v}p(\mathbf{x}^v | \mathbf{z}^v) \log \frac{p(\mathbf{x}^v | \mathbf{z}^v)}{q(\mathbf{x}^v | \mathbf{z}^v)} d\mathbf{x}^v \\
& = \int_{\mathbf{X}^v}p(\mathbf{x}^v | \mathbf{z}^v) \log p(\mathbf{x}^v | \mathbf{z}^v) d\mathbf{x}^v - \int_{\mathbf{X}^v}p(\mathbf{x}^v | \mathbf{z}^v) \log q(\mathbf{x}^v | \mathbf{z}^v) d\mathbf{x}^v 
\end{aligned}
$$

$$
\begin{aligned}
&D_{KL}\left(P(\mathbf{X}^v|\mathbf{Z}^v) ||D(\mathbf{X}^v|\mathbf{Z}^v)  \right) \ge 0 
\end{aligned}
$$

Therefore

$$
\begin{aligned}
\int_{\mathbf{X}^v}p(\mathbf{x}^v | \mathbf{z}^v) \log p(\mathbf{x}^v | \mathbf{z}^v) d\mathbf{x}^v \ge \int_{\mathbf{X}^v}p(\mathbf{x}^v | \mathbf{z}^v) \log q(\mathbf{x}^v | \mathbf{z}^v) d\mathbf{x}^v 
\end{aligned}
$$

## Formulation
$$
	\begin{aligned}
		&I(\mathbf{X}^v;\mathbf{Z}^v) = \int_{\mathbf{Z}^v}\! \int_{\mathbf{X}^v} \!\! {p\left(\mathbf{x}^v, \mathbf{y}^v\right)\log\frac{p(\mathbf{x}^v, \mathbf{z}^v)}{p(\mathbf{x}^v)p(\mathbf{z}^v)}d\mathbf{x}^vd\mathbf{z}^v} \\
		&= H(\mathbf{X}^v) \! +\! \int_{\mathbf{Z}^v}\!\!{p(\mathbf{z}^v)\!\int_{\mathbf{X}^v}\!\!{p(\mathbf{x}^v|\mathbf{z}^v)\log p(\mathbf{x}^v|\mathbf{z}^v)d\mathbf{x}^vd\mathbf{z}^v}} \\
		&\ge \int_{\mathbf{Z}^v}\!\!{p(\mathbf{\mathbf{z}}^v)}\int_{\mathbf{X}^v}\!\!{p(\mathbf{x}^v|\mathbf{z}^v)} \log {q(\mathbf{x}^v|\mathbf{z}^v)}d\mathbf{x}^vd\mathbf{z}^v\\
		& = \int_{\mathbf{X}^v}\!\!{p(\mathbf{x}^v)}\int_{\mathbf{Z}^v}\!\!{p(\mathbf{z}^v|\mathbf{x}^v)} \log {q(\mathbf{x}^v|\mathbf{z}^v)}d\mathbf{z}^vd\mathbf{x}^v  \\
		& = \mathbb{E}_{P(\mathbf{x}^v,\mathbf{z}^v)}\log {D(\mathbf{x}^v|\mathbf{z}^v)}
	\end{aligned}
$$

assume $D(\mathbf{X}^v|\mathbf{Z}^v) = \mathcal{N}\left(\mu (\mathbf{X}^v|\mathbf{Z}^v), \sigma^2I\right)$, we have:
$$
\begin{aligned}
& I(\mathbf{X}^v;\mathbf{Z}^v) \\
&= -\mathbb{E}_{P(\mathbf{x}^v, \mathbf{z}^v)} \left[ \log \left( \frac{1}{(2\pi\sigma^2)^{d_x	^v/2}} e^{-\frac{\left\|{\mathbf{x}^v} - \mu^v(\mathbf{z})\right\|^2}{2\sigma^2}} \right) \right]
\end{aligned}
$$