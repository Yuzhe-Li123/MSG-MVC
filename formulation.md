## Condition
$$
\mathbf{p}_i, \mathbf{q}_j^v \text{ are column vectors}
$$
$$
\begin{aligned}
I(\mathbf{P^\top}, (\mathbf{Q}^v)^\top) &= \sum_{i = 1}^{K} \sum_{j = 1}^{K}{p(\mathbf{p}_i,\mathbf{q}^v_j)} \log \frac{p(\mathbf{p}_i,\mathbf{q}^v_j)}{p(\mathbf{p}_i)p(\mathbf{q}^v_j)} \\
\end{aligned}
$$

$$
\begin{aligned}
	p(\mathbf{p}_i, \mathbf{q}_i^v) > p(\mathbf{p}_i) p(\mathbf{q}_i^v), \\ 	p(\mathbf{p}_i, \mathbf{q}_j^v) = p(\mathbf{p}_i) p(\mathbf{q}_j^v), \quad i\neq j
\end{aligned}
$$

$$
\begin{aligned}
	p(\mathbf{q}_i^v | \mathbf{p}_i) > \delta
\end{aligned}
$$

$$
\begin{aligned}
	p(\mathbf{p}_i) \approx \frac{1}{K},\quad e^{sim(\mathbf{p}_i,\mathbf{q}_i^v)/\tau}  \propto m_{ij}
\end{aligned}
$$

$$
\begin{aligned}
  m_{ij} = \frac{p(\mathbf{p}_i,\mathbf{q}^v_j)}{p(\mathbf{p}_i)p(\mathbf{q}^v_j)}
\end{aligned}
$$

## Formulation
$$
\begin{aligned}
\frac{1}{V}\sum_{v = 1}^{V}&I(\mathbf{P}, \mathbf{Q}^v) = \frac{1}{V}
\sum_{v = 1}^{V} \sum_{i = 1}^{K} \sum_{j = 1}^{K}{p(\mathbf{p}_i,\mathbf{q}^v_j)} \log \frac{p(\mathbf{p}_i,\mathbf{q}^v_j)}{p(\mathbf{p}_i)p(\mathbf{q}^v_j)} \\ 
& = \frac{1}{V}\sum_{v=1}^{V}\sum_{i = 1}^{K}p(\mathbf{p}_i,\mathbf{q}^v_i)\log\frac{p(\mathbf{p}_i,\mathbf{q}^v_i)}{p(\mathbf{p}_i)p(\mathbf{q}^v_i)} + \frac{1}{V}\sum_{v = 1}^{V}\sum_{i = 1}^{K}\sum_{j \neq i}^{K} 0\\
&\approx \frac{1}{KV}\sum_{v = 1}^{V}\sum_{i = 1}^{K}p(\mathbf{q}^v_i|\mathbf{p}_i)\log\frac{p(\mathbf{p}_i,\mathbf{q}^v_i)}{p(\mathbf{p}_i)p(\mathbf{q}^v_i)} \\
& \ge \frac{\delta}{KV}\sum_{v=1}^{V}\sum_{i = 1}^K\log \frac{m_{ii}}{\sum_{j = 1}^{K}m_{ij}} + \frac{\delta}{K}\sum_{v=1}^{V}\sum_{i = 1}^{K}\log \sum_{j = 1}^{K}m_{ij} \\
& = -{\delta} \mathcal{L} + \frac{\delta}{K}\sum_{v = 1}^{V} \sum_{i = 1}^{K} \log \left( K - 1 + m_{ii}\right) \\
& \ge -\delta \mathcal{L} + \delta  \log K
\end{aligned}
$$

$$
\begin{aligned}
	\mathcal{L} = -\frac{1}{KV}\sum_{v = 1}^{V}\sum_{i = 1}^{K}\log \frac{e^{sim(\mathbf{p}_i, \mathbf{q}^v_i)/\tau}}{\sum_{j = 1}^{K}e^{sim(\mathbf{p}_i, \mathbf{q}^v_j)/\tau}}
\end{aligned}
$$