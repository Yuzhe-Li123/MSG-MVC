import torch
import torch.nn as nn


class ChebyKANLinear(nn.Module):
    def __init__(self, input_dim, output_dim, degree = 3, drop_type=None, drop_prob=0.0):
        super(ChebyKANLinear, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree
        
        # DropKAN相关参数
        self.drop_type = drop_type
        self.drop_prob = drop_prob
        if drop_type is not None and drop_type not in ['dropkanpa', 'dropkanps']:
            raise ValueError("drop_type must be None, 'dropkanpa', or 'dropkanps'")

        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        self.epsilon = 1e-7
        self.pre_mul = False
        self.post_mul = False
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))
        self.register_buffer("arange", torch.arange(0, degree + 1, 1))

    def forward(self, x):
        # 切比雪夫多项式定义在[-1, 1]，使用tanh归一化输入
        b, c_in = x.shape
        
        if self.pre_mul:
            mul_1 = x[:,::2]
            mul_2 = x[:,1::2]
            mul_res = mul_1 * mul_2
            x = torch.concat([x[:,:x.shape[1]//2], mul_res])
        
        # 扩展维度以适应多项式度数
        x = x.view((b, c_in, 1)).expand(-1, -1, self.degree + 1)
        x = torch.tanh(x)  # 归一化到[-1,1]
        x = torch.acos(x.clamp(-1 + self.epsilon, 1 - self.epsilon))
        x = x * self.arange
        x = x.cos()  # 得到切比雪夫多项式值

        # 应用DropKANps（post-spline掩码）
        if self.training and self.drop_type == 'dropkanps':
            mask = torch.bernoulli(torch.full((x.shape[0], x.shape[1], 1), 1 - self.drop_prob, device=x.device))
            x = x * mask  # 样条输出后掩码，无需缩放（样条均值为0）

        # 计算切比雪夫插值（拆分 einsum 以便应用DropKANpa）
        x_coeff = torch.einsum("bid,iod->bio", x, self.cheby_coeffs)  # 先聚合度数维度

        # 应用DropKANpa（post-activation掩码）
        if self.training and self.drop_type == 'dropkanpa':
            mask = torch.bernoulli(torch.full((x_coeff.shape[0], x_coeff.shape[1], 1), 1 - self.drop_prob, device=x.device))
            x_coeff = x_coeff * mask
            x_coeff = x_coeff / (1 - self.drop_prob + self.epsilon)  # 缩放以保持期望

        # 从求和改为均值（根据第一篇论文的改进）
        y = x_coeff.sum(dim=1) / self.inputdim  # 除以输入维度得到均值
        y = y.view(-1, self.outdim)

        if self.post_mul:
            mul_1 = y[:,::2]
            mul_2 = y[:,1::2]
            mul_res = mul_1 * mul_2
            y = torch.concat([y[:,:y.shape[1]//2], mul_res])
        
        return y