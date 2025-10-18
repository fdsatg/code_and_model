import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_


def A(x, Phi):
    B, nC, H, W = x.shape
    temp = x * Phi
    y = torch.sum(temp, 1)
    y = y / nC * 2
    return y


def At(y, Phi):
    temp = torch.unsqueeze(y, 1).repeat(1, Phi.shape[1], 1, 1)
    x = temp * Phi
    return x


def shift_3d(inputs, step=2):
    [bs, nC, row, col] = inputs.shape
    for i in range(nC):
        inputs[:, i, :, :] = torch.roll(inputs[:, i, :, :], shifts=step * i, dims=2)
    return inputs


def shift_back_3d(inputs, step=2):
    [bs, nC, row, col] = inputs.shape
    for i in range(nC):
        inputs[:, i, :, :] = torch.roll(
            inputs[:, i, :, :], shifts=(-1) * step * i, dims=2
        )
    return inputs


class CA(nn.Module):
    def __init__(self, channel, reduction):
        super(CA, self).__init__()
        self.conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.conv(x)
        return x * y


class BasicBlock(nn.Module):
    def __init__(self, dim, reduction=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            LayerNorm(dim, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dim, 4 * dim, kernel_size=1, padding=0),
            nn.GELU(),
            nn.Conv2d(4 * dim, dim, kernel_size=1, padding=0),
            CA(dim, reduction),
        )

    def forward(self, x):
        x = self.block(x) + x
        return x


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class DUSE(nn.Module):
    def __init__(
        self,
        dim,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.basic_dim = 28
        self.inv_alpha = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.labd = nn.Parameter(
            torch.zeros((1, self.dim, 1, 1)) + 1e-3, requires_grad=True
        )
        self.conv1 = nn.Conv2d(dim, self.basic_dim, kernel_size=1)
        self.conv2 = nn.Conv2d(self.basic_dim, dim, kernel_size=1)
        self.for_c = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
        )
        self.back_c = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
        )
        self.unit = nn.Sequential(
            BasicBlock(dim),
            nn.Sigmoid(),
        )
        self.funtion = nn.Sequential(
            BasicBlock(self.basic_dim + 1),
            nn.Conv2d(self.basic_dim + 1, self.basic_dim, kernel_size=1),
        )

    def forward(self, x: Tensor, y: Tensor, Phi):
        xk = self.conv1(x)
        xk = shift_3d(xk)
        Phi = Phi + self.funtion(torch.cat([y.unsqueeze(dim=1), Phi], dim=1))
        Axy = A(xk, Phi) - y
        r = xk - self.inv_alpha * At(Axy, Phi)
        r = shift_back_3d(r)
        r = self.conv2(r)

        xt = self.for_c(r)
        xt = torch.mul(
            torch.sign(xt), F.relu(torch.abs(xt) - self.labd * self.inv_alpha)
        )
        xt = self.back_c(xt) + r

        xt = self.unit(xt) * x

        x = xt + x
        return x


class USampling(nn.Module):
    def __init__(self, scale_factor, dim, out_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=dim,
                out_channels=out_dim,
                kernel_size=2,
                stride=scale_factor,
            ),
            LayerNorm(out_dim, eps=1e-6, data_format="channels_first"),
        )

    def forward(self, x):
        return self.block(x)


class DSampling(nn.Module):
    def __init__(self, scale_factor, dim, out_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, out_dim, kernel_size=2, stride=scale_factor),
            LayerNorm(out_dim, eps=1e-6, data_format="channels_first"),
        )

    def forward(self, x):
        return self.block(x)


class UBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.down1 = nn.Sequential(
            BasicBlock(dim),
            DSampling(2, dim, 2 * dim),
        )
        self.down2 = nn.Sequential(
            BasicBlock(2 * dim),
            DSampling(2, 2 * dim, 4 * dim),
        )

        self.mid = nn.Sequential(
            BasicBlock(4 * dim),
        )

        self.up1 = nn.Sequential(
            USampling(2, 4 * dim, 2 * dim),
            BasicBlock(2 * dim),
        )
        self.up2 = nn.Sequential(
            USampling(2, 2 * dim, dim),
            BasicBlock(dim),
        )

    def forward(self, x, x1=None, x2=None):
        b, c, h_inp, w_inp = x.shape
        hb, wb = 4, 4
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode="reflect")

        if x1 != None:
            xk1 = self.down1(x) + x1
        else:
            xk1 = self.down1(x)
        if x1 != None:
            xk2 = self.down2(xk1) + x2
        else:
            xk2 = self.down2(xk1)

        xk3 = self.mid(xk2)

        xk4 = self.up1(xk3) + xk1
        xk5 = self.up2(xk4) + x

        return xk5[:, :, :h_inp, :w_inp], (xk4, xk3)


class MSGate(nn.Module):
    def __init__(self, dim, num_experts):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.ublock = UBlock(dim)
        self.avg1 = nn.AdaptiveAvgPool2d(1)
        self.avg2 = nn.AdaptiveAvgPool2d(1)
        self.avg3 = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.dim * (1 + 2 + 4), self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.num_experts),
            nn.Softmax(dim=-1),
        )

    def forward(self, x, x1, x2):
        x, (x1, x2) = self.ublock(x, x1, x2)
        x_avg0 = self.avg1(x)
        x_avg1 = self.avg2(x1)
        x_avg2 = self.avg3(x2)
        x_avg = torch.cat([x_avg0, x_avg1, x_avg2], dim=1)
        gate_scores = self.gate(x_avg)
        return x, gate_scores, (x1, x2)


class SwitchMoE(nn.Module):
    def __init__(
        self,
        dim: int,
        num_experts: int,
        topk: int = 1,
        use_aux_loss: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.topk = topk
        self.use_aux_loss = use_aux_loss

        self.experts = nn.ModuleList(
            [
                DUSE(
                    dim,
                    *args,
                    **kwargs,
                )  # -> experts DUNs
                for _ in range(self.num_experts)
            ]
        )

    def forward(self, x: Tensor, y: Tensor, A, gate_scores):
        top_k_scores, top_k_indices = gate_scores.topk(self.topk, dim=-1)
        top_k_indices = top_k_indices.squeeze(dim=1)
        top_k_scores = top_k_scores.squeeze(dim=1)

        expert_outputs = torch.zeros_like(x)
        for i in range(self.topk):
            expert_outputs = expert_outputs + top_k_scores[i] * self.experts[
                top_k_indices[i]
            ](x, y, A)

        if self.use_aux_loss and self.training:
            load = gate_scores.sum(0)  # Sum over all experts
            loss = 1e-3 * ((load.std() / (load.mean() + 1e-6)) ** 2)

            return expert_outputs, loss

        return expert_outputs, None


class DAM(nn.Module):
    def __init__(self, dim, head, dim_head):
        super(DAM, self).__init__()
        self.basic_dim = 28
        self.c1 = nn.Sequential(
            BasicBlock(dim),
            nn.Conv2d(dim, self.basic_dim, kernel_size=1),
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(2 * self.basic_dim, dim, kernel_size=1),
            BasicBlock(dim),
        )
        self.c3 = nn.Sequential(
            nn.Conv2d(dim, head * dim_head, kernel_size=1),
        )
        self.fution = nn.Sequential(
            BasicBlock(self.basic_dim + 1),
            nn.Conv2d(self.basic_dim + 1, self.basic_dim, kernel_size=1, padding=0),
        )

    def forward(self, x, y, Phi):
        xt = self.c1(x)
        xt = shift_3d(xt)
        Phi = Phi + self.fution(torch.cat([y.unsqueeze(dim=1), Phi], dim=1))
        r1 = A(xt, Phi)
        dg1 = xt - At(r1, Phi)
        dg1 = shift_back_3d(dg1)

        r2 = r1 - y
        dg2 = At(r2, Phi)
        dg2 = shift_back_3d(dg2)

        dg = torch.cat([dg1, dg2], dim=1)
        dg = x * torch.sigmoid(self.c2(dg))
        x = x + dg
        x = self.c3(x)
        return x


class DASA(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False, groups=dim),
        )
        self.mm = DAM(dim, heads, dim_head)
        self.dim = dim

    def forward(self, x: Tensor, y: Tensor, A: Tensor):
        res = x
        b, c, h, w = x.shape
        x = Rearrange("b c h w -> b (h w) c")(x)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)

        # degrade aware
        mask_attn = self.mm(res, y, A).permute(0, 2, 3, 1)

        q, k, v, mask_attn = map(
            lambda t: Rearrange("b n (h d) -> b h n d", h=self.num_heads)(t),
            (q_inp, k_inp, v_inp, mask_attn.flatten(1, 2)),
        )  # (b, heads, hw, d)
        v = v * mask_attn

        k = k.transpose(-2, -1)  # (b, heads, d, hw)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-2, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = k @ q  # attn = K^T*Q  (b, heads, d, d)
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v  # (b, heads, d, hw)

        x = x.permute(0, 3, 1, 2)  # (b, hw, heads, d)
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)  # (b, hw, heads * d)
        out_c = self.proj(x).view(b, h, w, c).permute(0, 3, 1, 2)  # (b, c, h, w)
        out_p = self.pos_emb(res)
        out = out_c + out_p

        return out


class DUMoEStage(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        num_experts: int = 3,
        topk: int = 1,
        dun_depth: int = 1,
        use_aux_loss: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head

        self.attn = DASA(dim, heads, dim_head)
        self.gate = MSGate(dim, num_experts)
        self.ffn = SwitchMoE(
            dim,
            num_experts,
            dun_depth,
            topk,
            use_aux_loss,
            *args,
            **kwargs,
        )
        self.ln1 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.ln2 = LayerNorm(dim, eps=1e-6, data_format="channels_first")

    def forward(self, x: Tensor, y: Tensor, A: Tensor, x1: Tensor, x2: Tensor):
        #### Atten ####
        resi = x
        x = self.attn(x, y, A)
        x = x + resi
        x = self.ln1(x)

        ##### MSGate #####
        x, gate_score, (x1, x2) = self.gate(x, x1, x2)

        ##### MoE #####
        resi = x
        x, loss = self.ffn(x, y, A, gate_score)
        x = x + resi
        x = self.ln2(x)
        return x, loss, (x1, x2)


class DUMoE(nn.Module):
    def __init__(
        self,
        heads: int = 8,
        dim: int = 32,
        dim_head: int = 64,
        num_experts: int = 3,
        depth: int = 5,
        dun_depth: int = 1,
        mult: float = 1.5,
        topk: int = 1,
        use_aux_loss: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.num_experts = num_experts
        self.depth = depth
        self.topk = topk
        self.basic_dim = 28
        self.mult = mult

        self.dims = [dim, int(dim * mult), int(dim * (mult + 1))]
        self.init_fution = nn.Conv2d(
            2 * self.basic_dim, self.basic_dim, 1, padding=0, bias=True
        )
        self.embedding = nn.Sequential(
            nn.Conv2d(self.basic_dim, self.dims[0], kernel_size=3, padding=1),
            BasicBlock(self.dims[0]),
        )
        self.c1 = nn.Conv2d(self.dims[0], self.dims[1], 1)
        self.c2 = nn.Conv2d(self.dims[0] * 2, self.dims[1] * 2, 1)
        self.c3 = nn.Conv2d(self.dims[0] * 4, self.dims[1] * 4, 1)

        self.layers = nn.ModuleList([])
        for i in range(3):
            self.layers.append(
                DUMoEStage(
                    self.dims[i],
                    heads,
                    dim_head,
                    num_experts,
                    topk,
                    dun_depth,
                    use_aux_loss,
                    *args,
                    **kwargs,
                )
            )

        self.to_out = nn.Sequential(
            BasicBlock(self.dims[-1]),
            nn.Conv2d(self.dims[-1], self.basic_dim, kernel_size=3, padding=1),
        )
        self.apply(self._init_weights)

    def forward(self, y, input_mask) -> Tensor:
        # Init
        Phi = input_mask
        x_init = self.initial(y, Phi)
        x_init = shift_back_3d(x_init)

        xk = self.embedding(x_init)
        x1, x2 = None, None
        total_loss = 0

        # head
        xk, loss, (x1, x2) = self.layers[0](xk, y, Phi, x1, x2)
        if loss != None:
            total_loss = total_loss + loss
        xk_head, x1_head, x2_head = xk, x1, x2

        # body
        xk = self.c1(xk)
        x1 = self.c2(x1)
        x2 = self.c3(x2)
        for _ in range(self.depth - 2):
            xk, loss, (x1, x2) = self.layers[1](xk, y, Phi, x1, x2)
            if loss != None:
                total_loss = total_loss + loss
        xk = torch.cat([xk, xk_head], dim=1)
        x1 = torch.cat([x1, x1_head], dim=1)
        x2 = torch.cat([x2, x2_head], dim=1)

        # tail
        xk, loss, (x1, x2) = self.layers[2](xk, y, Phi, x1, x2)
        if loss != None:
            total_loss = total_loss + loss

        # output
        xk = self.to_out(xk)

        if self.training:
            return xk[:, :, :, : y.shape[1]], total_loss / self.depth

        return xk[:, :, :, : y.shape[1]]

    def initial(self, y, Phi):
        """
        :param y: [b,256,310]
        :param Phi: [b,28,256,310]
        :return: [b,28,256,310]
        """
        B, C, H, W = Phi.shape
        y_shift = y.unsqueeze(1).repeat((1, C, 1, 1))
        x_init = self.init_fution(torch.cat([y_shift, Phi], dim=1))

        return x_init

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
