import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_


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
        block_size,
        dim,
        A,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.block_size = block_size
        self.dim = dim
        self.A = A
        self.inv_alpha = nn.ParameterList(
            [nn.Parameter(torch.tensor([0.5]), requires_grad=True)]
        )
        self.labd = nn.ParameterList(
            [nn.Parameter(torch.tensor([1e-3]), requires_grad=True)]
        )
        self.conv1 = nn.Conv2d(dim, 1, kernel_size=1)
        self.conv2 = nn.Conv2d(1, dim, kernel_size=1)
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

    def forward(self, x: Tensor, y: Tensor):
        xt = self.conv1(x)
        Axt = F.conv2d(xt, self.A, stride=self.block_size, padding=0, bias=None)
        Axty = Axt - y
        AtAxty = F.conv_transpose2d(Axty, self.A, stride=self.block_size)
        r = xt - self.inv_alpha[0] * AtAxty
        r = self.conv2(r)

        xt = self.for_c(r)
        xt = torch.mul(
            torch.sign(xt), F.relu(torch.abs(xt) - self.labd[0] * self.inv_alpha[0])
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
        x_avg = torch.concat([x_avg0, x_avg1, x_avg2], dim=1)
        gate_scores = self.gate(x_avg)
        return x, gate_scores, (x1, x2)


class SwitchMoE(nn.Module):
    def __init__(
        self,
        dim: int,
        num_experts: int,
        A: int,
        block_size: int,
        topk: int = 1,
        use_aux_loss: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.block_size = block_size
        self.topk = topk
        self.use_aux_loss = use_aux_loss

        self.experts = nn.ModuleList(
            [
                DUSE(
                    block_size,
                    dim,
                    A,
                    *args,
                    **kwargs,
                )  # -> experts DUNs
                for _ in range(self.num_experts)
            ]
        )

    def forward(self, x: Tensor, y: Tensor, gate_scores):
        top_k_scores, top_k_indices = gate_scores.topk(self.topk, dim=-1)
        top_k_indices = top_k_indices.squeeze(dim=1)
        top_k_scores = top_k_scores.squeeze(dim=1)

        expert_outputs = torch.zeros_like(x)
        for i in range(self.topk):
            expert_outputs = expert_outputs + top_k_scores[i] * self.experts[
                top_k_indices[i]
            ](x, y)

        if self.use_aux_loss and self.training:
            load = gate_scores.sum(0)  # Sum over all experts
            loss = 1e-3 * ((load.std() / (load.mean() + 1e-6)) ** 2)

            return expert_outputs, loss

        return expert_outputs, None


class DAM(nn.Module):
    def __init__(self, dim, head, dim_head, block_size=32):
        super(DAM, self).__init__()
        self.block_size = block_size
        self.c1 = nn.Sequential(
            BasicBlock(dim),
            nn.Conv2d(dim, 1, kernel_size=1),
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(2, dim, kernel_size=1),
            BasicBlock(dim),
        )
        self.c3 = nn.Sequential(
            nn.Conv2d(dim, head * dim_head, kernel_size=1),
        )

    def forward(self, x, y, A):
        xt = self.c1(x)
        r1 = F.conv2d(xt, A, stride=self.block_size, padding=0, bias=None)
        dg1 = xt - F.conv_transpose2d(r1, A, stride=self.block_size)

        r2 = r1 - y
        dg2 = F.conv_transpose2d(r2, A, stride=self.block_size)

        dg = torch.concat([dg1, dg2], dim=1)
        dg = x * torch.sigmoid(self.c2(dg))
        x = x + dg
        x = self.c3(x)
        return x


class DASA(nn.Module):  # degrade aware self-attention
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
        v = v.transpose(-2, -1)  # (b, heads, d, hw)
        q = F.normalize(q, dim=-2, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = k @ q  # attn = K^T*Q  (b, heads, d, d)
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)  # (b, heads, d, d)
        x = attn @ v  # (b, heads, d, hw)

        x = x.permute(0, 3, 1, 2)  # (b, hw, heads, d)
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)  # (b, hw, heads * d)
        out_c = self.proj(x).view(b, h, w, c).permute(0, 3, 1, 2)  # (b, c, h, w)
        out_p = self.pos_emb(res)
        out = out_c + out_p

        return out, attn


class DUMoEStage(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        A: Tensor,
        block_size: int,
        num_experts: int = 3,
        topk: int = 1,
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
            A,
            block_size,
            topk=topk,
            use_aux_loss=use_aux_loss,
            *args,
            **kwargs,
        )
        self.ln1 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.ln2 = LayerNorm(dim, eps=1e-6, data_format="channels_first")

    def forward(self, x: Tensor, y: Tensor, A: Tensor, x1: Tensor, x2: Tensor):
        #### Atten ####
        resi = x
        x, attn = self.attn(x, y, A)
        x = x + resi
        x = self.ln1(x)

        ##### Gate #####
        x, gate_score, (x1, x2) = self.gate(x, x1, x2)

        ##### MoE #####
        resi = x
        x, loss = self.ffn(x, y, gate_score)
        x = x + resi
        x = self.ln2(x)
        return x, loss, (x1, x2)


class DUMoE(nn.Module):
    def __init__(
        self,
        ratio: int = 10,
        dim: int = 32,
        mult: float = 1.5,
        block_size: int = 32,
        heads: int = 8,
        dim_head: int = 64,
        num_experts: int = 3,
        depth: int = 5,
        topk: int = 1,
        use_aux_loss: bool = True,
        share_weights: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.mult = mult
        self.num_experts = num_experts
        self.depth = depth
        self.ratio = ratio
        self.block_size = block_size
        self.topk = topk

        A = torch.from_numpy(self.load_sampling_matrix()).float()
        self.A = nn.Parameter(
            Rearrange("m (1 b1 b2) -> m 1 b1 b2", b1=self.block_size)(A),
            requires_grad=True,
        )

        self.dims = [dim, int(dim * mult), int(dim * (mult + 1))]
        self.embedding = nn.Sequential(
            nn.Conv2d(1, self.dims[0], kernel_size=3, padding=1),
            BasicBlock(self.dims[0]),
        )
        self.c1 = nn.Conv2d(self.dims[0], self.dims[1], 1)
        self.c2 = nn.Conv2d(self.dims[0] * 2, self.dims[1] * 2, 1)
        self.c3 = nn.Conv2d(self.dims[0] * 4, self.dims[1] * 4, 1)

        self.share_weights = share_weights
        stage_num = 3
        if not self.share_weights:
            stage_num = self.depth

        self.layers = nn.ModuleList([])
        self.layers.append(
            DUMoEStage(
                self.dims[0],
                heads,
                dim_head,
                self.A,
                block_size,
                num_experts,
                topk,
                use_aux_loss,
                *args,
                **kwargs,
            )
        )

        for _ in range(stage_num - 2):
            self.layers.append(
                DUMoEStage(
                    self.dims[1],
                    heads,
                    dim_head,
                    self.A,
                    block_size,
                    num_experts,
                    topk,
                    use_aux_loss,
                    *args,
                    **kwargs,
                )
            )

        self.layers.append(
            DUMoEStage(
                self.dims[2],
                heads,
                dim_head,
                self.A,
                block_size,
                num_experts,
                topk,
                use_aux_loss,
                *args,
                **kwargs,
            )
        )
        self.to_out = nn.Sequential(
            BasicBlock(self.dims[-1]),
            nn.Conv2d(self.dims[-1], 1, kernel_size=3, padding=1),
        )
        self.apply(self._init_weights)

    def forward(self, x: Tensor) -> Tensor:
        # Sampling
        y = F.conv2d(x, self.A, stride=self.block_size, padding=0, bias=None)

        # Init
        x_init = F.conv_transpose2d(y, self.A, stride=self.block_size)
        xk = self.embedding(x_init)
        x1, x2 = None, None
        total_loss = 0

        # head
        xk, loss, (x1, x2) = self.layers[0](xk, y, self.A, x1, x2)
        if loss != None:
            total_loss = total_loss + loss
        xk_t, x1_t, x2_t = xk, x1, x2

        # body
        xk = self.c1(xk)
        x1 = self.c2(x1)
        x2 = self.c3(x2)
        for i in range(self.depth - 2):
            if self.share_weights:
                xk, loss, (x1, x2) = self.layers[1](xk, y, self.A, x1, x2)
            else:
                xk, loss, (x1, x2) = self.layers[i + 1](xk, y, self.A, x1, x2)
            if loss != None:
                total_loss = total_loss + loss

        xk = torch.concat([xk, xk_t], dim=1)
        x1 = torch.concat([x1, x1_t], dim=1)
        x2 = torch.concat([x2, x2_t], dim=1)

        # tail
        xk, loss, (x1, x2) = self.layers[-1](xk, y, self.A, x1, x2)
        if loss != None:
            total_loss = total_loss + loss

        # output
        xk = self.to_out(xk)

        if self.training:
            return xk, total_loss / self.depth

        return xk

    def load_sampling_matrix(self):
        path = "./data/sampling_matrix"
        data = np.load(f"{path}/{self.ratio}_{self.block_size}.npy")
        return data

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
