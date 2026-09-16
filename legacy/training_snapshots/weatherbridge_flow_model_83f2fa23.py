"""WeatherBridge — transport-aware VFI backbone for temporal
interpolation of reanalysis fields.

Motivation. The capacity-matched study showed the *backbone* barely moves
bulk RMSE at 6h — the bilinear scaffold carries the low-frequency signal and
all learned backbones cluster. The one thing every WeatherBridge variant so
far omits is **explicit transport**: weather advects, but ``bilinear +
additive residual`` never warps. Proper VFI (EMA-VFI / VFIMamba / ATM-VFI)
estimates motion and warps the anchor frames toward the query time. This
backbone adds that, cheaply, while staying simpler and faster than the
DC-AE reference.

Design (all learned work at low resolution — pixel-unshuffle-style stride
stem, so activations shrink ~4-16x and bs scales back up):

    input  = [x0, xT, xT - x0, static]          # frame-diff hands the net motion
    trunk  = strided circular-conv UNet + tau-AdaLN bottleneck
    heads (at stem resolution, upsampled to full res):
      * flow   F0, FT, A0, AT       -> constant-acceleration anchor warps
      * blend  alpha(tau, motion)   -> per-pixel/channel transport fusion
      * gate   beta                 -> transport vs linear-scaffold routing
      * resid  Delta                -> fine + coarse + optional SFNO correction
      * hydro  H(T)                 -> optional Z-column and MSLP correction
    x_tr  = alpha * warp(x0) + (1 - alpha) * warp(xT)
    x_mix = beta * x_tr + (1 - beta) * x_linear
    x_hat = x_mix + tanh(scale) * Delta + optional_hydro_correction

Gated skips. Decoder skips are fused through a **zero-init tau-gate**
(``GatedSkip``): near the anchors (tau=1,5) the gate stays ~0 so the warped
scaffold is trusted; mid-interval (tau=3) it opens for fine correction. This
is exactly the fix for the old ``WB-Skip`` overshoot (ungated high-res skips
injected uncontrolled HF). Set ``gated_skip=False`` / ``use_skip=False`` for
the ablation arms.

forward(x0, xT, tau, cond=None, static=None) -> (x_hat, aux_dict)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- tau embedding
class SinusoidalPosEmb(nn.Module):
    """Low-order harmonic basis over NORMALISED tau in [0, 1].

    tau reaches the net as tau_hour / delta_t (see the trainer's
    TauRescaleAnd24chWrapper), so it always lives in [0, 1]. Using a small set
    of LOW harmonic frequencies {pi, 2pi, ..., half*pi} keeps the
    tau -> modulation map smooth and low-order, so the model INTERPOLATES to
    unseen tau (train {1,3,5} -> eval {2,4}) instead of memorising the training
    tau and zig-zagging in between. This is the fix for the t2m held-out spike:
    t2m carries the strongest diurnal cycle, so its optimal correction varies
    most with tau -> a rich (128-freq) embedding overfit the 3 training tau and
    blew up at {2,4}. A few smooth harmonics still capture the diurnal
    curvature over the 6 h window while generalising across tau."""

    def __init__(self, dim: int, base_period: float = 16.0):
        super().__init__()
        self.dim = dim
        self.base_period = base_period  # kept for signature compat (unused)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        k = torch.arange(1, half + 1, device=t.device, dtype=torch.float32)
        args = t.view(-1, 1).float() * (math.pi * k).view(1, -1)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class TimeMLP(nn.Module):
    def __init__(self, dim: int = 256, freq_dim: int = 8, base_period: float = 16.0):
        super().__init__()
        self.emb = SinusoidalPosEmb(freq_dim, base_period)
        self.net = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(),
                                  nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(self.emb(t))


class AdaLNZero(nn.Module):
    """Zero-init AdaLN modulation on (B, C, H, W) conditioned on tau embedding."""

    def __init__(self, dim: int, time_dim: int = 256):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, dim), dim)
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, dim * 3))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, h: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        g, b, gate = self.mlp(temb).chunk(3, dim=-1)
        g = g[:, :, None, None]; b = b[:, :, None, None]; gate = gate[:, :, None, None]
        return h + gate * ((1 + g) * self.norm(h) + b)


# ---------------------------------------------------------------- conv blocks
def _pole_pad_latitude(x: torch.Tensor, pad: int) -> torch.Tensor:
    """Append latitude rows reflected through each pole at the antipode."""
    if pad == 0:
        return x
    if x.size(-1) % 2:
        raise ValueError("spherical padding requires an even longitude width")
    if pad > x.size(-2):
        raise ValueError("spherical padding exceeds the latitude height")
    half_width = x.size(-1) // 2
    top = torch.roll(
        x[..., :pad, :].flip(-2),
        shifts=half_width,
        dims=-1,
    )
    bottom = torch.roll(
        x[..., -pad:, :].flip(-2),
        shifts=half_width,
        dims=-1,
    )
    return torch.cat((top, x, bottom), dim=-2)


def _sphere_pad(x: torch.Tensor, pad: int) -> torch.Tensor:
    """Pad an equirectangular field across the true spherical boundaries."""
    x = _pole_pad_latitude(x, pad)
    return F.pad(x, (pad, pad, 0, 0), mode="circular")


class SphereConv2d(nn.Conv2d):
    """Convolution with periodic longitude and antipodal pole padding."""

    def __init__(self, ci: int, co: int, kernel_size: int = 3, stride: int = 1):
        if kernel_size % 2 != 1:
            raise ValueError("SphereConv2d requires an odd kernel size")
        super().__init__(ci, co, kernel_size, stride=stride, padding=0)
        self.sphere_padding = kernel_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(_sphere_pad(x, self.sphere_padding))


def _conv3x3(ci: int, co: int, stride: int, spherical_ops: bool) -> nn.Module:
    if spherical_ops:
        return SphereConv2d(ci, co, 3, stride=stride)
    return nn.Conv2d(
        ci,
        co,
        3,
        stride,
        1,
        padding_mode="circular",
    )


def conv_block(ci, co, stride=1, spherical_ops: bool = False):
    return nn.Sequential(
        _conv3x3(ci, co, stride, spherical_ops),
        nn.GroupNorm(min(8, co), co), nn.SiLU(),
        _conv3x3(co, co, 1, spherical_ops),
        nn.GroupNorm(min(8, co), co), nn.SiLU(),
    )


class GatedSkip(nn.Module):
    """Fuse a decoder feature with an encoder skip through a zero-init tau-gate.

    out = up + gate(tau) * proj(cat[up, skip]); gate zero-init so training
    starts as if there were no skip (no overshoot), and only opens where the
    interpolation genuinely needs the encoder detail.
    """

    def __init__(
        self,
        dim: int,
        time_dim: int = 256,
        spherical_ops: bool = False,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            _conv3x3(dim * 2, dim, 1, spherical_ops),
            nn.GroupNorm(min(8, dim), dim), nn.SiLU())
        self.gate = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, dim))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, up, skip, temb):
        g = torch.tanh(self.gate(temb))[:, :, None, None]
        return up + g * self.proj(torch.cat([up, skip], dim=1))


def warp(
    x: torch.Tensor,
    flow: torch.Tensor,
    *,
    periodic_longitude: bool = False,
) -> torch.Tensor:
    """Backward-warp ``x`` by pixel-space flow.

    The retained checkpoints use this equirectangular grid-sample operator in
    both axes. The opt-in spherical branch wraps longitude and maps samples
    crossing either pole to the antipodal reflected latitude.
    """
    B, C, H, W = x.shape
    if not periodic_longitude:
        yy, xx = torch.meshgrid(
            torch.arange(H, device=x.device, dtype=x.dtype),
            torch.arange(W, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        gx = (xx + flow[:, 0]) / max(W - 1, 1) * 2 - 1
        gy = (yy + flow[:, 1]) / max(H - 1, 1) * 2 - 1
        grid = torch.stack([gx, gy], dim=-1)
        return F.grid_sample(
            x,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
    yy, xx = torch.meshgrid(
        torch.arange(H, device=x.device, dtype=torch.float32),
        torch.arange(W, device=x.device, dtype=torch.float32),
        indexing="ij",
    )
    flow_float = flow.float()
    # The branch's tanh-bounded velocity and acceleration heads permit at
    # most 12 pixels of displacement. Sixteen antipodal ghost rows therefore
    # cover every production sample while allowing grid_sample to interpolate
    # continuously on both sides of the physical pole at y=-0.5/H-0.5.
    pole_pad = min(16, H)
    sample = _pole_pad_latitude(x, pole_pad)
    sample = torch.cat((sample, sample[..., :1]), dim=-1)
    source_y = yy[None] + flow_float[:, 1] + pole_pad
    source_x = torch.remainder(xx[None] + flow_float[:, 0], W)
    gx = source_x / max(W, 1) * 2 - 1
    gy = source_y / max(sample.size(-2) - 1, 1) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1)
    with torch.autocast(device_type=x.device.type, enabled=False):
        warped = F.grid_sample(
            sample.float(),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
    return warped.to(x.dtype)


class SphericalConv(nn.Module):
    """Spherical Fourier Neural Operator layer (SFNO, Bonev et al. 2023 /
    NVIDIA). Uses the real Spherical Harmonic Transform (torch_harmonics) to
    project onto the LOW spherical-harmonic modes, mixes them with a learned
    complex weight, then transforms back. Unlike a planar FFT this is the
    global basis on the lat/lon sphere and avoids the planar-FFT pole seam for
    smooth planetary mass fields such as geopotential and MSLP."""

    def __init__(self, in_ch: int, out_ch: int, nlat: int, nlon: int,
                 lmax: int = 20, mmax: int = 20, grid: str = "equiangular"):
        super().__init__()
        from torch_harmonics import RealSHT, InverseRealSHT
        self.sht = RealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid=grid)
        self.isht = InverseRealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid=grid)
        s = 1.0 / (in_ch * out_ch)
        self.w = nn.Parameter(s * torch.randn(in_ch, out_ch, lmax, mmax, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = self.sht(x)                                     # (B,C,lmax,mmax) complex
        c = torch.einsum("bilm,iolm->bolm", c, self.w)
        return self.isht(c)


class SpectralBranch(nn.Module):
    """Global low-wavenumber residual branch on the SPHERE (SFNO). Runs in fp32
    (SHT + complex weights are not autocast-safe). Zero-init output so it starts
    neutral. Falls back to identity-zero if torch_harmonics is unavailable."""

    def __init__(self, base: int, out_channels: int, width: int = 20,
                 nlat: int = 360, nlon: int = 720, modes: int = 20):
        super().__init__()
        self.pin = nn.Conv2d(base, width, 1)
        self.spec = SphericalConv(width, width, nlat, nlon, lmax=modes, mmax=modes)
        self.act = nn.GELU()
        self.pout = nn.Conv2d(width, out_channels, 1)
        nn.init.zeros_(self.pout.weight); nn.init.zeros_(self.pout.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        with torch.autocast(h.device.type, enabled=False):
            z = self.pin(h.float())
            z = self.act(self.spec(z) + z)
            return self.pout(z)


class WeatherBridgeModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 24,
        out_channels: int = 24,
        n_static_features: int = 3,
        hidden: int = 96,
        n_levels: int = 3,
        time_emb_dim: int = 256,
        residual_scale_init: float = 0.10,
        use_skip: bool = True,
        gated_skip: bool = True,
        flow_scale: float = 8.0,
        lat_crop: int = 0,
        use_accel: bool = False,
        strong_residual: bool = False,
        mass_aware_gate: bool = False,
        spectral_branch: bool = False,
        hydro_couple: bool = False,
        dual_stream: bool = False,
        spherical_ops: bool = False,
        endpoint_preserving: bool = False,
        query_independent_trajectory: bool = False,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.n_static_features = int(n_static_features)
        self.hidden = int(hidden)
        self.n_levels = int(n_levels)
        self.use_skip = bool(use_skip)
        self.gated_skip = bool(gated_skip)
        self.flow_scale = float(flow_scale)
        self.lat_crop = int(lat_crop)
        # Constant-acceleration transport: warp by F*t + 1/2 A*t^2 instead of F*t.
        # The tau-dependence of the warp is then a POLYNOMIAL (analytically smooth)
        # so it extrapolates cleanly to held-out tau, and the quadratic term
        # captures curved parcel trajectories (rotating fronts, diurnal drift)
        # that a linear flow forces onto the tau-conditioned residual.
        self.use_accel = bool(use_accel)
        # Strong residual: prepend a conv block before the (zero-init) fine
        # residual head so the NON-transport correction has decoder-like
        # capacity. This targets fields where warping does not help — smooth
        # mass (mslp) and the diurnal-local part of t2m — where a thin 1-conv
        # residual trails WeatherDCAE's full DC-AE decoder. Warp still owns
        # the moving fields; this only deepens the additive correction.
        self.strong_residual = bool(strong_residual)
        self.spherical_ops = bool(spherical_ops)
        self.endpoint_preserving = bool(endpoint_preserving)
        self.query_independent_trajectory = bool(
            query_independent_trajectory
        )

        self.time_mlp = TimeMLP(time_emb_dim)
        ch = [hidden * (2 ** i) for i in range(n_levels + 1)]

        # input: [x0, xT, xT-x0, static]
        enc_in = 3 * in_channels + n_static_features
        self.encoder = nn.ModuleList()
        self.encoder.append(
            conv_block(
                enc_in,
                ch[0],
                stride=1,
                spherical_ops=self.spherical_ops,
            )
        )
        for i in range(n_levels):
            self.encoder.append(
                conv_block(
                    ch[i],
                    ch[i + 1],
                    stride=2,
                    spherical_ops=self.spherical_ops,
                )
            )

        self.adaln = AdaLNZero(ch[-1], time_dim=time_emb_dim)

        # decoder
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        self.skip = nn.ModuleList()
        for i in range(n_levels):
            self.up.append(nn.ConvTranspose2d(ch[-(i + 1)], ch[-(i + 2)], 4, 2, 1))
            self.dec.append(
                conv_block(
                    ch[-(i + 2)],
                    ch[-(i + 2)],
                    spherical_ops=self.spherical_ops,
                )
            )
            if self.use_skip:
                self.skip.append(
                    GatedSkip(
                        ch[-(i + 2)],
                        time_emb_dim,
                        spherical_ops=self.spherical_ops,
                    )
                    if gated_skip
                    else _conv3x3(
                        ch[-(i + 2)] * 2,
                        ch[-(i + 2)],
                        1,
                        self.spherical_ops,
                    )
                )

        base = ch[0]
        # heads (operate on the stem-resolution feature)
        n_flow = 8 if self.use_accel else 4   # F0,FT(+A0,AT) each 2ch
        self.flow_head = _conv3x3(
            base,
            n_flow,
            1,
            self.spherical_ops,
        )
        self.blend_head = _conv3x3(
            base,
            out_channels,
            1,
            self.spherical_ops,
        )
        self.res_coarse = _conv3x3(
            ch[1],
            out_channels,
            1,
            self.spherical_ops,
        )
        # optional decoder-capacity block before the zero-init fine head
        self.res_body = (
            conv_block(base, base, spherical_ops=self.spherical_ops)
            if self.strong_residual
            else nn.Identity()
        )
        self.res_fine = _conv3x3(
            base,
            out_channels,
            1,
            self.spherical_ops,
        )
        for m in (self.flow_head, self.blend_head, self.res_coarse, self.res_fine):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
        # global low-wavenumber (FNO) residual for smooth planetary mass fields
        self.spectral = SpectralBranch(base, out_channels) if spectral_branch else None
        # hydrostatic coupling: correct Z1000-700 + mslp from the predicted T
        # column (thickness ~ layer-mean T). 1x1 (local column relation), zero-init.
        self.hydro = nn.Conv2d(4, 5, 1) if hydro_couple else None
        if self.hydro is not None:
            nn.init.zeros_(self.hydro.weight); nn.init.zeros_(self.hydro.bias)
        # ---- dual-stream: a second, SKIP-LESS global decoder from the bottleneck
        # latent (DC-AE-like: no warp, no skips -> global planetary reconstruction)
        # fused with the transport stream by a learned per-channel gate. Winds/
        # moisture route to transport (gate->1), smooth mass to the decoder (->0).
        self.dual_stream = bool(dual_stream)
        if self.dual_stream:
            self.gdec = nn.ModuleList()
            for i in range(n_levels):
                co = ch[-(i + 2)]
                self.gdec.append(nn.Sequential(
                    nn.ConvTranspose2d(ch[-(i + 1)], co, 4, 2, 1),
                    nn.GroupNorm(min(8, co), co), nn.SiLU(),
                    _conv3x3(co, co, 1, self.spherical_ops),
                    nn.GroupNorm(min(8, co), co), nn.SiLU()))
            self.gout = _conv3x3(
                ch[0],
                out_channels,
                1,
                self.spherical_ops,
            )
            nn.init.zeros_(self.gout.weight); nn.init.zeros_(self.gout.bias)
            # per-channel stream gate: sigmoid(0)=0.5 start, learns routing
            self.stream_gate = nn.Parameter(torch.zeros(out_channels))
        self.scale = nn.Parameter(torch.full((out_channels,), float(residual_scale_init)))
        # Per-channel warp gate: sigmoid(warp_gate)=1 -> trust the flow-warped
        # frames (moving fields: wind, moisture); =0 -> fall back to the plain
        # bilinear scaffold (quasi-static fields: t2m/mslp locked to orography
        # where warping only distorts). Init +2 (~0.88) so training starts near
        # the warp-only behaviour and only pulls static channels down.
        self.warp_gate = nn.Parameter(torch.full((out_channels,), 2.0))
        # Mass-aware init: for quasi-static / mass channels (Z1000-700, t2m,
        # mslp in the 24ch order) start warp_gate at -2 (sigmoid ~0.12) so they
        # ride the smooth bilinear scaffold from step 0 instead of the warped
        # frames. Warping a planetary-scale pressure/geopotential field only
        # injects spurious small-scale structure; these fields are the decoder's
        # (residual's) job, not transport's.
        if mass_aware_gate:
            # ONLY the true mass/pressure fields (geopotential, mslp) — these are
            # quasi-static and warping them only distorts. t2m is deliberately
            # EXCLUDED: its weakness is the diurnal residual, not warp distortion,
            # and it carries partial temperature-front advection that the warp can
            # help — forcing beta->0 there would bias against a real signal.
            mass_idx = [16, 17, 18, 19, 23]  # Z1000,Z925,Z850,Z700,mslp
            with torch.no_grad():
                for c in mass_idx:
                    if c < out_channels:
                        self.warp_gate[c] = -2.0

    def _prep_static(self, static, B, device):
        if self.n_static_features <= 0:
            return None
        if static.dim() == 3:
            static = static.unsqueeze(0)
        if static.size(0) != B:
            static = (static.repeat_interleave(B // static.size(0), 0)
                      if B % static.size(0) == 0 else static.expand(B, -1, -1, -1))
        return static.to(device)

    def forward(self, x0, xT, tau, cond=None, static=None):
        B, _, H, W = x0.shape
        tau_b = tau.view(-1, 1, 1, 1) if tau.dim() == 1 else tau.view(B, 1, 1, 1)
        trajectory_tau = (
            torch.full_like(tau.view(-1), 0.5)
            if self.query_independent_trajectory
            else tau.view(-1)
        )
        temb = self.time_mlp(trajectory_tau)
        if temb.size(0) != B and B % temb.size(0) == 0:
            temb = temb.repeat_interleave(B // temb.size(0), 0)

        st = self._prep_static(static, B, x0.device)
        parts = [x0, xT, xT - x0]
        if st is not None:
            parts.append(st)
        h = torch.cat(parts, dim=1)

        feats = []
        for blk in self.encoder:
            h = blk(h); feats.append(h)
        h = self.adaln(h, temb)
        bott = h if self.dual_stream else None   # global latent for decoder stream

        coarse_feat = None
        for i in range(self.n_levels):
            h = self.up[i](h)
            skip = feats[-(i + 2)]
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear",
                                   align_corners=False)
            if self.use_skip:
                if self.gated_skip:
                    h = self.skip[i](h, skip, temb)
                else:
                    h = self.skip[i](torch.cat([h, skip], dim=1))
            h = self.dec[i](h)
            if i == self.n_levels - 2:
                coarse_feat = h  # one level below full res -> coarse residual

        # upsample stem feature to full res if needed
        if h.shape[-2:] != (H, W):
            h = F.interpolate(h, size=(H, W), mode="bilinear", align_corners=False)

        # --- flow warp (explicit transport) ---
        fa = torch.tanh(self.flow_head(h)) * self.flow_scale       # (B,4|8,H,W)
        tf = tau_b                                                  # tau in [0,1]
        tr = 1.0 - tau_b                                            # reverse time
        if self.use_accel:
            # constant-acceleration displacement: F*t + 1/2 A*t^2 (smooth in tau)
            w0 = warp(
                x0,
                fa[:, :2] * tf + 0.5 * fa[:, 4:6] * tf * tf,
                periodic_longitude=self.spherical_ops,
            )
            wT = warp(
                xT,
                fa[:, 2:4] * tr + 0.5 * fa[:, 6:8] * tr * tr,
                periodic_longitude=self.spherical_ops,
            )
        else:
            w0 = warp(
                x0,
                fa[:, :2] * tf,
                periodic_longitude=self.spherical_ops,
            )
            wT = warp(
                xT,
                fa[:, 2:4] * tr,
                periodic_longitude=self.spherical_ops,
            )
        flow = fa
        # --- learned blend (fixes linear-chord error) ---
        blend_logits = self.blend_head(h)
        if self.endpoint_preserving:
            alpha = (
                (1.0 - tau_b)
                + 2.0
                * tau_b
                * (1.0 - tau_b)
                * torch.tanh(blend_logits)
            ).clamp(0.0, 1.0)
        else:
            alpha = torch.sigmoid(blend_logits + (1.0 - 2.0 * tau_b))
        warped = alpha * w0 + (1.0 - alpha) * wT
        # per-channel warp gate: blend warped frames with the un-warped bilinear
        # scaffold so static fields (t2m/mslp) can opt out of transport.
        x_bilinear = (1.0 - tau_b) * x0 + tau_b * xT
        beta = torch.sigmoid(self.warp_gate).view(1, -1, 1, 1)
        warped = beta * warped + (1.0 - beta) * x_bilinear
        # --- pyramid residual ---
        delta = self.res_fine(self.res_body(h))
        if coarse_feat is not None:
            dc = self.res_coarse(coarse_feat)
            delta = delta + F.interpolate(dc, size=(H, W), mode="bilinear",
                                          align_corners=False)
        if self.spectral is not None:            # global low-wavenumber correction
            delta = delta + self.spectral(h).to(delta.dtype)
        s = torch.tanh(self.scale).view(1, -1, 1, 1)
        endpoint_factor = (
            4.0 * tau_b * (1.0 - tau_b)
            if self.endpoint_preserving
            else None
        )
        x_hat = (
            warped + endpoint_factor * s * delta
            if endpoint_factor is not None
            else warped + s * delta
        )
        if self.dual_stream:
            # global skip-less decoder stream (DC-AE-like) from the bottleneck
            g = bott
            for blk in self.gdec:
                g = blk(g)
            if g.shape[-2:] != (H, W):
                g = F.interpolate(g, size=(H, W), mode="bilinear", align_corners=False)
            global_delta = self.gout(g)
            x_decoder = (
                x_bilinear + endpoint_factor * global_delta
                if endpoint_factor is not None
                else x_bilinear + global_delta
            )
            gate = torch.sigmoid(self.stream_gate).view(1, -1, 1, 1)
            x_hat = gate * x_hat + (1.0 - gate) * x_decoder
        if self.hydro is not None:
            # hydrostatic coupling: nudge Z1000-700 + mslp from the predicted
            # T column (thickness ~ layer-mean T). Zero-init so neutral at start.
            corr = self.hydro(x_hat[:, 0:4]).to(x_hat.dtype)
            if endpoint_factor is not None:
                corr = endpoint_factor * corr
            idx = torch.tensor([16, 17, 18, 19, 23], device=x_hat.device)
            x_hat = x_hat.index_add(1, idx, corr)              # dtype-safe channel add
        if self.endpoint_preserving:
            x_hat = torch.where(tau_b == 0, x0, x_hat)
            x_hat = torch.where(tau_b == 1, xT, x_hat)
        return x_hat, {
            "warped": warped,
            "delta": delta,
            "flow": flow,
            "alpha": alpha,
        }


# Legacy import retained for checkpoints/scripts created before the paper
# nomenclature was finalised. New code should import ``WeatherBridgeModel``.
WeatherBridgeFlowModel = WeatherBridgeModel
