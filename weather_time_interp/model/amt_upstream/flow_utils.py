import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import ImageFile
from torch.nn.modules.utils import _pair

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _antipodal_shift(x):
    """Shift longitude by 180 degrees, including odd coarse-grid widths."""
    width = x.size(-1)
    half = width // 2
    shifted = torch.roll(x, shifts=half, dims=-1)
    if width % 2:
        shifted = 0.5 * (
            shifted
            + torch.roll(x, shifts=half + 1, dims=-1)
        )
    return shifted


def _pole_parity_view(x, pole_parity):
    if pole_parity is None:
        return None
    parity = torch.as_tensor(
        pole_parity,
        device=x.device,
        dtype=x.dtype,
    )
    if parity.numel() != x.size(-3):
        raise ValueError("pole parity must have one value per input channel")
    return parity.reshape(1, -1, 1, 1)


def _pole_ghost_rows(x, count, *, top, pole_parity=None):
    if count == 0:
        return x[..., :0, :]
    if count > x.size(-2):
        raise ValueError("pole padding exceeds the latitude height")
    source = x[..., :count, :] if top else x[..., -count:, :]
    ghosts = _antipodal_shift(source.flip(-2))
    parity = _pole_parity_view(x, pole_parity)
    return ghosts if parity is None else ghosts * parity


def spherical_pad(x, padding, pole_parity=None):
    """Pad a lat-lon tensor with periodic longitude and antipodal poles.

    ``padding`` follows ``torch.nn.functional.pad`` order:
    ``(left, right, top, bottom)``. An integer requests equal padding.
    """
    if isinstance(padding, int):
        left = right = top = bottom = padding
    else:
        if len(padding) != 4:
            raise ValueError("spherical padding must have four entries")
        left, right, top, bottom = (int(value) for value in padding)
    if min(left, right, top, bottom) < 0:
        raise ValueError("spherical padding must be non-negative")
    if x.ndim != 4:
        raise ValueError("spherical padding expects a BCHW tensor")
    if x.size(-2) < 1 or x.size(-1) < 1:
        raise ValueError("cannot pad an empty spatial grid")

    top_rows = _pole_ghost_rows(
        x,
        top,
        top=True,
        pole_parity=pole_parity,
    )
    bottom_rows = _pole_ghost_rows(
        x,
        bottom,
        top=False,
        pole_parity=pole_parity,
    )
    padded = torch.cat((top_rows, x, bottom_rows), dim=-2)
    if left or right:
        indices = torch.arange(
            -left,
            x.size(-1) + right,
            device=x.device,
        ).remainder(x.size(-1))
        padded = padded.index_select(-1, indices)
    return padded


def _pole_pad_latitude(x, pad, pole_parity=None):
    """Reflect latitude ghost rows through each pole at the antipode."""
    if pad == 0:
        return x
    return spherical_pad(
        x,
        (0, 0, int(pad), int(pad)),
        pole_parity=pole_parity,
    )


def spherical_resize(x, scale_factor):
    """Bilinear feature resize that preserves spherical edge continuity."""
    scale_h, scale_w = _pair(scale_factor)
    scale_h = float(scale_h)
    scale_w = float(scale_w)
    if scale_h <= 0 or scale_w <= 0:
        raise ValueError("resize scale factors must be positive")
    if scale_h <= 1.0 and scale_w <= 1.0:
        return F.interpolate(
            x,
            scale_factor=(scale_h, scale_w),
            mode="bilinear",
            align_corners=False,
        )
    if not scale_h.is_integer() or not scale_w.is_integer():
        raise ValueError("spherical upsampling requires integer scale factors")

    up_h = int(scale_h)
    up_w = int(scale_w)
    expanded = spherical_pad(x, (1, 1, 1, 1))
    resized = F.interpolate(
        expanded,
        scale_factor=(up_h, up_w),
        mode="bilinear",
        align_corners=False,
    )
    target_h = x.size(-2) * up_h
    target_w = x.size(-1) * up_w
    return resized[
        ...,
        up_h : up_h + target_h,
        up_w : up_w + target_w,
    ]


def spherical_avg_pool2d(x):
    """Stride-2 average pooling without dropping odd pole/seam cells."""
    pad_bottom = x.size(-2) % 2
    pad_right = x.size(-1) % 2
    if pad_bottom or pad_right:
        x = spherical_pad(x, (0, pad_right, 0, pad_bottom))
    return F.avg_pool2d(x, kernel_size=2, stride=2)


class SphereConv2d(nn.Conv2d):
    """Conv2d with state-compatible spherical boundary padding."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
        pole_parity=None,
    ):
        if padding_mode != "zeros":
            raise ValueError("SphereConv2d manages its own padding")
        sphere_padding = _pair(padding)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode="zeros",
            device=device,
            dtype=dtype,
        )
        self.sphere_padding = sphere_padding
        self.register_buffer(
            "pole_parity",
            (
                None
                if pole_parity is None
                else torch.as_tensor(pole_parity, dtype=torch.float32)
            ),
            persistent=False,
        )

    def forward(self, x):
        pad_h, pad_w = self.sphere_padding
        padded = spherical_pad(
            x,
            (pad_w, pad_w, pad_h, pad_h),
            pole_parity=self.pole_parity,
        )
        return F.conv2d(
            padded,
            self.weight,
            self.bias,
            self.stride,
            0,
            self.dilation,
            self.groups,
        )


class SphereConvTranspose2d(nn.ConvTranspose2d):
    """4x4 stride-2 transpose convolution with spherical input ghosts."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=4,
        stride=2,
        padding=1,
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        if (
            _pair(kernel_size) != (4, 4)
            or _pair(stride) != (2, 2)
            or _pair(padding) != (1, 1)
            or _pair(output_padding) != (0, 0)
            or _pair(dilation) != (1, 1)
            or padding_mode != "zeros"
        ):
            raise ValueError(
                "SphereConvTranspose2d supports AMT 4x4/stride-2 geometry"
            )
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

    def forward(self, x, output_size=None):
        if output_size is not None:
            raise ValueError(
                "SphereConvTranspose2d does not support output_size"
            )
        expanded = spherical_pad(x, (1, 1, 1, 1))
        output = F.conv_transpose2d(
            expanded,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
            self.dilation,
        )
        return output[..., 2:-2, 2:-2]


def spherical_grid_sample(img, coords, pole_pad=16):
    """Sample pixel coordinates on a periodic lat-lon sphere."""
    if coords.size(-1) != 2:
        raise ValueError("spherical coordinates must end with (x, y)")
    height, width = img.shape[-2:]
    if height < 1 or width < 1:
        raise ValueError("cannot sample an empty spatial grid")

    if height == 1:
        sample = img
        source_y = torch.zeros_like(coords[..., 1], dtype=torch.float32)
    else:
        pad = min(int(pole_pad), height)
        sample = _pole_pad_latitude(img, pad)
        source_y = coords[..., 1].float() + pad
    sample = torch.cat((sample, sample[..., :1]), dim=-1)
    source_x = torch.remainder(coords[..., 0].float(), width)
    grid = torch.stack(
        (
            2.0 * source_x / max(width, 1) - 1.0,
            2.0 * source_y / max(sample.size(-2) - 1, 1) - 1.0,
        ),
        dim=-1,
    )
    with torch.autocast(device_type=img.device.type, enabled=False):
        output = F.grid_sample(
            sample.float(),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
    return output.to(img.dtype)


def warp(img, flow):
    """AMT feature warp with spherical longitude and pole boundaries."""
    batch, _, height, width = flow.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=img.device, dtype=torch.float32),
        torch.arange(width, device=img.device, dtype=torch.float32),
        indexing="ij",
    )
    coords = torch.stack(
        (
            xx[None] + flow[:, 0].float(),
            yy[None] + flow[:, 1].float(),
        ),
        dim=-1,
    )
    return spherical_grid_sample(img, coords)


def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.
    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel

def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image

def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.
    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)
