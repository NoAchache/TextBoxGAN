import os
from functools import partial

import numpy as np
import tensorflow as tf

from config import cfg
from models.stylegan2.layers.upfirdn import custom_ops
from models.stylegan2.utils import apply_conv_in_good_format


def _get_plugin():
    loc = os.path.dirname(os.path.abspath(__file__))
    cu_fn = "upfirdn_2d.cu"
    return custom_ops.get_plugin(os.path.join(loc, cu_fn))


def _setup_kernel(k):
    k = np.asarray(k, dtype=np.float32)
    if k.ndim == 1:
        k = np.outer(k, k)
    k /= np.sum(k)
    assert k.ndim == 2
    assert k.shape[0] == k.shape[1]
    return k


def compute_paddings(resample_kernel, up, down, is_conv, convW=3, factor=2, gain=1):
    assert not (up and down)

    k = [1] * factor if resample_kernel is None else resample_kernel
    if up:
        k = _setup_kernel(k) * (gain * (factor**2))
        if is_conv:
            p = (k.shape[0] - factor) - (convW - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1
        else:
            p = k.shape[0] - factor
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2
    elif down:
        k = _setup_kernel(k) * gain
        if is_conv:
            p = (k.shape[0] - factor) + (convW - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2 + 1
        else:
            p = k.shape[0] - factor
            pad0 = (p + 1) // 2
            pad1 = p // 2
    else:
        k = resample_kernel
        pad0, pad1 = 0, 0
    return k, pad0, pad1


def upsample_2d(x, res_h, res_w, pad0, pad1, k, factor=2):
    assert isinstance(factor, int) and factor >= 1
    return _simple_upfirdn_2d(
        x, res_h, res_w, k, up_x=factor, up_y=factor, pad0=pad0, pad1=pad1
    )


def upsample_conv_2d(x, w_res, h_res, w, pad0, pad1, k, factor=2):

    # Check weight shape.
    w = tf.convert_to_tensor(w)
    assert w.shape.rank == 4
    convH = w.shape[0]
    convW = w.shape[1]
    inC = tf.shape(w)[2]
    outC = tf.shape(w)[3]

    num_groups = tf.shape(x)[1] // inC

    # Transpose weights.
    w = tf.reshape(w, [convH, convW, inC, num_groups, -1])
    w = tf.transpose(w[::-1, ::-1], [0, 1, 4, 3, 2])
    w = tf.reshape(w, [convH, convW, -1, num_groups * inC])

    output_height = (h_res - 1) * factor + convH
    output_width = (w_res - 1) * factor + convW

    # Execute.
    output_shape = (
        [
            tf.shape(x)[0],
            output_height,
            output_width,
            outC,
        ]
        if cfg.cpu_only
        else [tf.shape(x)[0], outC, output_height, output_width]
    )

    partial_conv_func = partial(tf.nn.conv2d_transpose, output_shape=output_shape)

    x = apply_conv_in_good_format(
        x, partial_conv_func, filters=w, h_w_stride=(factor, factor), padding="VALID"
    )

    return _simple_upfirdn_2d(x, output_height, output_width, k, pad0=pad0, pad1=pad1)


def conv_downsample_2d(x, w_res, h_res, w, pad0, pad1, k, reduce_height):
    w = tf.convert_to_tensor(w)
    h_stride = 2 if reduce_height else 1
    w_stride = 2
    x = _simple_upfirdn_2d(x, w_res, h_res, k, pad0=pad0, pad1=pad1)
    return apply_conv_in_good_format(
        x, tf.nn.conv2d, filters=w, h_w_stride=(h_stride, w_stride), padding="VALID"
    )


def upfirdn_2d(
    x, k, upx=1, upy=1, downx=1, downy=1, padx0=0, padx1=0, pady0=0, pady1=0
):
    r"""Pad, upsample, FIR filter, and downsample a batch of 2D images.
    Accepts a batch of 2D images of the shape `[majorDim, inH, inW, minorDim]`
    and performs the following operations for each image, batched across
    `majorDim` and `minorDim`:
    1. Pad the image with zeros by the specified number of pixels on each side
       (`padx0`, `padx1`, `pady0`, `pady1`). Specifying a negative value
       corresponds to cropping the image.
    2. Upsample the image by inserting the zeros after each pixel (`upx`, `upy`).
    3. Convolve the image with the specified 2D FIR filter (`k`), shrinking the
       image so that the footprint of all output pixels lies within the input image.
    4. Downsample the image by throwing away pixels (`downx`, `downy`).
    This sequence of operations bears close resemblance to scipy.signal.upfirdn().
    The fused op is considerably more efficient than performing the same calculation
    using standard TensorFlow ops. It supports gradients of arbitrary order.
    Args:
        x:      Input tensor of the shape `[majorDim, inH, inW, minorDim]`.
        k:      2D FIR filter of the shape `[firH, firW]`.
        upx:    Integer upsampling factor along the X-axis (default: 1).
        upy:    Integer upsampling factor along the Y-axis (default: 1).
        downx:  Integer downsampling factor along the X-axis (default: 1).
        downy:  Integer downsampling factor along the Y-axis (default: 1).
        padx0:  Number of pixels to pad on the left side (default: 0).
        padx1:  Number of pixels to pad on the right side (default: 0).
        pady0:  Number of pixels to pad on the top side (default: 0).
        pady1:  Number of pixels to pad on the bottom side (default: 0).
        impl:   Name of the implementation to use. Can be `"ref"` or `"cuda"` (default).
    Returns:
        Tensor of the shape `[majorDim, outH, outW, minorDim]`, and same datatype as `x`.
    """

    upfirdn_2d_func = (
        upfirdn_2d_cuda if tf.test.is_built_with_cuda() else upfirdn_2d_ref
    )
    return upfirdn_2d_func(
        x=x,
        k=k,
        upx=upx,
        upy=upy,
        downx=downx,
        downy=downy,
        padx0=padx0,
        padx1=padx1,
        pady0=pady0,
        pady1=pady1,
    )


def _simple_upfirdn_2d(x, x_res_h, x_res_w, k, up_x=1, up_y=1, down=1, pad0=0, pad1=0):
    assert x.shape.rank == 4
    y = x
    y = tf.reshape(y, [-1, x_res_h, x_res_w, 1])
    y = upfirdn_2d(
        y,
        k,
        upx=up_x,
        upy=up_y,
        downx=down,
        downy=down,
        padx0=pad0,
        padx1=pad1,
        pady0=pad0,
        pady1=pad1,
    )
    y = tf.reshape(y, [-1, tf.shape(x)[1], tf.shape(y)[1], tf.shape(y)[2]])
    return y


def upfirdn_2d_cuda(x, k, upx, upy, downx, downy, padx0, padx1, pady0, pady1):
    """Fast CUDA implementation of `upfirdn_2d()` using custom ops."""

    x = tf.convert_to_tensor(x)
    k = np.asarray(k, dtype=np.float32)
    majorDim, inH, inW, minorDim = x.shape.as_list()
    kernelH, kernelW = k.shape
    assert inW >= 1 and inH >= 1
    assert kernelW >= 1 and kernelH >= 1
    assert isinstance(upx, int) and isinstance(upy, int)
    assert isinstance(downx, int) and isinstance(downy, int)
    assert isinstance(padx0, int) and isinstance(padx1, int)
    assert isinstance(pady0, int) and isinstance(pady1, int)

    outW = (inW * upx + padx0 + padx1 - kernelW) // downx + 1
    outH = (inH * upy + pady0 + pady1 - kernelH) // downy + 1
    assert outW >= 1 and outH >= 1

    kc = tf.constant(k, dtype=x.dtype)
    gkc = tf.constant(k[::-1, ::-1], dtype=x.dtype)
    gpadx0 = kernelW - padx0 - 1
    gpady0 = kernelH - pady0 - 1
    gpadx1 = inW * upx - outW * downx + padx0 - upx + 1
    gpady1 = inH * upy - outH * downy + pady0 - upy + 1

    @tf.custom_gradient
    def func(x):
        y = _get_plugin().up_fir_dn2d(
            x=x,
            k=kc,
            upx=upx,
            upy=upy,
            downx=downx,
            downy=downy,
            padx0=padx0,
            padx1=padx1,
            pady0=pady0,
            pady1=pady1,
        )
        y.set_shape([majorDim, outH, outW, minorDim])

        @tf.custom_gradient
        def grad(dy):
            dx = _get_plugin().up_fir_dn2d(
                x=dy,
                k=gkc,
                upx=downx,
                upy=downy,
                downx=upx,
                downy=upy,
                padx0=gpadx0,
                padx1=gpadx1,
                pady0=gpady0,
                pady1=gpady1,
            )
            dx.set_shape([majorDim, inH, inW, minorDim])
            return dx, func

        return y, grad

    return func(x)


def upfirdn_2d_ref(x, k, upx, upy, downx, downy, padx0, padx1, pady0, pady1):
    """Slow reference implementation of `upfirdn_2d()` using standard TensorFlow ops."""

    x = tf.convert_to_tensor(x)
    k = np.asarray(k, dtype=np.float32)
    _, inH, inW, minorDim = x.shape.as_list()
    kernelH, kernelW = k.shape
    assert inW >= 1 and inH >= 1
    assert kernelW >= 1 and kernelH >= 1
    assert isinstance(upx, int) and isinstance(upy, int)
    assert isinstance(downx, int) and isinstance(downy, int)
    assert isinstance(padx0, int) and isinstance(padx1, int)
    assert isinstance(pady0, int) and isinstance(pady1, int)
    # Upsample (insert zeros).
    x = tf.reshape(x, [-1, inH, 1, inW, 1, minorDim])
    x = tf.pad(x, [[0, 0], [0, 0], [0, upy - 1], [0, 0], [0, upx - 1], [0, 0]])
    x = tf.reshape(x, [-1, inH * upy, inW * upx, minorDim])

    # Pad (crop if negative).
    x = tf.pad(
        x,
        [
            [0, 0],
            [max(pady0, 0), max(pady1, 0)],
            [max(padx0, 0), max(padx1, 0)],
            [0, 0],
        ],
    )
    x = x[
        :,
        max(-pady0, 0) : x.shape[1] - max(-pady1, 0),
        max(-padx0, 0) : x.shape[2] - max(-padx1, 0),
        :,
    ]

    # Convolve with filter.
    x = tf.transpose(x, [0, 3, 1, 2])
    x = tf.reshape(x, [-1, 1, inH * upy + pady0 + pady1, inW * upx + padx0 + padx1])
    w = tf.constant(k[::-1, ::-1, np.newaxis, np.newaxis], dtype=x.dtype)

    x = apply_conv_in_good_format(
        x, tf.nn.conv2d, filters=w, h_w_stride=(1, 1), padding="VALID"
    )

    x = tf.reshape(
        x,
        [
            -1,
            minorDim,
            inH * upy + pady0 + pady1 - kernelH + 1,
            inW * upx + padx0 + padx1 - kernelW + 1,
        ],
    )
    x = tf.transpose(x, [0, 2, 3, 1])

    # Downsample (throw away pixels).
    return x[:, ::downy, ::downx, :]
