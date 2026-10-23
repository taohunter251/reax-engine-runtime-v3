import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice

@triton.jit
def _prepare(x, low_columns, high_columns, smoothing, centers, low, high, scales,
             K: tl.constexpr, KL: tl.constexpr, KH: tl.constexpr,
             BL: tl.constexpr, BH: tl.constexpr, CENTER: tl.constexpr):
    row = tl.program_id(0)
    if KL:
        col = tl.arange(0, BL)
        source = tl.load(low_columns + col, mask=col < KL, other=0)
        value = tl.load(x + row * K + source, mask=col < KL, other=0.).to(tl.float32)
        if CENTER:
            mean = tl.load(centers + col, mask=col < KL, other=0.)
            value -= mean
        sm = tl.load(smoothing + col, mask=col < KL, other=1.)
        value /= sm
        scale = tl.maximum(tl.max(tl.abs(value)), 1.e-12) / 127.
        quantized = libdevice.nearbyint(libdevice.div_rn(value, scale))
        quantized = tl.minimum(tl.maximum(quantized, -127.), 127.)
        tl.store(low + row * KL + col, quantized.to(tl.int8), mask=col < KL)
        tl.store(scales + row, scale)
    else:
        tl.store(scales + row, 1.)
    col_h = tl.arange(0, BH)
    source_h = tl.load(high_columns + col_h, mask=col_h < KH, other=0)
    value_h = tl.load(x + row * K + source_h, mask=col_h < KH, other=0.)
    tl.store(high + row * KH + col_h, value_h, mask=col_h < KH)

@triton.jit
def prepare_kernel(x,low_columns,high_columns,center,smoothing,low,high,scales,
                   M:tl.constexpr,K:tl.constexpr,G:tl.constexpr,HP:tl.constexpr,P:tl.constexpr,
                   B:tl.constexpr,BH:tl.constexpr):
    row,part=tl.program_id(0),tl.program_id(1)
    offset=tl.arange(0,B)
    position=part*G+offset
    source=tl.load(low_columns+position,mask=offset<G,other=-1)
    value=tl.load(x+row*K+tl.maximum(source,0),mask=(offset<G)&(source>=0),other=0.).to(tl.float32)
    mean=tl.load(center+position,mask=offset<G,other=0.)
    smooth=tl.load(smoothing+position,mask=offset<G,other=1.)
    value=tl.where(source>=0,(value-mean)/smooth,0.)
    scale=tl.maximum(tl.max(tl.abs(value)),1.e-12)/127.
    codes=libdevice.nearbyint(libdevice.div_rn(value,scale))
    tl.store(low+row*(P*G)+position,codes.to(tl.int8),mask=offset<G)
    tl.store(scales+part*M+row,scale)
    if HP:
        offset_h=tl.arange(0,BH)
        index=part*HP+offset_h
        source_h=tl.load(high_columns+index,mask=offset_h<HP,other=-1)
        value_h=tl.load(x+row*K+tl.maximum(source_h,0),mask=(offset_h<HP)&(source_h>=0),other=0.)
        tl.store(high+row*(P*HP)+index,value_h,mask=offset_h<HP)


@triton.jit
def reduce_kernel(partials,out,TOTAL:tl.constexpr,P:tl.constexpr,B:tl.constexpr):
    index=tl.program_id(0)*B+tl.arange(0,B)
    value=tl.load(partials+index,mask=index<TOTAL,other=0.).to(tl.float32)
    for part in tl.static_range(1,P):
        value=value+tl.load(partials+part*TOTAL+index,mask=index<TOTAL,other=0.).to(tl.float32)
    tl.store(out+index,value.to(tl.bfloat16),mask=index<TOTAL)

