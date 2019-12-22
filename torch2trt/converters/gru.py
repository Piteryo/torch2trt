from torch2trt.torch2trt import *

@tensorrt_converter('torch.nn.GRUCell.forward')
def convert_gru(ctx):
    module = ctx.method_args[0]
    input = ctx.method_args[1]
    hidden = ctx.method_args[2]
    input_trt = trt_(ctx.network, input)
    hidden_trt = trt_(ctx.network, hidden)
    output = ctx.method_return

    # kernel_size = (module.kernel_size[0], 1)
    # stride = (module.stride[0], 1)
    # padding = (module.padding[0], 0)
    # dilation = (module.dilation[0], 1)
    layer_count = 1
    hidden_size = module.hidden_size
    max_seq_len = 1

    kernel_hh = module.weight_hh.detach().cpu().numpy()

    rh = kernel_hh[:512,:]
    zh = kernel_hh[512:1024, :]
    hh = kernel_hh[1024:1536,:]
    kernel_ih = module.weight_ih.detach().cpu().numpy()

    ri = kernel_ih[:512,:]
    zi = kernel_ih[512:1024, :]
    hi = kernel_ih[1024:1536,:]

    #bias = trt.Weights(torch_dtype_to_trt(module.weight.dtype))
    # if module.bias is not None:
    bias_hh = module.bias_hh.detach().cpu().numpy()
    brh = bias_hh[:512]
    bzh = bias_hh[512:1024]
    bhh = bias_hh[1024:1536]
    bias_ih = module.bias_ih.detach().cpu().numpy()

    bri = bias_ih[:512]
    bzi = bias_ih[512:1024]
    bhi = bias_ih[1024:1536]

    # reshape to 2D
    layer_hidden = ctx.network.add_shuffle(hidden_trt)
    layer_hidden.reshape_dims = (1, -1)
    
    layer_input = ctx.network.add_shuffle(input_trt)
    layer_input.reshape_dims = (1, -1)

    layer_rep = ctx.network.add_rnn_v2(
        input=layer_input.get_output(0),
        layer_count=layer_count,
        hidden_size=hidden_size,
        max_seq_length = 1,
        op=trt.tensorrt.RNNOperation.GRU)
    
    layer_rep.hidden_state = layer_hidden.get_output(0)

    print(layer_rep)

    layer_rep.set_weights_for_gate(0,trt.tensorrt.RNNGateType.UPDATE, True, zi)
    layer_rep.set_weights_for_gate(0,trt.tensorrt.RNNGateType.RESET, True, ri)
    layer_rep.set_weights_for_gate(0,trt.tensorrt.RNNGateType.HIDDEN, True, hi)


    layer_rep.set_weights_for_gate(0,trt.tensorrt.RNNGateType.UPDATE, False, zh)
    layer_rep.set_weights_for_gate(0,trt.tensorrt.RNNGateType.RESET, False, rh)
    layer_rep.set_weights_for_gate(0,trt.tensorrt.RNNGateType.HIDDEN, False, hh)


    # layer_rep.set_weights_for_gate(0,trt.tensorrt.RNNGateType.CELL, False, z)
    # layer_rep.set_weights_for_gate(0,trt.tensorrt.RNNGateType.FORGET, False, r)
    # layer_rep.set_weights_for_gate(0,trt.tensorrt.RNNGateType.OUTPUT, False, h)
    # layer_rep.set_weights_for_gate(0,trt.tensorrt.RNNGateType.INPUT, False, h)


    layer_rep.set_bias_for_gate(0, trt.tensorrt.RNNGateType.UPDATE, True, bzi)
    layer_rep.set_bias_for_gate(0, trt.tensorrt.RNNGateType.RESET, True, bri)
    layer_rep.set_bias_for_gate(0, trt.tensorrt.RNNGateType.HIDDEN, True, bhi)


    layer_rep.set_bias_for_gate(0, trt.tensorrt.RNNGateType.UPDATE, False, bzh)
    layer_rep.set_bias_for_gate(0, trt.tensorrt.RNNGateType.RESET, False, brh)
    layer_rep.set_bias_for_gate(0, trt.tensorrt.RNNGateType.HIDDEN, False, bhh)

    # layer_rep.set_bias_for_gate(0, trt.tensorrt.RNNGateType.CELL, False, br)
    # layer_rep.set_bias_for_gate(0, trt.tensorrt.RNNGateType.FORGET, False, bz)
    # layer_rep.set_bias_for_gate(0, trt.tensorrt.RNNGateType.OUTPUT, False, bh)

    # layer_rep.set_bias_for_gate(0, trt.tensorrt.RNNGateType.INPUT, False, br)


        
    # reshape back to 1D
    layer_res = ctx.network.add_shuffle(layer_rep.get_output(0))
    layer_res.reshape_dims = (output.shape[-1],)

    output._trt = layer_res.get_output(0)
