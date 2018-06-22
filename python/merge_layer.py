# -*- coding: utf-8 -*-
import numpy as np  
import os.path as osp
import google.protobuf as pb
import google.protobuf.text_format
import argparse, time, sys, os, copy
caffe_root = './'
os.chdir(caffe_root)
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe

GPU_ID = 0
caffe.set_mode_gpu()
caffe.set_device(GPU_ID)

can_be_merged_layers = ['Convolution', 'BatchNorm', 'Scale']
sequence_dict = {('BatchNorm', 'Scale', 'Convolution', 'BatchNorm', 'Scale'):1, 
                 ('Convolution', 'BatchNorm', 'Scale'):2, 
                 ('BatchNorm', 'Scale', 'Convolution'):3, 
                 ('BatchNorm', 'Scale'):4}

def load_and_fill_biases(src_model, src_weights, dst_model):
    with open(src_model) as f:
        model = caffe.proto.caffe_pb2.NetParameter()
        pb.text_format.Merge(f.read(), model)

    for i, layer in enumerate(model.layer):
        if layer.type == 'Convolution':
            # Add bias layer if needed
            if layer.convolution_param.bias_term == False:
                layer.convolution_param.bias_term = True
                layer.convolution_param.bias_filler.type = 'constant'
                layer.convolution_param.bias_filler.value = 0.0

    with open(dst_model, 'w') as f:
        f.write(pb.text_format.MessageToString(model))
    
    net_src = caffe.Net(src_model, src_weights, caffe.TEST)
    net_dst = caffe.Net(dst_model, caffe.TEST)
    for key in net_src.params.keys():
        for i in range(len(net_src.params[key])):
            net_dst.params[key][i].data[:] = net_src.params[key][i].data[:]

    return net_dst

def copy_double(data):
    return np.array(data, copy = True, dtype = np.double)

def merge_ConvBnScale(net, sequence_key, eps = 0.0):
    key_conv  = sequence_key[0]
    key_bn    = sequence_key[1]
    key_scale = sequence_key[2]

    print 'Combine {:s} + {:s} + {:s}'.format(key_conv, key_bn, key_scale)
    # Copy
    bn_mean        = copy_double(net.params[key_bn][0].data)
    bn_variance    = copy_double(net.params[key_bn][1].data)
    num_bn_samples = copy_double(net.params[key_bn][2].data)

    bn_mean     = bn_mean / num_bn_samples[0]
    bn_variance = np.sqrt(bn_variance / num_bn_samples[0] + eps)

    # and Invalidate the BN layer
    net.params[key_bn][0].data[:] = 0
    net.params[key_bn][1].data[:] = 1
    net.params[key_bn][2].data[:] = 1
    if num_bn_samples[0] == 0:
        num_bn_samples[0] = 1
    
    scale_weight = copy_double(net.params[key_scale][0].data)
    scale_bias   = copy_double(net.params[key_scale][1].data)
    net.params[key_scale][0].data[:] = 1
    net.params[key_scale][1].data[:] = 0

    weight = copy_double(net.params[key_conv][0].data)
    bias   = copy_double(net.params[key_conv][1].data)
    alpha  = scale_weight / bn_variance
    net.params[key_conv][1].data[:] = bias * alpha + (scale_bias - bn_mean * alpha)
    for i in range(len(alpha)):
        net.params[key_conv][0].data[i] = weight[i] * alpha[i]
    return net

def merge_BnScaleConv(net, sequence_key, eps = 0.0):
    key_bn    = sequence_key[0]
    key_scale = sequence_key[1]
    key_conv  = sequence_key[2]

    print 'Combine {:s} + {:s} + {:s}'.format(key_bn, key_scale, key_conv)
    # Copy
    bn_mean        = copy_double(net.params[key_bn][0].data)
    bn_variance    = copy_double(net.params[key_bn][1].data)
    num_bn_samples = copy_double(net.params[key_bn][2].data)

    bn_mean     = bn_mean / num_bn_samples[0]
    bn_variance = np.sqrt(bn_variance / num_bn_samples[0] + eps)

    # and Invalidate the BN layer
    net.params[key_bn][0].data[:] = 0
    net.params[key_bn][1].data[:] = 1
    net.params[key_bn][2].data[:] = 1
    if num_bn_samples[0] == 0:
        num_bn_samples[0] = 1

    scale_weight = copy_double(net.params[key_scale][0].data)
    scale_bias   = copy_double(net.params[key_scale][1].data)
    net.params[key_scale][0].data[:] = 1
    net.params[key_scale][1].data[:] = 0

    weight = copy_double(net.params[key_conv][0].data)
    bias   = copy_double(net.params[key_conv][1].data)

    w_factor = scale_weight / bn_variance
    for c_out in range(weight.shape[0]):
        b_factor = 0
        for c_in in range(weight.shape[1]):
            b_factor += (scale_bias[c_in] - w_factor[c_in]*bn_mean[c_in])*np.sum(weight[c_out, c_in,:,:])
        bias[c_out] += b_factor
        for c_in in range(weight.shape[1]):
            weight[c_out, c_in, :, :] = w_factor[c_in]*weight[c_out, c_in,:,:]

    net.params[key_conv][0].data[:] = weight
    net.params[key_conv][1].data[:] = bias

    return net

def merge_BnScale(net, sequence_key, eps = 0.0):
    key_bn    = sequence_key[0]
    key_scale = sequence_key[1]

    # Copy
    bn_mean        = copy_double(net.params[key_bn][0].data)
    bn_variance    = copy_double(net.params[key_bn][1].data)
    num_bn_samples = copy_double(net.params[key_bn][2].data)

    bn_mean     = bn_mean / num_bn_samples[0]
    bn_variance = np.sqrt(bn_variance / num_bn_samples[0] + eps)

    # and Invalidate the BN layer
    net.params[key_bn][0].data[:] = 0
    net.params[key_bn][1].data[:] = 1
    net.params[key_bn][2].data[:] = 1
    if num_bn_samples[0] == 0:
        num_bn_samples[0] = 1

    if net.params.has_key(key_scale):
        print 'Combine {:s} + {:s}'.format(key_bn, key_scale)
        scale_weight = copy_double(net.params[key_scale][0].data)
        scale_bias   = copy_double(net.params[key_scale][1].data)

    scale_weight = scale_weight / bn_variance
    scale_bias   = scale_bias - scale_weight * bn_mean 

    net.params[key_scale][0].data[:] = scale_weight
    net.params[key_scale][1].data[:] = scale_bias

    return net

def merge_layer_in_net(net, ConvBnScale_BNeps, BnScaleConv_BNeps, BnScale_BNeps):
    remained_layer_dict = {}
    i = 0
    num_layer = len(net.layers)
    while (i < num_layer):
        sequence_ids = []
        isFind_sequence_bottom = False
        sequence_bottom_isConv = False
        sequence_bottom_name = None
        sequence_top_name    = None 
        if net.layers[i].type in can_be_merged_layers:
            isFind_sequence_bottom = True
            if net.layers[i].type == 'Convolution':
                sequence_bottom_isConv = True
            sequence_ids.append(i)
            layer_top  = net.top_names[net._layer_names[i]][0]
            j = i + 1
            while (j < num_layer):
                bottoms_of_j = net.bottom_names[net._layer_names[j]]
                if (len(bottoms_of_j) == 1) and (layer_top in bottoms_of_j[0]) \
                  and (net.layers[j].type in can_be_merged_layers):
                    if sequence_bottom_isConv:
                        sequence_bottom_isConv = False
                        isFind_sequence_bottom = False
                        sequence_bottom_name = None
                    elif isFind_sequence_bottom:
                        isFind_sequence_bottom = False
                        bottoms_of_sequence = net.bottom_names[net._layer_names[i]][0]
                        k = i - 1
                        while (k >= 0):
                            if net.top_names[net._layer_names[k]][0] in bottoms_of_sequence:
                                sequence_bottom_name = net.top_names[net._layer_names[k]][0]
                            k -= 1
                    sequence_ids.append(j)
                    layer_top = net.top_names[net._layer_names[j]][0]
                    i = j
                else:
                    break 
                j += 1
            if len(sequence_ids) > 1:
                sequence_top_name = copy.copy(layer_top)
        i += 1
        
        if len(sequence_ids) > 1:
            sequence_name = [net._layer_names[k] for k in sequence_ids]
            sequence_key = tuple([net.layers[k].type for k in sequence_ids])
            if sequence_dict[sequence_key] == 1:
                net = merge_ConvBnScale(net, sequence_name[2:], eps = ConvBnScale_BNeps)
                net = merge_BnScaleConv(net, sequence_name[:3], eps = BnScaleConv_BNeps)
                conv_layer = sequence_name[2]
                remained_layer_dict[conv_layer] = [sequence_bottom_name, sequence_top_name]
            elif sequence_dict[sequence_key] == 2:
                net = merge_ConvBnScale(net, sequence_name, eps = ConvBnScale_BNeps)
                conv_layer = sequence_name[0]
                remained_layer_dict[conv_layer] = [sequence_bottom_name, sequence_top_name]
            elif sequence_dict[sequence_key] == 3:
                net = merge_BnScaleConv(net, sequence_name, eps = BnScaleConv_BNeps)
                conv_layer = sequence_name[2]
                remained_layer_dict[conv_layer] = [sequence_bottom_name, sequence_top_name]
            elif sequence_dict[sequence_key] == 4:
                net = merge_BnScale(net, sequence_name, eps = BnScale_BNeps)
                scale_layer = sequence_name[1]
                remained_layer_dict[scale_layer] = [sequence_bottom_name, sequence_top_name]
            else:
                print 'Not supperted yet, for merging', sequence_key 

    return net, remained_layer_dict

def get_merged_prototxt(net, src_prototxt, dst_prototxt, remained_layer_dict):
    # delete merged layer
    with open(src_prototxt) as f:
        prototxt = caffe.proto.caffe_pb2.NetParameter()
        pb.text_format.Merge(f.read(), prototxt)
    merged_layers = []
    for layer in prototxt.layer:
        if layer.type == 'BatchNorm':
            zero_mean   = np.all(net.params[layer.name][0].data == 0)
            one_var     = np.all(net.params[layer.name][1].data == 1)
            length_is_1 = (net.params[layer.name][2].data == 1)
            if zero_mean and one_var and length_is_1:
                print 'Delete layer: {}'.format(layer.name)
                merged_layers.append(layer)
        if layer.type == 'Scale':
            no_scaling = np.all(net.params[layer.name][0].data == 1)
            zero_bias  = np.all(net.params[layer.name][1].data == 0)
            if no_scaling and zero_bias:
                print 'Delete layer: {}'.format(layer.name)
                merged_layers.append(layer)
    map(prototxt.layer.remove, merged_layers)
    with open(dst_prototxt, 'w') as f:
        f.write(pb.text_format.MessageToString(prototxt))

    # update remained layer's bottom and top 
    with open(dst_prototxt, 'r') as fi:
        fi_list = fi.readlines()
    fo_list = copy.copy(fi_list)
    num_lines = len(fi_list)
    bottom_id = -1
    top_id    = -1
    for i, line in enumerate(fi_list):
        line = line.strip()
        if ('layer' in line) and ('{' in line):
            layer_mark = 0
            isUpdate  = False  
            j = i + 1
            while (j < num_lines):
                sub_line = fi_list[j].strip()
                if ('name' in sub_line) and (':' in sub_line):
                    layer_mark += 1    
                    layer_name = sub_line.split(':')[1].strip()[1:-1] 
                    if layer_name in remained_layer_dict.keys():
                        isUpdate = True
                    else:
                        break
                if ('bottom' in sub_line)  and (':' in sub_line):
                    bottom_id = j
                    layer_mark += 1
                if ('top' in sub_line)  and (':' in sub_line):
                    top_id = j 
                    layer_mark += 1
                if layer_mark == 3:
                    break
                j += 1 
            if isUpdate:
                new_bottom_name = remained_layer_dict[layer_name][0]
                new_top_name    = remained_layer_dict[layer_name][1]
                if new_bottom_name:
                    fo_list[bottom_id] = '  bottom: "{}"\n'.format(new_bottom_name)
                if new_top_name:
                    fo_list[top_id] = '  top: "{}"\n'.format(new_top_name)
                print "Update {}'s bottom and top: new bottom = {}, new top = {}".\
                    format(layer_name, new_bottom_name, new_top_name)
    with open(dst_prototxt, 'w') as fo:
        fo.writelines(fo_list)


def main():
    args = parse_args()
    src_model         = args.src_model
    src_weights       = args.src_weights
    merged_model      = args.merged_model
    merged_weights    = args.merged_weights
    ConvBnScale_BNeps = args.ConvBnScale_BNeps
    BnScaleConv_BNeps = args.BnScaleConv_BNeps
    BnScale_BNeps     = args.BnScale_BNeps
    # Set default output file names
    if merged_model is None:
        file_name = osp.splitext(src_model)[0]
        merged_model = file_name + '_bnmerge.prototxt'
    if merged_weights is None:
        file_name = osp.splitext(src_weights)[0]
        merged_weights = file_name + '_bnmerge.caffemodel'

    net = load_and_fill_biases(src_model, src_weights, src_model+'.temp.pt')
    net, remained_layer_dict = merge_layer_in_net(net, ConvBnScale_BNeps, BnScaleConv_BNeps, BnScale_BNeps)
    get_merged_prototxt(net, src_model+'.temp.pt', merged_model, remained_layer_dict)
    net.save(merged_weights)

def parse_args():
    parser = argparse.ArgumentParser(description="Merge Batch Normalized model for inference")
    parser.add_argument('--src_model', 
        default = 'models/***.prototxt')
    parser.add_argument('--src_weights', 
        default = 'models/***.caffemodel')
    parser.add_argument('--merged_model', 
        default = 'models/***_bnmerge_final.prototxt')
    parser.add_argument('--merged_weights', 
        default = 'models/***_bnmerge_final.caffemodel')
    parser.add_argument('--ConvBnScale_BNeps', default = 0.0, type = float)
    parser.add_argument('--BnScaleConv_BNeps', default = 0.0, type = float)
    parser.add_argument('--BnScale_BNeps', default = 0.0, type = float)
    args = parser.parse_args()
    if (not os.path.isfile(args.src_model)) or (not os.path.isfile(args.src_weights)):
        parser.print_help()
        sys.exit(1)
    return args

if __name__ == '__main__':
    tic = time.time()
    main()
    toc = time.time()
    print 'running time:{} s'.format(toc - tic)

