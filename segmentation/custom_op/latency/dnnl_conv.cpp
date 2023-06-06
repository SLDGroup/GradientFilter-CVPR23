/*******************************************************************************
* Copyright 2016-2020 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <chrono>
#include <csignal>

#include <assert.h>

#include <numeric>
#include <math.h>

#include "oneapi/dnnl/dnnl.hpp"

using namespace dnnl;

inline dnnl::memory::dim product(const dnnl::memory::dims &dims) {
    return std::accumulate(dims.begin(), dims.end(), (dnnl::memory::dim)1,
            std::multiplies<dnnl::memory::dim>());
}

// Read from handle, write to memory
inline void write_to_dnnl_memory(void *handle, dnnl::memory &mem) {
    dnnl::engine eng = mem.get_engine();
    size_t size = mem.get_desc().get_size();

    if (!handle) throw std::runtime_error("handle is nullptr.");

    if (eng.get_kind() == dnnl::engine::kind::cpu) {
        uint8_t *dst = static_cast<uint8_t *>(mem.get_data_handle());
        if (!dst) throw std::runtime_error("get_data_handle returned nullptr.");
        for (size_t i = 0; i < size; ++i)
            dst[i] = ((uint8_t *)handle)[i];
        return;
    }

    assert(!"not expected");
}

void simple_net(engine::kind engine_kind) {
    printf("SimpleNet\n");
    using tag = memory::format_tag;
    using dt = memory::data_type;
    auto dtype = dt::f32;

    auto eng = engine(engine_kind, 0);
    stream s(eng);

    // Vector of primitives and their execute arguments
    std::vector<primitive> net_fwd, net_bwd, net_avg_bwd;
    std::vector<std::unordered_map<int, memory>> net_fwd_args, net_bwd_args, net_avg_bwd_args;

    const int batch = 32;
    const int c_in = 512, c_out = 512;
    const int x_h = 30, x_w = 40;
    const int s_h = 1, s_w = 1;
    const int k_h = 3, k_w = 3;
    const int pad_h = k_h / 2, pad_w = k_w / 2;
    const int y_h = x_h / s_h, y_w = x_w / s_w;

    std::vector<float> net_src(batch * c_in * x_h * x_w);
    std::vector<float> net_dst(batch * c_out * y_h * y_w);

    // initializing non-zero values for src
    for (size_t i = 0; i < net_src.size(); ++i)
        net_src[i] = sinf((float)i);

    // AlexNet: conv
    // {batch, 3, 227, 227} (x) {96, 3, 11, 11} -> {batch, 96, 55, 55}
    // strides: {4, 4}

    memory::dims conv_src_tz = {batch, c_in, x_h, x_w};
    memory::dims conv_weights_tz = {c_out, c_in, k_h, k_w};
    memory::dims conv_bias_tz = {c_out};
    memory::dims conv_dst_tz = {batch, c_out, y_h, y_w};
    memory::dims conv_strides = {s_h, s_w};
    memory::dims conv_padding = {pad_h, pad_w};

    std::vector<float> conv_weights(product(conv_weights_tz));
    std::vector<float> conv_bias(product(conv_bias_tz));
    std::vector<float> net_diff_dst(batch * c_out * y_h * y_w);

    // initializing non-zero values for weights and bias
    for (size_t i = 0; i < conv_weights.size(); ++i)
        conv_weights[i] = sinf((float)i);
    for (size_t i = 0; i < conv_bias.size(); ++i)
        conv_bias[i] = sinf((float)i);
    for (size_t i = 0; i < net_diff_dst.size(); ++i)
        net_diff_dst[i] = sinf((float)i);

    // create memory for user data
    auto conv_user_src_memory
            = memory({{conv_src_tz}, dtype, tag::nchw}, eng);
    write_to_dnnl_memory(net_src.data(), conv_user_src_memory);
    auto conv_user_weights_memory
            = memory({{conv_weights_tz}, dtype, tag::oihw}, eng);
    write_to_dnnl_memory((void *)conv_weights.data(), conv_user_weights_memory);
    auto conv_user_bias_memory = memory({{conv_bias_tz}, dtype, tag::x}, eng);
    write_to_dnnl_memory(conv_bias.data(), conv_user_bias_memory);

    // create memory descriptors for convolution data w/ no specified
    // format tag(`any`)
    // tag `any` lets a primitive(convolution in this case)
    // chose the memory format preferred for best performance.
    auto start = std::chrono::high_resolution_clock::now();

    auto conv_src_md = memory::desc({conv_src_tz}, dtype, tag::any);
    auto conv_bias_md = memory::desc({conv_bias_tz}, dtype, tag::any);
    auto conv_weights_md = memory::desc({conv_weights_tz}, dtype, tag::any);
    auto conv_dst_md = memory::desc({conv_dst_tz}, dtype, tag::any);

    // create a convolution primitive descriptor
    auto conv_desc = convolution_forward::desc(prop_kind::forward,
            algorithm::convolution_direct, conv_src_md, conv_weights_md,
            conv_bias_md, conv_dst_md, conv_strides, conv_padding,
            conv_padding);
//     std::raise(SIGINT);
    auto conv_pd = convolution_forward::primitive_desc(conv_desc, eng);

    // create reorder primitives between user input and conv src if needed
    auto conv_src_memory = conv_user_src_memory;
    if (conv_pd.src_desc() != conv_user_src_memory.get_desc()) {
        conv_src_memory = memory(conv_pd.src_desc(), eng);
        net_fwd.push_back(reorder(conv_user_src_memory, conv_src_memory));
        net_fwd_args.push_back({{DNNL_ARG_FROM, conv_user_src_memory},
                {DNNL_ARG_TO, conv_src_memory}});
    }

    auto conv_weights_memory = conv_user_weights_memory;
    if (conv_pd.weights_desc() != conv_user_weights_memory.get_desc()) {
        conv_weights_memory = memory(conv_pd.weights_desc(), eng);
        net_fwd.push_back(
                reorder(conv_user_weights_memory, conv_weights_memory));
        net_fwd_args.push_back({{DNNL_ARG_FROM, conv_user_weights_memory},
                {DNNL_ARG_TO, conv_weights_memory}});
    }

    // create memory for conv dst
    auto conv_dst_memory = memory(conv_pd.dst_desc(), eng);

    // finally create a convolution primitive
    net_fwd.push_back(convolution_forward(conv_pd));
    net_fwd_args.push_back({{DNNL_ARG_SRC, conv_src_memory},
            {DNNL_ARG_WEIGHTS, conv_weights_memory},
            {DNNL_ARG_BIAS, conv_user_bias_memory},
            {DNNL_ARG_DST, conv_dst_memory}});

    //-----------------------------------------------------------------------
    //----------------- Backward Stream -------------------------------------
    // create memory for user diff dst data
    auto net_diff_memory
            = memory({{conv_dst_tz}, dtype, tag::nchw}, eng);
    // write_to_dnnl_memory(net_diff_dst.data(), net_diff_memory);

    // Backward convolution with respect to weights
    // create user format diff weights and diff bias memory
    std::vector<float> conv_user_diff_weights_buffer(product(conv_weights_tz));
    std::vector<float> conv_diff_bias_buffer(product(conv_bias_tz));
    std::vector<float> conv_user_diff_src_buffer(product(conv_src_tz));

    auto conv_user_diff_weights_memory
            = memory({{conv_weights_tz}, dtype, tag::nchw}, eng);
    // write_to_dnnl_memory(conv_user_diff_weights_buffer.data(),
            // conv_user_diff_weights_memory);
    auto conv_diff_bias_memory = memory({{conv_bias_tz}, dtype, tag::x}, eng);
    // write_to_dnnl_memory(conv_diff_bias_buffer.data(), conv_diff_bias_memory);
    auto conv_user_diff_src_memory = memory({{conv_src_tz}, dtype, tag::nchw}, eng);
    // write_to_dnnl_memory(conv_user_diff_src_buffer.data(), conv_user_diff_src_memory);

    // create memory descriptors
    auto conv_bwd_src_md = memory::desc({conv_src_tz}, dtype, tag::any);
    auto conv_diff_bias_md = memory::desc({conv_bias_tz}, dtype, tag::any);
    auto conv_diff_weights_md
            = memory::desc({conv_weights_tz}, dtype, tag::any);
    auto conv_diff_dst_md = memory::desc({conv_dst_tz}, dtype, tag::any);
    auto conv_bwd_weight_md = memory::desc({conv_weights_tz}, dtype, tag::any);
    auto conv_diff_src_md = memory::desc({conv_src_tz}, dtype, tag::any);

    // create backward convolution primitive descriptor
    auto conv_bwd_weights_desc
            = convolution_backward_weights::desc(algorithm::convolution_direct,
                    conv_bwd_src_md, conv_diff_weights_md, conv_diff_bias_md,
                    conv_diff_dst_md, conv_strides, conv_padding, conv_padding);
    auto conv_bwd_weights_pd = convolution_backward_weights::primitive_desc(
            conv_bwd_weights_desc, eng, conv_pd);

    auto conv_bwd_data_desc 
            = convolution_backward_data::desc(algorithm::convolution_direct, 
                    conv_diff_src_md, conv_bwd_weight_md, 
                    conv_diff_dst_md, conv_strides, conv_padding, conv_padding
            );
    auto conv_bwd_data_pd = convolution_backward_data::primitive_desc(conv_bwd_data_desc, eng, conv_pd);

    // for best performance convolution backward might chose
    // different memory format for src and diff_dst
    // than the memory formats preferred by forward convolution
    // for src and dst respectively
    // create reorder primitives for src from forward convolution to the
    // format chosen by backward convolution
    auto conv_bwd_src_memory = conv_src_memory;
    if (conv_bwd_weights_pd.src_desc() != conv_src_memory.get_desc()) {
        conv_bwd_src_memory = memory(conv_bwd_weights_pd.src_desc(), eng);
        net_bwd.push_back(reorder(conv_src_memory, conv_bwd_src_memory));
        net_bwd_args.push_back({{DNNL_ARG_FROM, conv_src_memory},
                {DNNL_ARG_TO, conv_bwd_src_memory}});
    }

    // create reorder primitives for diff_dst between diff_src from relu_bwd
    // and format preferred by conv_diff_weights
    auto conv_diff_dst_memory = net_diff_memory;
    if (conv_bwd_weights_pd.diff_dst_desc()
            != net_diff_memory.get_desc()) {
        conv_diff_dst_memory = memory(conv_bwd_weights_pd.diff_dst_desc(), eng);
        net_bwd.push_back(reorder(net_diff_memory, conv_diff_dst_memory));
        net_bwd_args.push_back({{DNNL_ARG_FROM, net_diff_memory},
                {DNNL_ARG_TO, conv_diff_dst_memory}});
    }

    // create backward convolution primitive
    net_bwd.push_back(convolution_backward_weights(conv_bwd_weights_pd));
    net_bwd_args.push_back({{DNNL_ARG_SRC, conv_bwd_src_memory},
            {DNNL_ARG_DIFF_DST, conv_diff_dst_memory},
            // delay putting DIFF_WEIGHTS until reorder (if needed)
            {DNNL_ARG_DIFF_BIAS, conv_diff_bias_memory}});

    // create reorder primitives between conv diff weights and user diff weights
    // if needed
    auto conv_diff_weights_memory = conv_user_diff_weights_memory;
    if (conv_bwd_weights_pd.diff_weights_desc()
            != conv_user_diff_weights_memory.get_desc()) {
        conv_diff_weights_memory
                = memory(conv_bwd_weights_pd.diff_weights_desc(), eng);
        net_bwd_args.back().insert(
                {DNNL_ARG_DIFF_WEIGHTS, conv_diff_weights_memory});

        net_bwd.push_back(reorder(
                conv_diff_weights_memory, conv_user_diff_weights_memory));
        net_bwd_args.push_back({{DNNL_ARG_FROM, conv_diff_weights_memory},
                {DNNL_ARG_TO, conv_user_diff_weights_memory}});
    } else {
        net_bwd_args.back().insert(
                {DNNL_ARG_DIFF_WEIGHTS, conv_diff_weights_memory});
    }

    auto conv_bwd_weights_memory = conv_weights_memory;
    if (conv_bwd_data_pd.weights_desc() != conv_weights_memory.get_desc()) {
        conv_bwd_weights_memory = memory(conv_bwd_data_pd.weights_desc(), eng);
        net_bwd.push_back(reorder(conv_weights_memory, conv_bwd_weights_memory));
        net_bwd_args.push_back({{DNNL_ARG_FROM, conv_weights_memory},
                {DNNL_ARG_TO, conv_bwd_weights_memory}});
    }

    net_bwd.push_back(convolution_backward_data(conv_bwd_data_pd));
    net_bwd_args.push_back({{DNNL_ARG_DIFF_DST, conv_diff_dst_memory},
            // delay putting DIFF_WEIGHTS until reorder (if needed)
            {DNNL_ARG_WEIGHTS, conv_bwd_weights_memory}});


    auto conv_diff_src_memory = conv_user_diff_src_memory;
    if (conv_bwd_data_pd.diff_src_desc()
            != conv_user_diff_src_memory.get_desc()) {
        conv_diff_src_memory
                = memory(conv_bwd_data_pd.diff_src_desc(), eng);
        net_bwd_args.back().insert(
                {DNNL_ARG_DIFF_SRC, conv_diff_src_memory});
        // std::raise(SIGINT);
        net_bwd.push_back(reorder(
                conv_diff_src_memory, conv_user_diff_src_memory));
        net_bwd_args.push_back({{DNNL_ARG_FROM, conv_diff_src_memory},
                {DNNL_ARG_TO, conv_user_diff_src_memory}});
    } else {
        net_bwd_args.back().insert(
                {DNNL_ARG_DIFF_SRC, conv_diff_src_memory});
    }

    // New Backward
    int kern_size = 4;
    int p_h = ceil(y_h * 1.0 / kern_size), p_w = ceil(y_w * 1.0 / kern_size);
    memory::dims avg_pool_kern = {kern_size, kern_size};
    memory::dims avg_pool_pad = {p_h * kern_size - y_h, p_w * kern_size - y_w};
    memory::dims avg_dst_grad_tz = {batch, c_out, p_h, p_w};

    auto avg_dst_grad_memory = memory({{avg_dst_grad_tz}, dtype, tag::nChw16c}, eng);
    auto avg_desc = pooling_forward::desc(
        prop_kind::forward, algorithm::pooling_avg_exclude_padding, 
        conv_diff_dst_memory.get_desc(), avg_dst_grad_memory.get_desc(), 
        avg_pool_kern, avg_pool_kern, {0, 0}, avg_pool_pad);
    auto avg_pd = pooling_forward::primitive_desc(avg_desc, eng);
    net_avg_bwd.push_back(pooling_forward(avg_pd));
    net_avg_bwd_args.push_back({{DNNL_ARG_SRC, conv_diff_dst_memory}, {DNNL_ARG_DST, avg_dst_grad_memory}});

    memory::dims weight_sum_tz = {c_in, c_out, 1, 1};
    auto weight_memory = memory({{conv_weights_tz}, dtype, tag::oihw}, eng);
    auto weight_memory_t_md = memory::desc({{c_out, c_in, 3, 3}}, dtype, tag::iohw);
    auto transpose_desc = reorder::primitive_desc(eng, conv_weights_memory.get_desc(), eng, weight_memory_t_md);
    net_avg_bwd.push_back(reorder(transpose_desc));
    net_avg_bwd_args.push_back({{DNNL_ARG_FROM, conv_weights_memory}, {DNNL_ARG_DST, weight_memory}});

    auto weight_sum_memory = memory({{weight_sum_tz}, dtype, tag::oihw}, eng);
    auto weight_sum_desc = reduction::desc(algorithm::reduction_sum, 
        weight_memory.get_desc(), weight_sum_memory.get_desc(), 0, 0);
    auto weight_sum_pd = reduction::primitive_desc(weight_sum_desc, eng);
    net_avg_bwd.push_back(reduction(weight_sum_pd));
    net_avg_bwd_args.push_back({{DNNL_ARG_SRC, conv_weights_memory}, {DNNL_ARG_DST, weight_sum_memory}});

    auto cc_desc = convolution_forward::desc(prop_kind::forward, 
        algorithm::convolution_direct, 
        {{avg_dst_grad_tz}, dtype, tag::any},
        {{{c_in, c_out, 1, 1}}, dtype, tag::any},
        {{avg_dst_grad_tz}, dtype, tag::any},
        {1, 1},
        {0, 0},
        {0, 0}
        );
    auto cc_pd = convolution_forward::primitive_desc(cc_desc, eng);

    auto cc_gy_memory = avg_dst_grad_memory;
    if (cc_pd.src_desc() != avg_dst_grad_memory.get_desc()) {
        cc_gy_memory = memory(cc_pd.src_desc(), eng);
        net_avg_bwd.push_back(reorder(avg_dst_grad_memory, cc_gy_memory));
        net_avg_bwd_args.push_back({{DNNL_ARG_FROM, avg_dst_grad_memory},
                {DNNL_ARG_TO, cc_gy_memory}});
    }
    auto cc_w_memory = weight_sum_memory;
    if (cc_pd.weights_desc() != cc_w_memory.get_desc()) {
        cc_w_memory = memory(cc_pd.weights_desc(), eng);
        net_avg_bwd.push_back(reorder(weight_sum_memory, cc_w_memory));
        net_avg_bwd_args.push_back({{DNNL_ARG_FROM, weight_sum_memory},
                {DNNL_ARG_TO, cc_w_memory}});
    }

    auto gx_memory = memory(conv_pd.dst_desc(), eng);
    net_avg_bwd.push_back(convolution_forward(cc_pd));
    net_avg_bwd_args.push_back({{DNNL_ARG_SRC, cc_gy_memory}, 
        {DNNL_ARG_WEIGHTS, cc_w_memory}, 
        {DNNL_ARG_DST, gx_memory}
    });


    // New Backward Weight
    int x_kern_size_h = kern_size * s_h, x_kern_size_w = kern_size * s_w;
    int x_ph = ceil(x_h * 1.0 / x_kern_size_h), x_pw = ceil(x_w * 1.0 / x_kern_size_w);
    assert(x_ph == p_h && x_pw == p_w && "Dimension Wrong");

    memory::dims avg_src_tz = {batch, c_in, p_h, p_w};

    auto avg_src_memory = memory({{avg_src_tz}, dtype, tag::nChw16c}, eng);
    auto avg_src_desc = pooling_forward::desc(
        prop_kind::forward, algorithm::pooling_avg_exclude_padding, 
        conv_src_memory.get_desc(), avg_src_memory.get_desc(), 
        {x_kern_size_h, x_kern_size_w}, {x_kern_size_h, x_kern_size_w}, {0, 0}, 
        {x_kern_size_h * p_h - x_h, x_kern_size_w * p_w - x_w});
    auto avg_src_pd = pooling_forward::primitive_desc(avg_src_desc, eng);
    net_avg_bwd.push_back(pooling_forward(avg_src_pd));
    net_avg_bwd_args.push_back({{DNNL_ARG_SRC, conv_src_memory}, {DNNL_ARG_DST, avg_src_memory}});

    auto avg_src_t_md = memory::desc({{{batch, c_in, p_h, p_w}}, dtype, tag::bacd});
    auto avg_src_t_memory = memory({{{c_in, batch, p_h, p_w}}, dtype, tag::abcd}, eng);
    auto avg_src_t_desc = reorder::primitive_desc(eng, avg_src_memory.get_desc(), eng, avg_src_t_md);
    net_avg_bwd.push_back(reorder(avg_src_t_desc));
    net_avg_bwd_args.push_back({{DNNL_ARG_SRC, avg_src_memory}, {DNNL_ARG_DST, avg_src_t_memory}});
    
    auto avg_dst_grad_t_md = memory::desc({{{batch, c_out, p_h, p_w}}, dtype, tag::bacd});
    auto avg_dst_grad_t_memory = memory({{{c_out, batch, p_h, p_w}}, dtype, tag::abcd}, eng);
    auto avg_dst_grad_t_desc = reorder::primitive_desc(eng, avg_dst_grad_memory.get_desc(), eng, avg_dst_grad_t_md);
    net_avg_bwd.push_back(reorder(avg_dst_grad_t_desc));
    net_avg_bwd_args.push_back({{DNNL_ARG_SRC, avg_dst_grad_memory}, {DNNL_ARG_DST, avg_dst_grad_t_memory}});


    auto weight_grad_memory = memory({{{c_out, c_in}}, dtype, tag::oi}, eng);
    auto inner_prod_desc = inner_product_forward::desc(prop_kind::forward, 
        avg_dst_grad_t_memory.get_desc(), avg_src_t_memory.get_desc(), weight_grad_memory.get_desc());
    auto inner_prod_pd = inner_product_forward::primitive_desc(inner_prod_desc, eng);
    net_avg_bwd.push_back(inner_prod_pd);
    net_avg_bwd_args.push_back({{DNNL_ARG_SRC, avg_dst_grad_t_memory}, 
        {DNNL_ARG_WEIGHTS, avg_src_t_memory}, 
        {DNNL_ARG_DST, weight_grad_memory}});


    // didn't we forget anything?
    assert(net_fwd.size() == net_fwd_args.size() && "something is missing");
    assert(net_bwd.size() == net_bwd_args.size() && "something is missing");
    assert(net_avg_bwd.size() == net_avg_bwd_args.size() && "something is missing");
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    printf("Operator Create Time: %.3f ms\n", duration.count() / 1000.0);

    int n_iter = 50; // number of iterations for training
    // execute
    start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < n_iter; iter++) {
        for (size_t i = 0; i < net_bwd.size(); ++i)
            net_bwd.at(i).execute(s, net_bwd_args.at(i));
        s.wait();
    }
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    printf("Vanilla BP Time: %.3f [%.3f] ms\n", duration.count()/1000.0, duration.count() / 1000.0 / n_iter);

    start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < n_iter; iter++) {
        for (size_t i = 0; i < net_avg_bwd.size(); ++i)
            net_avg_bwd.at(i).execute(s, net_avg_bwd_args.at(i));
        s.wait();
    }
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    printf("New BP Time: %.3f [%.3f] ms\n", duration.count() / 1000.0, duration.count() / 1000.0 / n_iter);
}

int main(int argc, char **argv) {
    for (int i = 0; i < 1; i++) {
        printf("Round %d\n", i);
        simple_net(dnnl::engine::kind::cpu);
    }
    return 0;
}
