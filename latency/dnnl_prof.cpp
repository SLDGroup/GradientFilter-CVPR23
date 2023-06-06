#include <assert.h>
#include <math.h>

#include <chrono>
#include <csignal>
#include <fstream>
#include <numeric>
#include <thread>

#include "oneapi/dnnl/dnnl.hpp"

using namespace dnnl;
using dt = memory::data_type;
using tag = memory::format_tag;

#define duration_ms(start, end)                                            \
    std::chrono::duration_cast<std::chrono::microseconds>((end) - (start)) \
            .count() /                                                     \
        1000.0
#define to_us(t) std::chrono::duration_cast<std::chrono::microseconds>(t)

class Conv {
   public:
    memory::dims x_shape;
    memory::dims w_shape;
    memory::dims b_shape;
    memory::dims y_shape;
    memory w, b;
    memory bwd_w, bwd_x;
    memory gw, gb;
    convolution_forward::primitive_desc conv_fwd_pd;
    convolution_backward_data::primitive_desc conv_bwd_x_pd;
    convolution_backward_weights::primitive_desc conv_bwd_w_pd;
    convolution_forward conv_fwd;
    convolution_backward_data conv_bwd_x;
    convolution_backward_weights conv_bwd_w;
    int conv_stride, kern_size;
    bool train_w, train_x;
    Conv(engine &eng, memory::dims x_shape, int c_out, bool train_w,
         bool train_x, int conv_stride, int kern_size)
        : x_shape(x_shape),
          train_w(train_w),
          train_x(train_x),
          conv_stride(conv_stride),
          kern_size(kern_size) {
        w_shape = {c_out, x_shape[1], kern_size, kern_size};
        b_shape = {c_out};
        y_shape = {x_shape[0], c_out, x_shape[2] / conv_stride,
                   x_shape[3] / conv_stride};
        auto x_md = memory::desc({x_shape}, dt::f32, tag::any);
        auto w_md = memory::desc({w_shape}, dt::f32, tag::any);
        auto b_md = memory::desc({b_shape}, dt::f32, tag::any);
        auto y_md = memory::desc({y_shape}, dt::f32, tag::any);
        memory::dims padding = {kern_size / 2, kern_size / 2};
        memory::dims stride = {conv_stride, conv_stride};
        if (train_w || train_x) {
            auto conv_fwd_desc = convolution_forward::desc(
                prop_kind::forward, algorithm::convolution_direct, x_md, w_md,
                b_md, y_md, stride, padding, padding);
            conv_fwd_pd =
                convolution_forward::primitive_desc(conv_fwd_desc, eng);
            conv_fwd = convolution_forward(conv_fwd_pd);
            if (train_w) {
                auto conv_bwd_w_desc = convolution_backward_weights::desc(
                    algorithm::convolution_direct, x_md, w_md, b_md, y_md,
                    {conv_stride, conv_stride}, padding, padding);
                conv_bwd_w_pd = convolution_backward_weights::primitive_desc(
                    conv_bwd_w_desc, eng, conv_fwd_pd);
                conv_bwd_w = convolution_backward_weights(conv_bwd_w_pd);
                gw = memory(conv_bwd_w_pd.diff_weights_desc(), eng);
                gb = memory(conv_bwd_w_pd.diff_bias_desc(), eng);
                bwd_x = memory(conv_bwd_w_pd.src_desc(), eng);
            }
            if (train_x) {
                auto conv_bwd_data_desc = convolution_backward_data::desc(
                    algorithm::convolution_direct, x_md, w_md, y_md,
                    {conv_stride, conv_stride}, padding, padding);
                conv_bwd_x_pd = convolution_backward_data::primitive_desc(
                    conv_bwd_data_desc, eng, conv_fwd_pd);
                conv_bwd_x = convolution_backward_data(conv_bwd_x_pd);
            }
        } else {
            auto conv_fwd_desc = convolution_forward::desc(
                prop_kind::forward_inference, algorithm::convolution_direct,
                x_md, w_md, b_md, y_md, {conv_stride, conv_stride}, padding,
                padding);
            conv_fwd_pd =
                convolution_forward::primitive_desc(conv_fwd_desc, eng);
            conv_fwd = convolution_forward(conv_fwd_pd);
        }
        w = memory(conv_fwd_pd.weights_desc(), eng);
        b = memory(conv_fwd_pd.bias_desc(), eng);
        bwd_w = w;
        if (w.get_desc() != conv_bwd_x_pd.weights_desc()) {
            bwd_w = memory(conv_bwd_x_pd.weights_desc(), eng);
        }
    }
    void forward_prim(memory &x, memory &y, std::vector<primitive> &prim,
                      std::vector<std::unordered_map<int, memory>> &args) {
        assert(x.get_desc() == conv_fwd_pd.src_desc());
        assert(y.get_desc() == conv_fwd_pd.dst_desc());
        prim.push_back(conv_fwd);
        args.push_back({{DNNL_ARG_SRC, x},
                        {DNNL_ARG_WEIGHTS, w},
                        {DNNL_ARG_BIAS, b},
                        {DNNL_ARG_DST, y}});
    }
    void forward_save_context_prim(
        memory &x, std::vector<primitive> &prim,
        std::vector<std::unordered_map<int, memory>> &args) {
        if (train_w) {
            prim.push_back(reorder(x, bwd_x));
            args.push_back({{DNNL_ARG_FROM, x}, {DNNL_ARG_TO, bwd_x}});
        }
        if (train_x && w.get_desc() != conv_bwd_x_pd.weights_desc()) {
            prim.push_back(reorder(w, bwd_w));
            args.push_back({{DNNL_ARG_FROM, w}, {DNNL_ARG_TO, bwd_w}});
        }
    }
    void backward_data_prim(
        memory &gy, memory &gx, std::vector<primitive> &prim,
        std::vector<std::unordered_map<int, memory>> &args) {
        assert(gy.get_desc() == conv_bwd_x_pd.diff_dst_desc());
        assert(gx.get_desc() == conv_bwd_x_pd.diff_src_desc());
        prim.push_back(conv_bwd_x);
        args.push_back({{DNNL_ARG_DIFF_DST, gy},
                        {DNNL_ARG_DIFF_SRC, gx},
                        {DNNL_ARG_WEIGHTS, bwd_w}});
    }
    void backward_weights_prim(
        memory &gy, std::vector<primitive> &prim,
        std::vector<std::unordered_map<int, memory>> &args) {
        assert(gy.get_desc() == conv_bwd_w_pd.diff_dst_desc());
        prim.push_back(conv_bwd_w);
        args.push_back({{DNNL_ARG_SRC, bwd_x},
                        {DNNL_ARG_DIFF_DST, gy},
                        {DNNL_ARG_DIFF_BIAS, gb},
                        {DNNL_ARG_DIFF_WEIGHTS, gw}});
    }
};

class ConvOurs {
   public:
    memory::dims x_shape;
    memory::dims w_shape;
    memory::dims b_shape;
    memory::dims y_shape;
    memory w, b;
    // Context
    memory bwd_w, bwd_x;
    // Runtime
    memory x_reorder, w_reorder;
    memory avg_gy, avg_gy_reorder, x_reduced, w_reduced;
    // Output
    memory gw, gb;
    // Forward
    convolution_forward::primitive_desc conv_fwd_pd;
    convolution_forward conv_fwd;
    // Backward Shared
    pooling_forward::primitive_desc avg_gy_pd;
    pooling_forward avg_gy_op;
    // Backward Data
    reduction::primitive_desc weight_sum_pd;
    reduction weight_sum_op;
    convolution_backward_data::primitive_desc bwd_x_conv_pd;
    convolution_backward_data bwd_x_conv_op;
    // Backward Weight
    pooling_forward::primitive_desc reduce_x_pd;
    pooling_forward reduce_x_op;
    convolution_backward_weights::primitive_desc bwd_w_conv_pd;
    convolution_backward_weights bwd_w_conv_op;
    int conv_stride, kern_size;
    bool train_w, train_x;
    int avg_rad;
    ConvOurs(engine &eng, memory::dims x_shape, int c_out, bool train_w,
             bool train_x, int conv_stride, int kern_size, int avg_rad)
        : x_shape(x_shape),
          train_w(train_w),
          train_x(train_x),
          conv_stride(conv_stride),
          kern_size(kern_size),
          avg_rad(avg_rad) {
        int batch = x_shape[0], c_in = x_shape[1], x_h = x_shape[2],
            x_w = x_shape[3];
        int y_h = x_h / conv_stride, y_w = x_w / conv_stride;
        w_shape = {c_out, c_in, kern_size, kern_size};
        b_shape = {c_out};
        y_shape = {batch, c_out, y_h, y_w};
        auto x_md = memory::desc({x_shape}, dt::f32, tag::any);
        auto w_md = memory::desc({w_shape}, dt::f32, tag::any);
        auto b_md = memory::desc({b_shape}, dt::f32, tag::any);
        auto y_md = memory::desc({y_shape}, dt::f32, tag::any);
        memory::dims padding = {kern_size / 2, kern_size / 2};
        memory::dims stride = {conv_stride, conv_stride};
        if (train_w || train_x) {
            auto conv_fwd_desc = convolution_forward::desc(
                prop_kind::forward, algorithm::convolution_direct, x_md, w_md,
                b_md, y_md, stride, padding, padding);
            conv_fwd_pd =
                convolution_forward::primitive_desc(conv_fwd_desc, eng);
            conv_fwd = convolution_forward(conv_fwd_pd);
            w = memory(conv_fwd_pd.weights_desc(), eng);
            b = memory(conv_fwd_pd.bias_desc(), eng);

            int p_h = ceil(y_shape[2] * 1.0 / avg_rad),
                p_w = ceil(y_shape[3] * 1.0 / avg_rad);
            memory::dims avg_gy_pad = {p_h * avg_rad - y_shape[2],
                                       p_w * avg_rad - y_shape[3]};
            memory::dims avg_gy_shape = {batch, c_out, p_h, p_w};
            if (train_w) {
                auto bwd_w_conv_desc = convolution_backward_weights::desc(
                    algorithm::convolution_direct,
                    {{batch, c_in, p_h, p_w}, dt::f32, tag::any},
                    {{c_out, c_in, 1, 1}, dt::f32, tag::any},
                    {{c_out}, dt::f32, tag::any},
                    {{batch, c_out, p_h, p_w}, dt::f32, tag::any}, {1, 1},
                    {0, 0}, {0, 0});
                bwd_w_conv_pd = convolution_backward_weights::primitive_desc(
                    bwd_w_conv_desc, eng, conv_fwd_pd);
                bwd_w_conv_op = convolution_backward_weights(bwd_w_conv_pd);
                bwd_x = memory(bwd_w_conv_pd.src_desc(), eng);
                int x_avg_rad = avg_rad * conv_stride;
                auto reduce_x_desc = pooling_forward::desc(
                    prop_kind::forward_inference, algorithm::pooling_avg,
                    {{batch, c_in, x_h, x_w}, dt::f32, tag::nchw},
                    {{batch, c_in, p_h, p_w}, dt::f32, tag::any},
                    {x_avg_rad, x_avg_rad}, {x_avg_rad, x_avg_rad}, {0, 0},
                    {x_avg_rad * p_h - x_h, x_avg_rad * p_w - x_w});
                reduce_x_pd =
                    pooling_forward::primitive_desc(reduce_x_desc, eng);
                reduce_x_op = pooling_forward(reduce_x_pd);
                if (conv_fwd_pd.src_desc() != reduce_x_pd.src_desc()) {
                    x_reorder = memory(reduce_x_pd.src_desc(), eng);
                }
                x_reduced = memory(reduce_x_pd.dst_desc(), eng);
                gw = memory(bwd_w_conv_pd.diff_weights_desc(), eng);
                gb = memory(bwd_w_conv_pd.diff_bias_desc(), eng);
                if (x_reduced.get_desc() != bwd_w_conv_pd.src_desc()) {
                    bwd_x = memory(bwd_w_conv_pd.src_desc(), eng);
                } else {
                    bwd_x = x_reduced;
                }
            }
            if (train_x) {
                auto bwd_x_conv_desc = convolution_backward_data::desc(
                    algorithm::convolution_direct,
                    {{batch, c_in, p_h, p_w}, dt::f32, tag::any},
                    {{c_out, c_in, 1, 1}, dt::f32, tag::any},
                    {{batch, c_out, p_h, p_w}, dt::f32, tag::any}, {1, 1},
                    {0, 0}, {0, 0});
                bwd_x_conv_pd = convolution_backward_data::primitive_desc(
                    bwd_x_conv_desc, eng, conv_fwd_pd);
                bwd_x_conv_op = convolution_backward_data(bwd_x_conv_pd);
                bwd_w = memory(bwd_x_conv_pd.weights_desc(), eng);
                if (kern_size != 1) {
                    auto weight_sum_desc = reduction::desc(
                        algorithm::reduction_sum,
                        {{c_out, c_in, kern_size, kern_size},
                         dt::f32,
                         tag::oihw},
                        {{c_out, c_in, 1, 1}, dt::f32, tag::any}, 0, 0);
                    weight_sum_pd =
                        reduction::primitive_desc(weight_sum_desc, eng);
                    weight_sum_op = reduction(weight_sum_pd);
                    if (weight_sum_pd.src_desc() !=
                        conv_fwd_pd.weights_desc()) {
                        w_reorder = memory(weight_sum_pd.src_desc(), eng);
                    } else {
                        w_reorder = w;
                    }
                    w_reduced = memory(weight_sum_pd.dst_desc(), eng);
                } else {
                    w_reduced = w;
                }
                if (w_reduced.get_desc() != bwd_x_conv_pd.weights_desc()) {
                    bwd_w = memory(bwd_x_conv_pd.weights_desc(), eng);
                } else {
                    bwd_w = w_reduced;
                }
            }

            auto avg_gy_desc = pooling_forward::desc(
                prop_kind::forward_inference, algorithm::pooling_avg,
                {{batch, c_out, y_h, y_w}, dt::f32, tag::nchw},
                {{batch, c_out, p_h, p_w}, dt::f32, tag::any},
                {avg_rad, avg_rad}, {avg_rad, avg_rad}, {0, 0}, avg_gy_pad);
            avg_gy_pd = pooling_forward::primitive_desc(avg_gy_desc, eng);
            avg_gy_op = pooling_forward(avg_gy_pd);
            avg_gy = memory(avg_gy_pd.dst_desc(), eng);
            if (avg_gy_pd.dst_desc() != bwd_x_conv_pd.diff_dst_desc()) {
                avg_gy_reorder = memory(bwd_x_conv_pd.diff_dst_desc(), eng);
            } else {
                avg_gy_reorder = avg_gy;
            }
        } else {
            auto conv_fwd_desc = convolution_forward::desc(
                prop_kind::forward_inference, algorithm::convolution_direct,
                x_md, w_md, b_md, y_md, {conv_stride, conv_stride}, {1, 1},
                {1, 1});
            conv_fwd_pd =
                convolution_forward::primitive_desc(conv_fwd_desc, eng);
            conv_fwd = convolution_forward(conv_fwd_pd);
            w = memory(conv_fwd_pd.weights_desc(), eng);
            b = memory(conv_fwd_pd.bias_desc(), eng);
        }
    }
    void forward_prim(memory &x, memory &y, std::vector<primitive> &prim,
                      std::vector<std::unordered_map<int, memory>> &args) {
        assert(x.get_desc() == conv_fwd_pd.src_desc());
        assert(y.get_desc() == conv_fwd_pd.dst_desc());
        prim.push_back(conv_fwd);
        args.push_back({{DNNL_ARG_SRC, x},
                        {DNNL_ARG_WEIGHTS, w},
                        {DNNL_ARG_BIAS, b},
                        {DNNL_ARG_DST, y}});
    }
    void forward_save_context_prim(
        memory &x, std::vector<primitive> &prim,
        std::vector<std::unordered_map<int, memory>> &args) {
        if (train_w) {
            if (x.get_desc() != reduce_x_pd.src_desc()) {
                prim.push_back(reorder(x, x_reorder));
                args.push_back({{DNNL_ARG_FROM, x}, {DNNL_ARG_TO, x_reorder}});
            } else {
                x_reorder = x;
            }
            prim.push_back(reduce_x_op);
            args.push_back(
                {{DNNL_ARG_SRC, x_reorder}, {DNNL_ARG_DST, x_reduced}});
            if (x_reduced.get_desc() != bwd_w_conv_pd.src_desc()) {
                prim.push_back(reorder(x_reduced, bwd_x));
                args.push_back(
                    {{DNNL_ARG_FROM, x_reduced}, {DNNL_ARG_TO, bwd_x}});
            }
        }
        if (train_x) {
            if (kern_size != 1) {
                if (w.get_desc() != weight_sum_pd.src_desc()) {
                    prim.push_back(reorder(w, w_reorder));
                    args.push_back(
                        {{DNNL_ARG_FROM, w}, {DNNL_ARG_TO, w_reorder}});
                } else {
                    w_reorder = w;
                }
                prim.push_back(weight_sum_op);
                args.push_back(
                    {{DNNL_ARG_SRC, w_reorder}, {DNNL_ARG_DST, w_reduced}});
            }
            if (w_reduced.get_desc() != bwd_x_conv_pd.weights_desc()) {
                prim.push_back(reorder(w_reduced, bwd_w));
                args.push_back(
                    {{DNNL_ARG_FROM, w_reduced}, {DNNL_ARG_TO, bwd_w}});
            }
        }
    }
    void backward_filter_prim(
        memory &gy, std::vector<primitive> &prim,
        std::vector<std::unordered_map<int, memory>> &args) {
        prim.push_back(avg_gy_op);
        args.push_back({{DNNL_ARG_SRC, gy}, {DNNL_ARG_DST, avg_gy}});
        if (avg_gy.get_desc() != bwd_x_conv_pd.diff_dst_desc()) {
            prim.push_back(reorder(avg_gy, avg_gy_reorder));
            args.push_back(
                {{DNNL_ARG_FROM, avg_gy}, {DNNL_ARG_TO, avg_gy_reorder}});
        }
    }
    void backward_data_prim(
        memory &gy, memory &gx, std::vector<primitive> &prim,
        std::vector<std::unordered_map<int, memory>> &args) {
        assert(gy.get_desc() == avg_gy_pd.src_desc());
        prim.push_back(bwd_x_conv_op);
        args.push_back({{DNNL_ARG_DIFF_DST, avg_gy_reorder},
                        {DNNL_ARG_DIFF_SRC, gx},
                        {DNNL_ARG_WEIGHTS, bwd_w}});
    }
    void backward_weights_prim(
        memory &gy, std::vector<primitive> &prim,
        std::vector<std::unordered_map<int, memory>> &args) {
        assert(gy.get_desc() == avg_gy_pd.src_desc());
        assert(avg_gy_reorder.get_desc() == bwd_w_conv_pd.diff_dst_desc());
        prim.push_back(bwd_w_conv_op);
        args.push_back({{DNNL_ARG_SRC, bwd_x},
                        {DNNL_ARG_DIFF_DST, avg_gy_reorder},
                        {DNNL_ARG_DIFF_BIAS, gb},
                        {DNNL_ARG_DIFF_WEIGHTS, gw}});
    }
};

void vanilla_conv() {
    auto eng = engine(engine::kind::cpu, 0);
    stream s(eng);
    auto c = Conv(eng, {4, 3, 224, 224}, 16, true, true, 1, 3);
    std::vector<primitive> fwd, fwd_context, bwd_data, bwd_weights;
    std::vector<std::unordered_map<int, memory>> fwd_args, fwd_context_args,
        bwd_data_args, bwd_weights_args;
    auto input = memory(c.conv_fwd_pd.src_desc(), eng);
    auto output = memory(c.conv_fwd_pd.dst_desc(), eng);
    auto grad_output = memory(c.conv_bwd_x_pd.diff_dst_desc(), eng);
    auto grad_input = memory(c.conv_bwd_x_pd.diff_src_desc(), eng);
    c.forward_prim(input, output, fwd, fwd_args);
    c.forward_save_context_prim(input, fwd_context, fwd_context_args);
    c.backward_data_prim(grad_output, grad_input, bwd_data, bwd_data_args);
    c.backward_weights_prim(grad_output, bwd_weights, bwd_weights_args);
    printf("Forward\n");
    for (size_t i = 0; i < fwd.size(); i++) {
        fwd.at(i).execute(s, fwd_args.at(i));
    }
    printf("Forward Context\n");
    for (size_t i = 0; i < fwd_context.size(); i++) {
        fwd_context.at(i).execute(s, fwd_context_args.at(i));
    }
    printf("Backward Data\n");
    for (size_t i = 0; i < bwd_data.size(); i++) {
        bwd_data.at(i).execute(s, bwd_data_args.at(i));
    }
    printf("Backward Weights\n");
    for (size_t i = 0; i < bwd_weights.size(); i++) {
        bwd_weights.at(i).execute(s, bwd_weights_args.at(i));
    }
}

void new_conv() {
    auto eng = engine(engine::kind::cpu, 0);
    stream s(eng);
    auto c = ConvOurs(eng, {4, 3, 224, 224}, 16, true, true, 1, 3, 4);
    std::vector<primitive> fwd, fwd_context, bwd_filter, bwd_data, bwd_weights;
    std::vector<std::unordered_map<int, memory>> fwd_args, fwd_context_args,
        bwd_filter_args, bwd_data_args, bwd_weights_args;
    auto input = memory(c.conv_fwd_pd.src_desc(), eng);
    auto output = memory(c.conv_fwd_pd.dst_desc(), eng);
    auto grad_output = memory(c.avg_gy_pd.src_desc(), eng);
    auto grad_input = memory(c.bwd_x_conv_pd.diff_src_desc(), eng);
    c.forward_prim(input, output, fwd, fwd_args);
    c.forward_save_context_prim(input, fwd_context, fwd_context_args);
    c.backward_filter_prim(grad_output, bwd_filter, bwd_filter_args);
    c.backward_data_prim(grad_output, grad_input, bwd_data, bwd_data_args);
    c.backward_weights_prim(grad_output, bwd_weights, bwd_weights_args);
    printf("Forward\n");
    for (size_t i = 0; i < fwd.size(); i++) {
        fwd.at(i).execute(s, fwd_args.at(i));
    }
    printf("Forward Context\n");
    for (size_t i = 0; i < fwd_context.size(); i++) {
        fwd_context.at(i).execute(s, fwd_context_args.at(i));
    }
    printf("Backward Filter\n");
    for (size_t i = 0; i < bwd_filter.size(); i++) {
        bwd_filter.at(i).execute(s, bwd_filter_args.at(i));
    }
    printf("Backward Data\n");
    for (size_t i = 0; i < bwd_data.size(); i++) {
        bwd_data.at(i).execute(s, bwd_data_args.at(i));
    }
    printf("Backward Weights\n");
    for (size_t i = 0; i < bwd_weights.size(); i++) {
        bwd_weights.at(i).execute(s, bwd_weights_args.at(i));
    }
}

struct ProfileResult {
    std::chrono::microseconds forward_start_timestamp;
    std::chrono::microseconds forward_end_timestamp;
    std::chrono::microseconds backward_start_timestamp;
    std::chrono::microseconds backward_end_timestamp;
    float forward_time;
    float backward_time;
    size_t context_size;
};

void profile_vanilla_conv(ProfileResult &result, engine &eng, stream &s,
                          int repeat, int warmup, int batch, int x_h, int x_w,
                          int c_in, int c_out, int stride, int kern_size) {
    std::chrono::time_point<std::chrono::system_clock> start, end;

    auto c = Conv(eng, {batch, c_in, x_h, x_w}, c_out, true, true, stride,
                  kern_size);
    std::vector<primitive> fwd, fwd_context, bwd_data, bwd_weights;
    std::vector<std::unordered_map<int, memory>> fwd_args, fwd_context_args,
        bwd_data_args, bwd_weights_args;
    auto input = memory(c.conv_fwd_pd.src_desc(), eng);
    auto output = memory(c.conv_fwd_pd.dst_desc(), eng);
    auto grad_output = memory(c.conv_bwd_x_pd.diff_dst_desc(), eng);
    auto grad_input = memory(c.conv_bwd_x_pd.diff_src_desc(), eng);
    c.forward_prim(input, output, fwd, fwd_args);
    c.forward_save_context_prim(input, fwd_context, fwd_context_args);
    c.backward_data_prim(grad_output, grad_input, bwd_data, bwd_data_args);
    c.backward_weights_prim(grad_output, bwd_weights, bwd_weights_args);

    for (int warmup_idx = 0; warmup_idx < warmup; warmup_idx++) {
        for (size_t i = 0; i < fwd.size(); i++) {
            fwd.at(i).execute(s, fwd_args.at(i));
        }
    }
    start = std::chrono::high_resolution_clock::now();
    for (int repeat_idx = 0; repeat_idx < repeat; repeat_idx++) {
        for (size_t i = 0; i < fwd.size(); i++) {
            fwd.at(i).execute(s, fwd_args.at(i));
        }
        for (size_t i = 0; i < fwd_context.size(); i++) {
            fwd_context.at(i).execute(s, fwd_context_args.at(i));
        }
        s.wait();
    }
    end = std::chrono::high_resolution_clock::now();
    result.forward_start_timestamp = to_us(start.time_since_epoch());
    result.forward_time = duration_ms(start, end) / repeat;
    result.forward_end_timestamp = to_us(end.time_since_epoch());

    start = std::chrono::high_resolution_clock::now();
    for (int repeat_idx = 0; repeat_idx < repeat; repeat_idx++) {
        for (size_t i = 0; i < bwd_data.size(); i++) {
            bwd_data.at(i).execute(s, bwd_data_args.at(i));
        }
        for (size_t i = 0; i < bwd_weights.size(); i++) {
            bwd_weights.at(i).execute(s, bwd_weights_args.at(i));
        }
        s.wait();
    }
    end = std::chrono::high_resolution_clock::now();
    result.backward_start_timestamp = to_us(start.time_since_epoch());
    result.backward_time = duration_ms(start, end) / repeat;
    result.backward_end_timestamp = to_us(end.time_since_epoch());

    result.context_size = c.bwd_x.get_desc().get_size();
}

void profile_ours_conv(ProfileResult &result, engine &eng, stream &s,
                       int repeat, int warmup, int batch, int x_h, int x_w,
                       int c_in, int c_out, int stride, int kern_size,
                       int avg_rad) {
    std::chrono::time_point<std::chrono::system_clock> start, end;

    auto c = ConvOurs(eng, {batch, c_in, x_h, x_w}, c_out, true, true, stride,
                      kern_size, avg_rad);
    std::vector<primitive> fwd, fwd_context, bwd_filter, bwd_data, bwd_weights;
    std::vector<std::unordered_map<int, memory>> fwd_args, fwd_context_args,
        bwd_filter_args, bwd_data_args, bwd_weights_args;
    auto input = memory(c.conv_fwd_pd.src_desc(), eng);
    auto output = memory(c.conv_fwd_pd.dst_desc(), eng);
    auto grad_output = memory(c.avg_gy_pd.src_desc(), eng);
    auto grad_input = memory(c.bwd_x_conv_pd.diff_src_desc(), eng);
    c.forward_prim(input, output, fwd, fwd_args);
    c.forward_save_context_prim(input, fwd_context, fwd_context_args);
    c.backward_filter_prim(grad_output, bwd_filter, bwd_filter_args);
    c.backward_data_prim(grad_output, grad_input, bwd_data, bwd_data_args);
    c.backward_weights_prim(grad_output, bwd_weights, bwd_weights_args);

    for (int warmup_idx = 0; warmup_idx < warmup; warmup_idx++) {
        for (size_t i = 0; i < fwd.size(); i++) {
            fwd.at(i).execute(s, fwd_args.at(i));
        }
    }
    start = std::chrono::high_resolution_clock::now();
    for (int repeat_idx = 0; repeat_idx < repeat; repeat_idx++) {
        for (size_t i = 0; i < fwd.size(); i++) {
            fwd.at(i).execute(s, fwd_args.at(i));
        }
        for (size_t i = 0; i < fwd_context.size(); i++) {
            fwd_context.at(i).execute(s, fwd_context_args.at(i));
        }
        s.wait();
    }
    end = std::chrono::high_resolution_clock::now();
    result.forward_start_timestamp = to_us(start.time_since_epoch());
    result.forward_time = duration_ms(start, end) / repeat;
    result.forward_end_timestamp = to_us(end.time_since_epoch());

    start = std::chrono::high_resolution_clock::now();
    for (int repeat_idx = 0; repeat_idx < repeat; repeat_idx++) {
        for (size_t i = 0; i < bwd_filter.size(); i++) {
            bwd_filter.at(i).execute(s, bwd_filter_args.at(i));
        }
        for (size_t i = 0; i < bwd_data.size(); i++) {
            bwd_data.at(i).execute(s, bwd_data_args.at(i));
        }
        for (size_t i = 0; i < bwd_weights.size(); i++) {
            bwd_weights.at(i).execute(s, bwd_weights_args.at(i));
        }
        s.wait();
    }
    end = std::chrono::high_resolution_clock::now();
    result.backward_start_timestamp = to_us(start.time_since_epoch());
    result.backward_time = duration_ms(start, end) / repeat;
    result.backward_end_timestamp = to_us(end.time_since_epoch());
    result.context_size = c.bwd_x.get_desc().get_size();
}

void print_profiling_result(ProfileResult &result) {
    float fwd_time, bwd_time;
    printf("  Forward Time: %.3fms\n", result.forward_time);
    printf("  Backward Time: %.3fms\n", result.backward_time);
    printf("  Total Time: %.3fms\n",
           result.forward_time + result.backward_time);
    printf("  Context Size: %.3fKB\n", result.context_size / 1024.0);
}

int main(int argc, char **argv) {
    auto eng = engine(engine::kind::cpu, 0);
    stream s(eng);
    if (argc == 1) {
        ProfileResult vanilla_conv, ours_conv;
        profile_ours_conv(ours_conv, eng, s, 50, 5, 16, 30, 40, 512, 512, 1, 3,
                          2);
        profile_vanilla_conv(vanilla_conv, eng, s, 50, 5, 16, 30, 40, 512, 512,
                             1, 3);
        printf("---------------------------------\n");
        printf("Vanilla Convolution\n");
        print_profiling_result(vanilla_conv);
        printf("---------------------------------\n");
        printf("Ours Convolution\n");
        print_profiling_result(ours_conv);
        printf("---------------------------------\n");
        printf("Backward Speedup: %.3fx\n",
               vanilla_conv.backward_time / ours_conv.backward_time);
        printf("Context Reduction: %.3fx\n",
               vanilla_conv.context_size * 1.0 / ours_conv.context_size);
        printf("---------------------------------\n");
    } else if (argc == 3) {
        int repeat, warmup;
        int c_in, c_out, stride, kern_size, batch, x_h, x_w, avg_rad;
        std::ifstream cfg_file(argv[1], std::ios_base::in);
        if (!cfg_file) {
            printf("Bad config file\n");
            return -1;
        }
        std::ofstream result_file(argv[2], std::ios_base::out);
        if (!result_file) {
            printf("Cannot create output file\n");
            return -1;
        }
        cfg_file >> repeat >> warmup;
        result_file
            << "C_IN, C_OUT, Stride, Kernel_size, Batch, X_H, X_W, Filter_Rad, "
            << "Forward_Time[ms], Backward_Time[ms], Total_Time[ms], "
            << "Context_Size[KB], "
            << "Forward_Start_Timestamp[us], "
            << "Forward_End_Timestamp[us], "
            << "Backward_Start_Timestamp[us], "
            << "Backward_End_Timestamp[us]" << std::endl;
        printf("Warmup: %d Repeat: %d\n", warmup, repeat);
        while (cfg_file >> c_in >> c_out >> stride >> kern_size >> batch >>
               x_h >> x_w >> avg_rad) {
            printf("Start New Test:\n");
            printf(
                "C_In: %d C_Out: %d Stride: %d Kernel Size: %d\nBatch Size: %d "
                "X_H: %d X_W: %d Filter Size: %d\n",
                c_in, c_out, stride, kern_size, batch, x_h, x_w, avg_rad);
            ProfileResult result;
            if (avg_rad < 1) {
                profile_vanilla_conv(result, eng, s, repeat, warmup, batch, x_h,
                                     x_w, c_in, c_out, stride, kern_size);
            } else {
                profile_ours_conv(result, eng, s, repeat, warmup, batch, x_h,
                                  x_w, c_in, c_out, stride, kern_size, avg_rad);
            }
            result_file << c_in << ", " << c_out << ", " << stride << ", "
                        << kern_size << ", ";
            result_file << batch << ", " << x_h << ", " << x_w << ", "
                        << avg_rad << ", ";
            result_file << result.forward_time << ", " << result.backward_time
                        << ", ";
            result_file << result.forward_time + result.backward_time << ", ";
            result_file << result.context_size / 1024.0 << ", ";
            result_file << result.forward_start_timestamp.count() << ", "
                        << result.forward_end_timestamp.count() << ", ";
            result_file << result.backward_start_timestamp.count() << ", "
                        << result.backward_end_timestamp.count();
            result_file << std::endl;
            printf("Done Sleep 500 ms\n");
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    } else {
        printf("Bad args\n");
        return -1;
    }
    return 0;
}
