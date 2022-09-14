#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <algorithm>
// #include <iostream>

// #include <omp.h>

#define NEAREST(x) int(std::round(x))
#define INRANGE(x,y) (0<=(y) && (y)<h && 0<=(x) && (x)<w)

// std::tuple<float, float, float> proj(
//     int x, int y, float d, 
//     torch::TensorAccessor<float, 2> iIr, torch::TensorAccessor<float, 2> iEr,
//     torch::TensorAccessor<float, 2> Is, torch::TensorAccessor<float, 2> Es
// ) {
//     float xi = x * 1.0 + 0.5;
//     float yi = y * 1.0 + 0.5;
//     float xc = (iIr[0][0] * xi + iIr[0][2]) * d;
//     float yc = (iIr[1][1] * yi + iIr[1][2]) * d;
//     float xw = xc * iEr[0][0] + yc * iEr[0][1] + d * iEr[0][2] + iEr[0][3];
//     float yw = xc * iEr[1][0] + yc * iEr[1][1] + d * iEr[1][2] + iEr[1][3];
//     float zw = xc * iEr[2][0] + yc * iEr[2][1] + d * iEr[2][2] + iEr[2][3];
//     float ds = xw * Es[2][0] + yw * Es[2][1] + zw * Es[2][2] + Es[2][3];
//     float xcs = (xw * Es[0][0] + yw * Es[0][1] + zw * Es[0][2] + Es[0][3]) / ds;
//     float ycs = (xw * Es[1][0] + yw * Es[1][1] + zw * Es[1][2] + Es[1][3]) / ds;
//     float xis = xcs * Is[0][0] + Is[0][2];
//     float yis = ycs * Is[1][1] + Is[1][2];
//     return {xis, yis, ds};
// }

// torch::Tensor vis_fusion_core(
//     const torch::Tensor& t_depth,  // m
//     const torch::Tensor& t_xy,  // m2
//     const torch::Tensor& t_valid,  // hw
//     const torch::Tensor& t_iIr,  // 33
//     const torch::Tensor& t_iEr,  // 44
//     const torch::Tensor& t_Is,  // v33
//     const torch::Tensor& t_Es,  // v44
//     const torch::Tensor& t_srcs_depth  // vhw
// ) {
//     omp_set_dynamic(false);
//     omp_set_num_threads(8);

//     int m = t_depth.size(0);
//     int h = t_valid.size(0);
//     int w = t_valid.size(1);
//     int v = t_srcs_depth.size(0);

//     auto depth = t_depth.accessor<float, 1>();
//     auto xy = t_xy.accessor<float, 2>();
//     auto valid = t_valid.accessor<bool, 2>();
//     auto iIr = t_iIr.accessor<float, 2>();
//     auto iEr = t_iEr.accessor<float, 2>();
//     auto srcs_depth = t_srcs_depth.accessor<float, 3>();

//     std::vector<std::vector<float>> depths(h*w);
//     for (int i=0; i<m; i++) {
//         int y = NEAREST(xy[i][1] - .5);
//         int x = NEAREST(xy[i][0] - .5);
//         if (INRANGE(x, y) && depth[i] > 1e-9 && valid[y][x])
//             depths[y*w+x].push_back(depth[i]);
//     }

//     auto t_out = torch::zeros({h,w}, torch::dtype(torch::kF32));
//     auto out = t_out.accessor<float, 2>();
//     #pragma omp parallel for
//     for (int i=0; i<h; i++) {
//         for (int j=0; j<w; j++) {
//             auto& d_list = depths[i*w+j];
//             if (d_list.size() == 0) continue;
//             std::sort(d_list.begin(), d_list.end());
//             for (int k=0; k<d_list.size(); k++) {
//                 float d = d_list[k];
//                 int vio = 0;
//                 for (int vi=0; vi<v; vi++) {
//                     auto t_Is_i = t_Is.index({vi});
//                     auto t_Es_i = t_Es.index({vi});
//                     auto [xis, yis, ds] = proj(j, i, d, iIr, iEr, t_Is_i.accessor<float, 2>(), t_Es_i.accessor<float, 2>());
//                     if (!INRANGE(NEAREST(xis), NEAREST(yis))) continue;
//                     float d_est = srcs_depth[vi][NEAREST(yis)][NEAREST(xis)];
//                     if (d_est > ds) vio++;
//                 }
//                 if (k >= vio || k == d_list.size()-1) {
//                     out[i][j] = d;
//                     break;
//                 }
//             }
//         }
//     }

//     return t_out;
// }

torch::Tensor vis_fusion_core(
    torch::Tensor t_depth,
    torch::Tensor t_xy,
    torch::Tensor t_violation,
    torch::Tensor t_valid
) {
    
    int m = t_depth.size(0);
    int h = t_valid.size(0);
    int w = t_valid.size(1);

    auto depth = t_depth.accessor<float, 1>();
    auto xy = t_xy.accessor<float, 2>();
    auto violation = t_violation.accessor<int, 1>();
    auto valid = t_valid.accessor<bool, 2>();

    std::vector<std::vector<std::tuple<float, int>>> depths(h*w);
    for (int i=0; i<m; i++) {
        int y = NEAREST(xy[i][1] - .5);
        int x = NEAREST(xy[i][0] - .5);
        if (INRANGE(x,y) && depth[i] > 1e-9 && valid[y][x])
            depths[y*w+x].push_back({depth[i], violation[i]});
    }

    auto t_out = torch::zeros({h,w}, torch::dtype(torch::kF32));
    auto out = t_out.accessor<float, 2>();
    for (int i=0; i<h; i++) {
        for (int j=0; j<w; j++) {
            auto& d_list = depths[i*w+j];
            if (d_list.size() == 0) continue;
            std::sort(d_list.begin(), d_list.end());
            for (int k=0; k<d_list.size(); k++) {
                auto [d, vio] = d_list[k];
                if (k >= vio || k == d_list.size()-1) {
                    out[i][j] = d;
                    break;
                }
            }
        }
    }

    return t_out;
}

#define INIT 0
#define IN_QUEUE 1
#define FINISH 2

torch::Tensor small_seg_core(
    const torch::Tensor& t_depth,
    const int window_size,
    const float depth_diff_thresh,
    const int seg_size_thresh
) {
    std::vector<std::tuple<int,int>> neighbors;
    for (int i=-window_size; i<=window_size; i++)
        for (int j=-window_size; j<=window_size; j++)
            if (!(i==0 && j==0))
                neighbors.push_back({i, j});

    int h = t_depth.size(0);
    int w = t_depth.size(1);

    auto depth = t_depth.accessor<float, 2>();

    auto t_out = torch::ones({h,w}, torch::dtype(torch::kU8));
    auto out = t_out.accessor<u_int8_t, 2>();

    auto *visit = new u_int8_t[h*w]; memset(visit, INIT, sizeof(u_int8_t)*h*w);
    for (int i=0; i<h; i++) {
        for (int j=0; j<w; j++) {
            if (depth[i][j] < 1e-9) {
                visit[i*w+j] = FINISH;
                out[i][j] = 0;
            }
        }
    }
    for (int i=0; i<h; i++) {
        for (int j=0; j<w; j++) {
            if (visit[i*w+j] == FINISH) continue;
            // if (visit[i*w+j] == IN_QUEUE) {cout << i << " " << j << " new loop element is in wrong state\n";}
            std::vector<std::tuple<int,int>> queue;
            queue.push_back({i, j});
            visit[i*w+j] = IN_QUEUE;
            for (int k=0; k<queue.size(); k++) {
                auto [curr_i, curr_j] = queue[k];
                float curr_d = depth[curr_i][curr_j];
                // if (visit[curr_i*w+curr_j] != IN_QUEUE) {cout << "new queue element is in wrong state";}
                for (auto [di, dj]: neighbors) {
                    int next_i = curr_i + di;
                    int next_j = curr_j + dj;
                    if (!(0<=next_i && next_i<h && 0<=next_j && next_j<w)) continue;
                    if (visit[next_i*w+next_j] != INIT) continue;
                    float next_d = depth[next_i][next_j];
                    // if (next_d < 1e-9) continue;
                    if (std::fabs(curr_d-next_d) >= (depth_diff_thresh * (curr_d+next_d))) continue;
                    queue.push_back({next_i, next_j});
                    visit[next_i*w+next_j] = IN_QUEUE;
                }
                visit[curr_i*w+curr_j] = FINISH;
            }
            if (queue.size() < seg_size_thresh)
                for (auto [curr_i, curr_j]: queue)
                    out[curr_i][curr_j] = 0;
        }
    }

    delete[] visit;
    return t_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vis_fusion_core", &vis_fusion_core, "");
    m.def("small_seg_core", &small_seg_core, "");
}