// Simulate cell sorting by protrusions
#include <curand_kernel.h>

#include "../include/dtypes.cuh"
#include "../include/inits.cuh"
#include "../include/links.cuh"
#include "../include/property.cuh"
#include "../include/solvers.cuh"
#include "../include/vtk.cuh"


const auto type_ratio = 0.2f;
const auto n_cells = 1000u;
const auto prots_per_cell = 5;
const auto r_protrusion = 1.25;
const auto prot_strength = 0.3f;
const auto n_time_steps = 300u;
const auto dt = 0.1;
const auto n_protrusions =
    static_cast<int>(n_cells * type_ratio * prots_per_cell);


__device__ float3 relu_force(float3 Xi, float3 r, float dist, int i, int j)
{
    float3 dF{0};
    if (i == j) return dF;

    if (dist > 1.0) return dF;

    auto F = fmaxf(0.7 - dist, 0) * 2 - fmaxf(dist - 0.8, 0);
    dF = r * F / dist;
    return dF;
}


__global__ void update_protrusions(
    const float3* __restrict__ d_X, curandState* d_state, Link* d_link)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_protrusions) return;

    auto a = static_cast<int>((i + 0.5) / prots_per_cell);
    auto b = d_link[i].b;
    auto r = d_X[a] - d_X[b];
    auto dist = norm3df(r.x, r.y, r.z);
    if ((dist > r_protrusion) or (dist < 0.75)) {
        d_link[i].a = a;
        d_link[i].b = a;
    }

    auto new_b = min(
        static_cast<int>(curand_uniform(&d_state[i]) * n_cells * type_ratio),
        static_cast<int>(n_cells * type_ratio - 1));
    if (a == new_b) return;

    r = d_X[a] - d_X[new_b];
    dist = norm3df(r.x, r.y, r.z);
    if (dist > r_protrusion) return;

    d_link[i].a = a;
    d_link[i].b = new_b;
}


int main(int argc, const char* argv[])
{
    // Prepare initial state
    Solution<float3, Grid_solver> cells{n_cells};
    random_sphere(1, cells);
    Links protrusions{n_protrusions, prot_strength};
    auto prot_forces = [&protrusions](
                           const float3* __restrict__ d_X, float3* d_dX) {
        return link_forces(protrusions, d_X, d_dX);
    };
    Property<> type{n_cells};
    for (auto i = 0; i < n_cells; i++) {
        type.h_prop[i] = (i < n_cells * type_ratio) ? 0 : 1;
    }

    // Integrate cell positions
    Vtk_output output{"sorting"};
    for (auto time_step = 0; time_step <= n_time_steps; time_step++) {
        cells.copy_to_host();
        protrusions.copy_to_host();
        update_protrusions<<<(n_protrusions + 32 - 1) / 32, 32>>>(
            cells.d_X, protrusions.d_state, protrusions.d_link);
        cells.take_step<relu_force>(dt, prot_forces);
        output.write_positions(cells);
        output.write_links(protrusions);
        output.write_property(type);
    }

    return 0;
}
