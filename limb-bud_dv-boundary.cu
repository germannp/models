#include <curand_kernel.h>
#include <stdio.h>
#include <random>

#include "../include/cudebug.cuh"
#include "../include/dtypes.cuh"
#include "../include/inits.cuh"
#include "../include/links.cuh"
#include "../include/polarity.cuh"
#include "../include/property.cuh"
#include "../include/solvers.cuh"
#include "../include/vtk.cuh"


const auto n_0 = 2977;
const auto n_max = 61000;
// const auto proliferation_rate = 0.01;
const auto mean_distance = 0.75;
const auto n_time_steps = 500;
const auto dt = 0.2;

MAKE_PT(Lb_cell, w, e, theta, phi);


__device__ Lb_cell lb_force(Lb_cell Xi, Lb_cell r, float dist, int i, int j)
{
    Lb_cell dF{0};
    if (i == j) {
        dF.w = 0.01 / (0.01 + Xi.e * Xi.e) - Xi.w;
        dF.e = 0.01 / (0.01 + Xi.w * Xi.w) - Xi.e;
        return dF;
    }

    if (dist > 1) return dF;

    auto F = fmaxf(0.7 - dist, 0) * 2 - fmaxf(dist - 0.8, 0) * 1.2;
    dF.x = r.x * F / dist;
    dF.y = r.y * F / dist;
    dF.z = r.z * F / dist;
    // dF.w = -0.1 * r.w;

    dF += bending_force(Xi, r, dist) * 0.1;
    return dF;
}


// __global__ void proliferate(
//     int n_0, Lb_cell* d_X, int* d_n_cells, curandState* d_state)
// {
//     auto i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i >= n_0) return;  // Dividing new cells is problematic!

//     auto rnd = curand_uniform(&d_state[i]);
//     if (rnd > proliferation_rate) return;

//     auto n = atomicAdd(d_n_cells, 1);
//     D_ASSERT(n <= n_max);
//     auto phi = curand_uniform(&d_state[i]) * M_PI;
//     auto theta = curand_uniform(&d_state[i]) * 2 * M_PI;
//     d_X[n].x = d_X[i].x + mean_distance / 4 * sinf(theta) * cosf(phi);
//     d_X[n].y = d_X[i].y + mean_distance / 4 * sinf(theta) * sinf(phi);
//     d_X[n].z = d_X[i].z + mean_distance / 4 * cosf(theta);
//     d_X[n].w = d_X[i].w / 2;
//     d_X[i].w = d_X[i].w / 2;
//     d_X[n].e = d_X[i].e / 2;
//     d_X[i].e = d_X[i].e / 2;
//     d_X[n].theta = d_X[i].theta;
//     d_X[n].phi = d_X[i].phi;
// }


int main(int argc, const char* argv[])
{
    // Prepare initial state
    Solution<Lb_cell, Grid_solver> cells{n_max, 80};
    *cells.h_n = n_0;
    regular_hexagon(mean_distance, cells);
    for (auto i = 0; i < *cells.h_n; i++) {
        cells.h_X[i].w = rand() / (RAND_MAX + 1.);
        cells.h_X[i].e = rand() / (RAND_MAX + 1.);
        // if (cells.h_X[i].x < 0) cells.h_X[i].w = 1;
    }
    cells.copy_to_device();
    // Links protrusions{0, 0.1};

    // Run simulation
    Vtk_output output{"dv-boundary"};
    for (auto time_step = 0; time_step <= n_time_steps; time_step++) {
        cells.copy_to_host();

        // proliferate<<<(cells.get_d_n() + 128 - 1) / 128, 128>>>(
        //     cells.get_d_n(), cells.d_X, cells.d_n, protrusions.d_state);
        cells.take_step<lb_force>(dt);

        output.write_positions(cells);
        output.write_polarity(cells);
        output.write_field(cells, "Wnt7a");
        output.write_field(cells, "En1", &Lb_cell::e);
    }

    return 0;
}
