// Simulate mono-polar migration
#include "../include/dtypes.cuh"
#include "../include/inits.cuh"
#include "../include/polarity.cuh"
#include "../include/solvers.cuh"
#include "../include/vtk.cuh"


const auto r_max = 1;
const auto n_cells = 500;
const auto n_time_steps = 150;
const auto dt = 0.05;


__device__ Po_cell relu_w_migration(
    Po_cell Xi, Po_cell r, float dist, int i, int j)
{
    Po_cell dF{0};
    if (i == j) return dF;

    if (dist > r_max) return dF;

    auto F = fmaxf(0.7 - dist, 0) * 2 - fmaxf(dist - 0.8, 0);
    dF.x = r.x * F / dist;
    dF.y = r.y * F / dist;
    dF.z = r.z * F / dist;

    dF += migration_force(Xi, r, dist);
    return dF;
}


int main(int argc, const char* argv[])
{
    // Prepare initial state
    Solution<Po_cell, Tile_solver> cells{n_cells};
    relaxed_sphere(0.75, cells);
    cells.h_X[0].x = 0;
    cells.h_X[0].y = 0;
    cells.h_X[0].z = 0;
    cells.h_X[0].phi = 0.01;
    cells.copy_to_device();

    // Integrate cell positions
    Vtk_output output{"random-walk"};
    for (auto time_step = 0; time_step <= n_time_steps; time_step++) {
        cells.copy_to_host();
        cells.take_step<relu_w_migration>(dt);
        output.write_positions(cells);
        output.write_polarity(cells);
    }

    return 0;
}