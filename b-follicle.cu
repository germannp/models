// Simulate B cell follicle formation
#include "../include/dtypes.cuh"
#include "../include/inits.cuh"
#include "../include/polarity.cuh"
#include "../include/property.cuh"
#include "../include/solvers.cuh"
#include "../include/vtk.cuh"


const auto r_max = 1;
const auto n_cells = 1000;
const auto n_time_steps = 500;
const auto dt = 0.05;
enum Cell_types { tcell, bcell, fdc };

MAKE_PT(Ln_cell, cxcl13, lta1b2, theta, phi);


__device__ Cell_types* d_type;

__device__ Ln_cell relu_w_migration(
    Ln_cell Xi, Ln_cell r, float dist, int i, int j)
{
    Ln_cell dF{0};
    if (i == j) {
        dF.cxcl13 = (d_type[i] == fdc) - 0.01 * Xi.cxcl13;
        return dF;
    }

    if (dist > r_max) return dF;

    auto F = fmaxf(0.7 - dist, 0) * 2 - fmaxf(dist - 0.8, 0);
    dF.x = r.x * F / dist;
    dF.y = r.y * F / dist;
    dF.z = r.z * F / dist;
    dF.cxcl13 = -r.cxcl13;

    dF += migration_force(Xi, r, dist);
    return dF;
}


int main(int argc, const char* argv[])
{
    // Prepare initial state
    Solution<Ln_cell, Tile_solver> cells{n_cells};
    relaxed_sphere(0.75, cells);
    Property<Cell_types> type{n_cells};
    cudaMemcpyToSymbol(d_type, &type.d_prop, sizeof(d_type));
    for (auto i = 0; i < n_cells; i++) type.h_prop[i] = bcell;
    type.h_prop[0] = fdc;
    type.copy_to_device();

    // Integrate cell positions
    Vtk_output output{"b-follicle"};
    for (auto time_step = 0; time_step <= n_time_steps; time_step++) {
        cells.copy_to_host();
        cells.take_step<relu_w_migration>(dt);
        output.write_positions(cells);
        output.write_property(type);
        output.write_field(cells, "Cxcl13", &Ln_cell::cxcl13);
    }

    return 0;
}