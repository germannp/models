// Makes initial conditions for a limb bud taking the morphology from a 3D model
//(3D mesh), then fills the volume with mesenchymal cells and the surface with
// epithelial cells, then lets teh system relax.

// Command line arguments
// argv[1]=input mesh file name
// argv[2]=output file tag
// argv[3]=target limb bud size (dx)
// argv[4]=cube relax_time
// argv[5]=limb bud relax_time
// argv[6]=links flag (activate if you want to use links in later simulations)
// argv[7]=wall flag (activate in limb buds, when you want a wall boundary
// cond.). argv[8]=AER flag (activate in limb buds) argv[9]=Optimum mesh file
// name

#include <curand_kernel.h>
#include <time.h>
#include <iostream>
#include <list>
#include <string>
#include <vector>

#include "../include/dtypes.cuh"
#include "../include/inits.cuh"
#include "../include/links.cuh"
#include "../include/mesh.cuh"
#include "../include/polarity.cuh"
#include "../include/property.cuh"
#include "../include/solvers.cuh"
#include "../include/vtk.cuh"

const auto r_max = 1.0;
const auto r_min = 0.8;
const auto dt = 0.1;
const auto n_max = 250000;
const auto skip_step = 1;
const auto prots_per_cell = 1;
const auto protrusion_strength = 0.2f;
const auto r_protrusion = 2.0f;

enum Cell_types { mesenchyme, epithelium, aer };

__device__ Cell_types* d_type;
__device__ int* d_freeze;

MAKE_PT(Cell, theta, phi);

__device__ Cell relaxation_force(Cell Xi, Cell r, float dist, int i, int j)
{
    Cell dF{0};

    if (i == j) return dF;

    if (d_freeze[i] == 1)
        return dF;  // frozen cells don't experience force so don't move

    if (dist > r_max) return dF;

    float F;
    if (d_type[i] == d_type[j]) {
        if (d_type[i] == mesenchyme)
            F = fmaxf(0.8 - dist, 0) * 2.f - fmaxf(dist - 0.8, 0);
        else
            F = fmaxf(0.8 - dist, 0) * 2.f - fmaxf(dist - 0.8, 0) * 2.f;
    } else {
        F = fmaxf(0.9 - dist, 0) * 2.f - fmaxf(dist - 0.9, 0) * 2.f;
    }
    dF.x = r.x * F / dist;
    dF.y = r.y * F / dist;
    dF.z = r.z * F / dist;

    if (d_type[i] >= epithelium && d_type[j] >= epithelium)
        dF += bending_force(Xi, r, dist) * 0.10f;

    return dF;
}

__device__ Cell wall_force(Cell Xi, Cell r, float dist, int i, int j)
{
    Cell dF{0};

    if (i == j) return dF;

    if (dist > r_max) return dF;

    float F;
    if (d_type[i] == d_type[j]) {
        if (d_type[i] == mesenchyme)
            F = fmaxf(0.8 - dist, 0) * 2.f - fmaxf(dist - 0.8, 0);
        else
            F = fmaxf(0.8 - dist, 0) * 2.f - fmaxf(dist - 0.8, 0) * 2.f;
    } else {
        F = fmaxf(0.9 - dist, 0) * 2.f - fmaxf(dist - 0.9, 0) * 2.f;
    }
    dF.x = r.x * F / dist;
    dF.y = r.y * F / dist;
    dF.z = r.z * F / dist;

    if (d_type[i] >= epithelium && d_type[j] >= epithelium)
        dF += bending_force(Xi, r, dist) * 0.5f;

    if (Xi.x < 0) dF.x = 0.f;

    return dF;
}

__global__ void update_protrusions(const int n_cells,
    const Grid* __restrict__ d_grid, const Cell* __restrict d_X,
    curandState* d_state, Link* d_link)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cells * prots_per_cell) return;

    auto j = static_cast<int>((i + 0.5) / prots_per_cell);
    auto rand_nb_cube =
        d_grid->d_cube_id[j] +
        d_nhood[min(static_cast<int>(curand_uniform(&d_state[i]) * 27), 26)];
    auto cells_in_cube =
        d_grid->d_cube_end[rand_nb_cube] - d_grid->d_cube_start[rand_nb_cube];
    if (cells_in_cube < 1) return;

    auto a = d_grid->d_point_id[j];
    auto b =
        d_grid->d_point_id[d_grid->d_cube_start[rand_nb_cube] +
                           min(static_cast<int>(
                                   curand_uniform(&d_state[i]) * cells_in_cube),
                               cells_in_cube - 1)];
    D_ASSERT(a >= 0);
    D_ASSERT(a < n_cells);
    D_ASSERT(b >= 0);
    D_ASSERT(b < n_cells);
    if (a == b) return;

    if ((d_type[a] != mesenchyme) or (d_type[b] != mesenchyme)) return;

    auto new_r = d_X[a] - d_X[b];
    auto new_dist = norm3df(new_r.x, new_r.y, new_r.z);
    if (new_dist > r_protrusion) return;

    auto link = &d_link[a * prots_per_cell + i % prots_per_cell];
    auto not_initialized = link->a == link->b;
    auto new_one = curand_uniform(&d_state[i]) < 0.05f;
    if (not_initialized || new_one) {
        link->a = a;
        link->b = b;
    }
}

__device__ float relaxation_friction(Cell Xi, Cell r, float dist, int i, int j)
{
    return 0;
}

__device__ float freeze_friction(Cell Xi, Cell r, float dist, int i, int j)
{
    if (i == j) return 0;
    if (d_freeze[i] == 1) return 0;
    return 1;
}

template<typename Pt, template<typename> class Solver>
void fill_solver_w_mesh_no_flank(
    Mesh mesh, Solution<Pt, Solver>& cells, unsigned int n_0 = 0)
{
    // eliminate the flank boundary
    int i = 0;
    while (i < mesh.facets.size()) {
        if (mesh.facets[i].n.x > -1.01f && mesh.facets[i].n.x < -0.99f)
            mesh.facets.erase(mesh.facets.begin() + i);
        else
            i++;
    }
    mesh.n_facets = mesh.facets.size();

    i = 0;
    while (i < mesh.vertices.size()) {
        if (mesh.vertices[i].x > -0.01f && mesh.vertices[i].x < 0.01f)
            mesh.vertices.erase(mesh.vertices.begin() + i);
        else
            i++;
    }
    mesh.n_vertices = mesh.vertices.size();


    *cells.h_n = mesh.n_facets + mesh.n_vertices;
    assert(n_0 < *cells.h_n);

    for (int i = 0; i < mesh.n_facets; i++) {
        auto T = mesh.facets[i];
        float r = sqrt(pow(T.n.x, 2) + pow(T.n.y, 2) + pow(T.n.z, 2));
        cells.h_X[i].x = T.C.x;
        cells.h_X[i].y = T.C.y;
        cells.h_X[i].z = T.C.z;
        cells.h_X[i].phi = atan2(T.n.y, T.n.x);
        cells.h_X[i].theta = acos(T.n.z / r);
    }
    for (int i = 0; i < mesh.n_vertices; i++) {
        auto P = mesh.vertices[i];
        cells.h_X[mesh.n_facets + i].x = P.x;
        cells.h_X[mesh.n_facets + i].y = P.y;
        cells.h_X[mesh.n_facets + i].z = P.z;
    }
}

template<typename Pt, template<typename> class Solver, typename Prop>
void fill_solver_w_epithelium(Solution<Pt, Solver>& incells,
    Solution<Pt, Solver>& outcells, Prop& type, unsigned int n_0 = 0)
{
    assert(n_0 < *incells.h_n);
    assert(n_0 < *outcells.h_n);

    int j = 0;
    for (int i = 0; i < *incells.h_n; i++) {
        if (type.h_prop[i] == epithelium) {
            outcells.h_X[j].x = incells.h_X[i].x;
            outcells.h_X[j].y = incells.h_X[i].y;
            outcells.h_X[j].z = incells.h_X[i].z;
            outcells.h_X[j].phi = incells.h_X[i].phi;
            outcells.h_X[j].theta = incells.h_X[i].theta;
            j++;
        }
    }
    *outcells.h_n = j;
}


int main(int argc, char const* argv[])
{
    std::string file_name = argv[1];
    std::string output_tag = argv[2];
    float target_dx = std::stof(argv[3]);
    int cube_relax_time = std::stoi(argv[4]);
    int epi_relax_time = std::stoi(argv[5]);
    bool links_flag = false;
    if (std::stoi(argv[6]) == 1) links_flag = true;
    bool wall_flag = false;
    if (std::stoi(argv[7]) == 1) wall_flag = true;
    bool AER_flag = false;
    if (std::stoi(argv[8]) == 1) AER_flag = true;
    std::string optimum_file_name = argv[9];

    Mesh mesh(file_name);

    // Compute max length in X axis to know how much we need to rescale
    auto min_point = mesh.get_minimum();
    auto diagonal_vector = mesh.get_maximum() - min_point;
    float resc = target_dx / diagonal_vector.x;
    std::cout << "xmax= " << min_point.x + diagonal_vector.x
              << " xmin= " << min_point.x << std::endl;
    std::cout << "dx= " << diagonal_vector.x << " target_dx= " << target_dx
              << " rescaling factor resc= " << resc << std::endl;


    // mesh defines the overall shape of the limb bud (mesench. + ectoderm)
    mesh.rescale(resc);
    // mesh.rotate(0.0f,0.0f,-0.2f); // formula for old "limb only" meshes
    // around    z    y     x
    mesh.rotate(0.5f, 0.0f, M_PI - 0.2f);  // formula for old "limb only" meshes

    // mesh_mesench defines the volume occupied by the mesenchyme (smaller than
    // mesh)
    Mesh mesh_mesench = mesh;
    mesh_mesench.grow_normally(-r_min, wall_flag);  //*1.3//*1.2

    // we use the maximum lengths of the mesh to draw a cube that includes the
    // mesh
    // Let's fill the cube with cells
    Solution<Cell, Grid_solver> cube{n_max};
    relaxed_cuboid(r_min, mesh.get_minimum(), mesh.get_maximum(), cube);
    auto n_cells_cube = *cube.h_n;

    cube.copy_to_host();

    for (int i = 0; i < n_cells_cube; i++) {
        cube.h_X[i].theta = 0.f;
        cube.h_X[i].phi = 0.f;
    }

    // The relaxed cube positions will be used to imprint epithelial cells
    std::vector<float3> cube_relax_points;
    for (auto i = 0; i < n_cells_cube; i++) {
        auto p = float3{cube.h_X[i].x, cube.h_X[i].y, cube.h_X[i].z};
        cube_relax_points.push_back(p);
    }

    // Variable indicating cell type
    Property<Cell_types> type{n_max};
    cudaMemcpyToSymbol(d_type, &type.d_prop, sizeof(d_type));
    // Variable that indicates which cells are 'frozen', so don't move
    Property<int> freeze{n_max, "freeze"};
    cudaMemcpyToSymbol(d_freeze, &freeze.d_prop, sizeof(d_freeze));

    for (auto i = 0; i < n_cells_cube; i++) {
        type.h_prop[i] = mesenchyme;
        freeze.h_prop[i] = 0;
    }

    cube.copy_to_device();
    type.copy_to_device();
    freeze.copy_to_device();

    // Declaration of links
    Links protrusions(n_max * prots_per_cell, protrusion_strength);
    protrusions.set_d_n(n_cells_cube * prots_per_cell);
    auto intercalation = std::bind(link_forces<Cell>, protrusions,
        std::placeholders::_1, std::placeholders::_2);

    Grid grid{n_max};

    // State for links
    curandState* d_state;
    cudaMalloc(&d_state, n_max * sizeof(curandState));
    auto seed = time(NULL);
    setup_rand_states<<<(n_max + 128 - 1) / 128, 128>>>(n_max, seed, d_state);

    if (links_flag) {
        // Vtk_output cubic_output(output_tag+".cubic_relaxation");

        // We apply the links to the relaxed cube to compress it (as will be the
        // mesench in the limb bud)
        for (auto time_step = 0; time_step <= cube_relax_time; time_step++) {
            // if(time_step%skip_step==0 || time_step==cube_relax_time){
            //     cube.copy_to_host();
            //     protrusions.copy_to_host();
            // }

            protrusions.set_d_n(cube.get_d_n() * prots_per_cell);
            grid.build(cube, r_protrusion);
            update_protrusions<<<(protrusions.get_d_n() + 32 - 1) / 32, 32>>>(
                cube.get_d_n(), grid.d_grid, cube.d_X, protrusions.d_state,
                protrusions.d_link);

            cube.take_step<relaxation_force, relaxation_friction>(
                dt, intercalation);

            // write the output
            // if(time_step%skip_step==0 || time_step==cube_relax_time) {
            //     cubic_output.write_positions(cube);
            //     cubic_output.write_links(protrusions);
            // }
        }
        std::cout
            << "Cube 2 integrated with links (only when links flag is active)"
            << std::endl;
    }

    // Fit the cube into a mesh and sort which cells are inside the mesh
    // For the mesenchyme we use the smaller mesh and the compressed cube
    // For the epithelium we use the larger mesh and the relaxed cube

    // Mesenchyme
    // Setup the list of points
    std::vector<float3> cube_points;
    for (auto i = 0; i < n_cells_cube; i++) {
        auto p = float3{cube.h_X[i].x, cube.h_X[i].y, cube.h_X[i].z};
        cube_points.push_back(p);
    }

    // Make a new list with the ones that are inside
    std::vector<float3> mes_cells;
    int n_cells_mes = 0;
    for (int i = 0; i < n_cells_cube; i++) {
        if (!mesh_mesench.test_exclusion(cube_points[i])) {
            mes_cells.push_back(cube_points[i]);
            n_cells_mes++;
        }
    }

    std::cout << "cells_in_cube " << n_cells_cube << " cells after fill "
              << n_cells_mes << std::endl;

    // Epithelium (we have to sort out which ones are inside the big mesh and
    // out of the small one)
    // Make a new list with the ones that are inside
    std::vector<float3> epi_cells;
    int n_cells_epi = 0;
    for (int i = 0; i < n_cells_cube; i++) {
        if (!mesh.test_exclusion(cube_relax_points[i]) and
            mesh_mesench.test_exclusion(cube_relax_points[i])) {
            epi_cells.push_back(cube_relax_points[i]);
            n_cells_epi++;
        }
    }

    int n_cells_total = n_cells_mes + n_cells_epi;

    std::cout << "cells_in_mes " << n_cells_mes << " cells_in_epi "
              << n_cells_epi << " cells_in_total " << n_cells_total
              << std::endl;

    Solution<Cell, Grid_solver> cells{n_max};
    *cells.h_n = n_cells_total;

    for (int i = 0; i < n_cells_mes; i++) {
        cells.h_X[i].x = mes_cells[i].x;
        cells.h_X[i].y = mes_cells[i].y;
        cells.h_X[i].z = mes_cells[i].z;
        cells.h_X[i].phi = 0.f;
        cells.h_X[i].theta = 0.f;
        type.h_prop[i] = mesenchyme;
        freeze.h_prop[i] = 1;
    }
    int count = 0;
    for (int i = n_cells_mes; i < n_cells_total; i++) {
        cells.h_X[i].x = epi_cells[count].x;
        cells.h_X[i].y = epi_cells[count].y;
        cells.h_X[i].z = epi_cells[count].z;
        type.h_prop[i] = epithelium;
        freeze.h_prop[i] = 0;
        // polarity
        auto p = epi_cells[count];
        int f = -1;
        float dmin = 1000000.f;
        // we use the closest facet on mesh to determine the polarity of the
        // epithelial cell
        for (int j = 0; j < mesh.n_facets; j++) {
            auto r = p - mesh.facets[j].C;
            float d = sqrt(r.x * r.x + r.y * r.y + r.z * r.z);
            if (d < dmin) {
                dmin = d;
                f = j;
            }
        }
        count++;
        if (mesh.facets[f].C.x < 0.1f &&
            wall_flag) {  // the cells contacting the flank
                          // boundary can't be epithelial 0.001
            type.h_prop[i] = mesenchyme;
            cells.h_X[i].phi = 0.f;
            cells.h_X[i].theta = 0.f;
            freeze.h_prop[i] = 1;
            continue;
        }
        cells.h_X[i].phi = atan2(mesh.facets[f].n.y, mesh.facets[f].n.x);
        cells.h_X[i].theta = acos(mesh.facets[f].n.z);
    }
    std::cout << "count " << count << " epi_cells " << n_cells_epi << std::endl;

    cells.copy_to_device();
    type.copy_to_device();
    freeze.copy_to_device();

    std::cout << "n_cells_total= " << n_cells_total << std::endl;

    if (AER_flag) {
        // Imprint the AER on the epithelium (based on a mesh file too)
        std::string AER_file = file_name;
        AER_file.insert(AER_file.length() - 4, "_AER");
        std::cout << "AER file " << AER_file << std::endl;
        Mesh AER(AER_file);
        AER.rescale(resc);

        for (int i = n_cells_mes; i < n_cells_total; i++) {
            float3 p{cells.h_X[i].x, cells.h_X[i].y, cells.h_X[i].z};
            for (int j = 0; j < AER.n_facets; j++) {
                auto r = p - AER.facets[j].C;
                float d = sqrt(r.x * r.x + r.y * r.y + r.z * r.z);
                if (d < r_min * 1.5f) {
                    type.h_prop[i] = aer;
                    break;
                }
            }
        }

        AER.write_vtk(output_tag + ".aer");
    }

    Vtk_output output{output_tag};

    for (auto time_step = 0; time_step <= epi_relax_time; time_step++) {
        // if (time_step % skip_step == 0 || time_step == epi_relax_time) {
        //     cells.copy_to_host();
        // }

        cells.take_step<relaxation_force, freeze_friction>(dt);

        // write the output
        // if (time_step % skip_step == 0 || time_step == epi_relax_time) {
        //     output.write_positions(cells);
        //     output.write_polarity(cells);
        //     output.write_property(type);
        //     output.write_property(freeze);
        // }
    }

    cells.copy_to_host();
    output.write_positions(cells);
    output.write_polarity(cells);
    output.write_property(type);

    // write down the mesh in the vtk file to compare it with the posterior
    // seeding
    mesh.write_vtk(output_tag);
    // write down the mesenchymal mesh in the vtk file to compare it with the
    // posterior filling
    mesh_mesench.write_vtk(output_tag + ".mesench");

    // Create a dummy mesh that depicts the x=0 plane, depicting the flank
    // boundary
    // Mesh wall;
    // min_point = mesh.get_minimum();
    // diagonal_vector = mesh.get_maximum() - min_point;
    // float3 A{0.f, 2 * min_point.y, 2 * min_point.z};
    // float3 B{0.f, 2 * min_point.y, 2 * (min_point.z + diagonal_vector.z)};
    // float3 C{0.f, 2 * (min_point.y + diagonal_vector.y), 2 * min_point.z};
    // float3 D{0.f, 2 * (min_point.y + diagonal_vector.y), 2 * (min_point.z +
    // diagonal_vector.z)}; Triangle ABC{A, B, C}; Triangle BCD{B, C, D};
    // wall.n_facets = 2;
    // wall.facets.push_back(ABC);
    // wall.facets.push_back(BCD);
    // wall.write_vtk(output_tag + ".wall");

    // for shape comparison purposes we write down the initial mesh as the
    // facets
    // centres and the cells epithelium in separate vtk files.

    std::cout << "writing mesh_T0" << std::endl;
    Solution<Cell, Grid_solver> mesh_T0{n_max};
    *mesh_T0.h_n = mesh.n_facets;
    fill_solver_w_mesh_no_flank(mesh, mesh_T0);
    Vtk_output output_mesh_T0{output_tag + ".mesh_T0"};
    output_mesh_T0.write_positions(mesh_T0);
    output_mesh_T0.write_polarity(mesh_T0);

    std::cout << "writing epi_T0" << std::endl;
    Solution<Cell, Grid_solver> epi_T0{n_max};
    *epi_T0.h_n = n_cells_total;
    fill_solver_w_epithelium(cells, epi_T0, type);
    Vtk_output output_epi_T0(output_tag + ".epi_T0");
    output_epi_T0.write_positions(epi_T0);
    output_epi_T0.write_polarity(epi_T0);

    // load the mesh for the optimal shape and process it with the same
    // parameters
    Mesh optimum_mesh{optimum_file_name};
    optimum_mesh.rescale(resc);
    optimum_mesh.rotate(0.0f, 0.0f, -0.2f);
    optimum_mesh.write_vtk(output_tag + ".optmesh");
    Solution<Cell, Grid_solver> optimum_mesh_TF{n_max};
    *optimum_mesh_TF.h_n = optimum_mesh.n_facets;
    fill_solver_w_mesh_no_flank(optimum_mesh, optimum_mesh_TF);
    Vtk_output output_opt_mesh_TF{output_tag + ".mesh_TF"};
    output_opt_mesh_TF.write_positions(optimum_mesh_TF);
    output_opt_mesh_TF.write_polarity(optimum_mesh_TF);

    std::cout << "DOOOOOOOOOOOOOOONE***************" << std::endl;

    return 0;
}
