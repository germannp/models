// Simulation of limb bud growth starting with realistic limb bud shape

// Command line arguments
// argv[1]=input file tag
// argv[2]=output file tag
// argv[3]=proliferation rate
// argv[4]=time steps

#include <curand_kernel.h>
#include <time.h>
#include <iostream>
#include <list>
#include <sstream>
#include <string>
#include <vector>

#include "../include/dtypes.cuh"
#include "../include/inits.cuh"
#include "../include/links.cuh"
#include "../include/polarity.cuh"
#include "../include/property.cuh"
#include "../include/solvers.cuh"
#include "../include/vtk.cuh"

const auto r_max = 1.0;
const auto r_min = 0.8;
const auto dt = 0.05f;
const auto n_max = 1200000;
const auto grid_size = 120;
const auto prots_per_cell = 1;
const auto protrusion_strength = 0.06f; // 0.2
const auto distal_strength = 0.20f; // 0.20 half that for ace of spades
const auto proximal_strength = 0.40f; // 0.40 half that for ace of spades
const auto r_protrusion = 2.0f;
const auto distal_threshold = 0.08f; // 0.08f;
const auto skip_step = 100;

const auto max_proliferation_rate = 0.0035f;
const auto n_time_steps = 1000;

enum Cell_types { mesenchyme, epithelium, aer };

__device__ Cell_types* d_type;
__device__ int* d_mes_nbs;
__device__ int* d_epi_nbs;
__device__ float* d_prolif_rate;
__device__ bool* d_is_distal;
__device__ bool* d_is_limb;
__device__ float* d_pd_clone;
__device__ float* d_ap_clone;
__device__ float* d_dv_clone;

Property<int> n_mes_nbs{n_max, "n_mes_nbs"};  // defining these here so function
Property<int> n_epi_nbs{n_max, "n_epi_nbs"};  // "neighbour_init" can see them

MAKE_PT(Cell, w, f, theta, phi);

__device__ Cell force(Cell Xi, Cell r, float dist, int i, int j)
{
    Cell dF{0};

    if (i == j) {
        // D_ASSERT(Xi.w >= 0);
        dF.w = -0.01 * (d_type[i] == mesenchyme) * Xi.w;
        dF.f = -0.01 * (d_type[i] == mesenchyme) * Xi.f;
        if (Xi.w < 0.f) dF.w = 0.f;
        if (Xi.f < 0.f) dF.f = 0.f;
        return dF;
    }

    if (dist > r_max) return dF;

    float F;
    if (d_type[i] == d_type[j]) {
        if (d_type[i] == mesenchyme)
            F = fmaxf(0.8 - dist, 0) * 2.f - fmaxf(dist - 0.8, 0);
        else
            F = fmaxf(0.8 - dist, 0) * 2.f - fmaxf(dist - 0.8, 0) * 2.f;
    } else if (d_type[i] > mesenchyme && d_type[j] > mesenchyme) {
        F = fmaxf(0.8 - dist, 0) * 2.f - fmaxf(dist - 0.8, 0) * 2.0f;
    } else {
        F = fmaxf(0.9 - dist, 0) * 2.f - fmaxf(dist - 0.9, 0) * 3.f;
    }
    dF.x = r.x * F / dist;
    dF.y = r.y * F / dist;
    dF.z = r.z * F / dist;

    dF.w = -r.w * (d_type[i] == mesenchyme) * 0.5f; //0.5
    dF.f = -r.f * (d_type[i] == mesenchyme) * 0.5f; //0.5

    //"diffusion" of spatial cues (intended to reduce noise)
    if(d_is_limb[i] == true and d_is_limb[j] == true){
        d_pd_clone[i] += (d_pd_clone[j] - d_pd_clone[i]) * 0.0005;
        d_ap_clone[i] += (d_ap_clone[j] - d_ap_clone[i]) * 0.0005;
        d_dv_clone[i] += (d_dv_clone[j] - d_dv_clone[i]) * 0.0005;
    }

    if (d_type[i] >= epithelium && d_type[j] >= epithelium)
        dF += bending_force(Xi, r, dist) * 0.1f;


    if (d_type[j] >= epithelium)
        atomicAdd(&d_epi_nbs[i], 1);
    else
        atomicAdd(&d_mes_nbs[i], 1);

    if (Xi.w < 0.f) Xi.w = 0.f;
    if (Xi.f < 0.f) Xi.f = 0.f;

    return dF;
}

__device__ Cell only_diffusion(Cell Xi, Cell r, float dist, int i, int j)
{
    Cell dF{0};

    if (i == j) {
        // D_ASSERT(Xi.w >= 0);
        dF.w = -0.01 * (d_type[i] == mesenchyme) * Xi.w;
        dF.f = -0.01 * (d_type[i] == mesenchyme) * Xi.f;
        if (Xi.w < 0.f) dF.w = 0.f;
        if (Xi.f < 0.f) dF.f = 0.f;
        return dF;
    }

    if (dist > r_max) return dF;

    dF.w = -r.w * (d_type[i] == mesenchyme) * 0.5f;
    dF.f = -r.f * (d_type[i] == mesenchyme) * 0.5f;


    if (Xi.w < 0.f) Xi.w = 0.f;
    if (Xi.f < 0.f) Xi.f = 0.f;

    return dF;
}

__device__ float friction(Cell Xi, Cell r, float dist, int i, int j)
{
    if (i == j) return 0;
    return 1;
}

__device__ void link_force(const Cell* __restrict__ d_X, const int a,
    const int b, const float strength, Cell* d_dX)
{
    float pd_strength = proximal_strength;
    if(d_X[a].f + d_X[b].f > 2.f * distal_threshold)
        pd_strength = distal_strength;

    auto r = d_X[a] - d_X[b];
    auto dist = norm3df(r.x, r.y, r.z);

    atomicAdd(&d_dX[a].x, -pd_strength * r.x / dist);
    atomicAdd(&d_dX[a].y, -pd_strength * r.y / dist);
    atomicAdd(&d_dX[a].z, -pd_strength * r.z / dist);
    atomicAdd(&d_dX[b].x, pd_strength * r.x / dist);
    atomicAdd(&d_dX[b].y, pd_strength * r.y / dist);
    atomicAdd(&d_dX[b].z, pd_strength * r.z / dist);
}

__global__ void update_protrusions(float dist_x_ratio, float dist_y_ratio,
    float prox_x_ratio, float prox_y_ratio, const int n_cells,
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
    if (d_is_limb[a] != d_is_limb[b]) return;

    auto new_r = d_X[a] - d_X[b];
    auto new_dist = norm3df(new_r.x, new_r.y, new_r.z);
    if (new_dist > r_protrusion) return;

    auto link = &d_link[a * prots_per_cell + i % prots_per_cell];
    auto not_initialized = link->a == link->b;
    auto old_r = d_X[link->a] - d_X[link->b];
    auto old_dist = norm3df(old_r.x, old_r.y, old_r.z);
    auto noise = curand_uniform(&d_state[i]);

    auto distal = (d_X[a].f + d_X[b].f) > 2 * distal_threshold;// distal_threshold;//0.025f;//0.20f; //0.025

    auto x_ratio = 0.5f; //0.25f;
    auto y_ratio = 0.75f; //0.50f;
    if(d_is_limb[a]){
        if(distal){
            x_ratio = dist_x_ratio; // 0.10
            y_ratio = dist_y_ratio; // 0.10
        }else{
            x_ratio = prox_x_ratio; // 0.0f
            y_ratio = prox_y_ratio; // 0.50f
        }
    }

    int x_y_or_z;
    if(curand_uniform(&d_state[i]) < x_ratio)
        x_y_or_z = 0;
    else if(curand_uniform(&d_state[i]) < y_ratio)
        x_y_or_z = 1;
    else
        x_y_or_z = 2;
    auto more_along_x = false;
    auto more_along_y = false;
    auto more_along_z = false;
    auto normal_to_z = false;
    more_along_x = fabs(new_r.x / new_dist) > fabs(old_r.x / old_dist) * (1.f - noise);
    more_along_y = fabs(new_r.y / new_dist) > fabs(old_r.y / old_dist) * (1.f - noise);
    more_along_z = fabs(new_r.z / new_dist) > fabs(old_r.z / old_dist) * (1.f - noise);
    normal_to_z = fabs(new_r.z / new_dist) < fabs(old_r.z / old_dist) * (1.f - noise);


    // auto old_pd = d_pd_clone[link->a] - d_pd_clone[link->b];
    // auto old_ap = d_ap_clone[link->a] - d_ap_clone[link->b];
    // auto old_dv = d_dv_clone[link->a] - d_dv_clone[link->b];
    // auto new_pd = d_pd_clone[a] - d_pd_clone[b];
    // auto new_ap = d_ap_clone[a] - d_ap_clone[b];
    // auto new_dv = d_dv_clone[a] - d_dv_clone[b];
    //
    // auto more_along_pd_clone = false;
    // auto more_along_ap_clone = false;
    // auto more_along_dv_clone = false;
    // more_along_pd_clone = fabs(new_pd / new_dist) < fabs(old_pd / old_dist) * (1.f - noise);
    // more_along_ap_clone = fabs(new_ap / new_dist) < fabs(old_ap / old_dist) * (1.f - noise);
    // more_along_dv_clone = fabs(new_dv / new_dist) < fabs(old_dv / old_dist) * (1.f - noise);

    auto normal_to_f =
        fabs(new_r.f / new_dist) < fabs(old_r.f / old_dist) * (1.f - noise);
    auto more_along_f =
        fabs(new_r.f / new_dist) > fabs(old_r.f / old_dist) * (1.f - noise);
    // auto more_along_w =
    //     fabs(new_r.w / new_dist) > fabs(old_r.w / old_dist) * (1.f - noise);

    if (distal){
        more_along_y = normal_to_z and normal_to_f;
        more_along_x = normal_to_z and more_along_f;
        // more_along_x = (new_r.f / new_dist) < (old_r.f / old_dist) * (1.f - noise);
        // more_along_pd_clone = normal_to_f;
        // more_along_ap_clone = more_along_f;
    }


    if (not_initialized or (x_y_or_z == 0 and more_along_x)
        or (x_y_or_z == 1 and more_along_y) or
        (x_y_or_z == 2 and more_along_z)) {
    // if (not_initialized or (x_y_or_z == 0 and more_along_ap_clone and more_along_dv_clone)
    //     or (x_y_or_z == 1 and more_along_pd_clone and more_along_dv_clone) or
    //     (x_y_or_z == 2 and more_along_pd_clone and more_along_ap_clone)) {
        link->a = a;
        link->b = b;
    }

}

__global__ void proliferate(float max_rate, float mean_distance, Cell* d_X,
    int* d_n_cells, curandState* d_state)
{
    D_ASSERT(*d_n_cells * max_rate <= n_max);
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *d_n_cells * (1 - max_rate))
        return;  // Dividing new cells is problematic!

    float rate = d_prolif_rate[i];
    // if(d_X[i].f > distal_threshold) // distal cells proliferate faster
    //     rate *= 1.25f;

    switch (d_type[i]) {
        case mesenchyme: {
            auto r = curand_uniform(&d_state[i]);
            if (r > rate) return;
            break;
        }
        default: {
            if (d_epi_nbs[i] > 7) return;
            if (d_mes_nbs[i] <= 0) return;
            auto r = curand_uniform(&d_state[i]);
            if (r > 2.5f * rate) return;  // 2.5
        }
    }

    auto n = atomicAdd(d_n_cells, 1);
    auto theta = curand_uniform(&d_state[i]) * 2 * M_PI;
    auto phi = curand_uniform(&d_state[i]) * M_PI;
    d_X[n].x = d_X[i].x + mean_distance / 4 * sinf(theta) * cosf(phi);
    d_X[n].y = d_X[i].y + mean_distance / 4 * sinf(theta) * sinf(phi);
    d_X[n].z = d_X[i].z + mean_distance / 4 * cosf(theta);
    if (d_type[i] == mesenchyme) {
        d_X[n].w = d_X[i].w / 2;
        d_X[i].w = d_X[i].w / 2;
        d_X[n].f = d_X[i].f / 2;
        d_X[i].f = d_X[i].f / 2;
        // d_X[n].w = d_X[i].w;
        // d_X[n].f = d_X[i].f;
    } else {
        d_X[n].w = d_X[i].w;
        d_X[n].f = d_X[i].f;
    }
    d_X[n].theta = d_X[i].theta;
    d_X[n].phi = d_X[i].phi;
    d_type[n] = d_type[i];
    d_prolif_rate[n] = d_prolif_rate[i];
    d_is_distal[n] = 0.0f;
    d_is_limb[n] = d_is_limb[i];
    d_pd_clone[n] = d_pd_clone[i];
    d_ap_clone[n] = d_ap_clone[i];
    d_dv_clone[n] = d_dv_clone[i];
}

__global__ void set_aer(float3 centroid, float pd_extension, Cell* d_X,
    int* d_n_cells)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > *d_n_cells) return;

    // if(d_X[i].z < centroid.z)
    //     d_prolif_rate[i] *= 0.75f;
    if(d_type[i] == epithelium or d_type[i] == aer) {
        if(d_X[i].x > centroid.x - pd_extension){
            // d_X[i].w = 1.0f;
            d_X[i].f = 0.0f;
            if(d_X[i].x > centroid.x - pd_extension and d_X[i].z < centroid.z + 4.4f
                and d_X[i].z > centroid.z - 4.4f) {
                d_type[i] = aer;
                d_X[i].f = 1.f;
            } else {
                d_type[i] = epithelium;
            }
        } else {
            d_type[i] = epithelium;
            d_X[i].f = 0.f;
            // d_X[i].w = 0.f;
        }
    }

}

__global__ void contain_aer(float3 centroid, Cell* d_X, int* d_n_cells)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i > *d_n_cells) return;

    if(d_type[i] == aer and
        (d_X[i].z > centroid.z + 4.4f or d_X[i].z < centroid.z - 4.4f)) {
        d_type[i] = epithelium;
        d_X[i].f = 0.f;
    }

    if(d_X[i].f > distal_threshold)
        d_is_distal[i] = true;
    else
        d_is_distal[i] = false;

}
// Double step solver means we have to initialise n_neibhbours before every
// step.
// This function is called before each step.
void neighbour_init(const Cell* __restrict__ d_X, Cell* d_dX)
{
    thrust::fill(thrust::device, n_epi_nbs.d_prop, n_epi_nbs.d_prop + n_max, 0);
    thrust::fill(thrust::device, n_mes_nbs.d_prop, n_mes_nbs.d_prop + n_max, 0);
}

template<typename Pt = float3, Link_force<Pt> force = linear_force<Pt>>
void link_forces_w_n_init(Links& links, const Pt* __restrict__ d_X, Pt* d_dX)
{
    thrust::fill(thrust::device, n_epi_nbs.d_prop, n_epi_nbs.d_prop + n_max, 0);
    thrust::fill(thrust::device, n_mes_nbs.d_prop, n_mes_nbs.d_prop + n_max, 0);
    link<Pt, force><<<(links.get_d_n() + 32 - 1) / 32, 32>>>(
        d_X, d_dX, links.d_link, links.get_d_n(), links.strength);
}

//*****************************************************************************

int main(int argc, char const* argv[])
{
    std::string file_name = argv[1];
    std::string codename = argv[2];
    std::string param1 = argv[3];
    std::string param2 = argv[4];
    std::string param3 = argv[5];
    std::string param4 = argv[6];

    std::string output_tag = codename + "_dt_" +
        std::to_string(distal_threshold) +
        "_drx_" + param1 + "_dry_" + param2 +
        "_prx_" + param3 + "_pry_" + param4;
    float dist_x_ratio = std::stof(param1);
    float dist_y_ratio = std::stof(param2);
    float prox_x_ratio = std::stof(param3);
    float prox_y_ratio = std::stof(param4);

    // float max_proliferation_rate = std::stof(argv[3]);
    // int n_time_steps = std::stoi(argv[4]);
    // int prolif_dist = std::stoi(argv[5]);

    // Load the initial conditions
    Vtk_input input(file_name);
    int n0 = input.n_points;
    Solution<Cell, Grid_solver> limb{n_max, grid_size};
    *limb.h_n = n0;
    input.read_positions(limb);
    input.read_polarity(limb);
    Property<Cell_types> type{n_max};
    cudaMemcpyToSymbol(d_type, &type.d_prop, sizeof(d_type));
    Property<int> intype{n_max};

    input.read_property(intype, "cell_type");  // we read it as an int, then we translate to
                                               // enum "Cell_types"
    for (int i = 0; i < n0; i++) {
        if (intype.h_prop[i] == 0)
            type.h_prop[i] = mesenchyme;
        else
            type.h_prop[i] = epithelium;
    }

    type.copy_to_device();

    std::cout << "initial ncells " << n0 << " nmax " << n_max << std::endl;

    cudaMemcpyToSymbol(d_mes_nbs, &n_mes_nbs.d_prop, sizeof(d_mes_nbs));
    cudaMemcpyToSymbol(d_epi_nbs, &n_epi_nbs.d_prop, sizeof(d_epi_nbs));

    // mark what is limb and what is flank
    Property<bool> is_limb{n_max, "is_limb"};
    cudaMemcpyToSymbol(
        d_is_limb, &is_limb.d_prop, sizeof(d_is_limb));
    for (int i = 0; i < n0; i++) {
        if(type.h_prop[i] == epithelium and limb.h_X[i].x > -5.f)
            limb.h_X[i].w = 1.0f;
        limb.h_X[i].f = 0.0f;
        if(limb.h_X[i].x > -2.0f and limb.h_X[i].y < 15.f){
            is_limb.h_prop[i] = true;
            // if(type.h_prop[i] == epithelium)
            //     limb.h_X[i].w = 1.f;
        }else
            is_limb.h_prop[i] = false;
    }
    is_limb.copy_to_device();

    float maximum = limb.h_X[0].x;
    int fixed;
    for (auto i = 1; i < n0; i++) {
        if (maximum < limb.h_X[i].x) {
            maximum = limb.h_X[i].x;
            fixed = i;
        }
    }
    float translate = 50.0 - limb.h_X[fixed].x;
    for (auto i = 0; i < n0; i++) {
        limb.h_X[i].x += translate;
    }
    limb.copy_to_device();

    limb.set_fixed(fixed);
    float3 X_fixed{limb.h_X[fixed].x, limb.h_X[fixed].y, limb.h_X[fixed].z};

    Links protrusions{n_max * prots_per_cell, protrusion_strength};
    protrusions.set_d_n(n0 * prots_per_cell);
    auto intercalation = std::bind(link_forces_w_n_init<Cell, link_force>,
        protrusions, std::placeholders::_1, std::placeholders::_2);

    Grid grid{n_max, grid_size};

    // determine cell-specific proliferation rates
    Property<float> prolif_rate{n_max, "prolif_rate"};
    cudaMemcpyToSymbol(
        d_prolif_rate, &prolif_rate.d_prop, sizeof(d_prolif_rate));
    Property<bool> is_distal{n_max, "is_distal"};
    cudaMemcpyToSymbol(
        d_is_distal, &is_distal.d_prop, sizeof(d_is_distal));
        for (int i = 0; i < n0; i++) {
            prolif_rate.h_prop[i] = max_proliferation_rate;
        }
    prolif_rate.copy_to_device();

    //set up clone-like tracking in the PD axis
    Property<float> pd_clone{n_max, "pd_clone"};
    cudaMemcpyToSymbol(
        d_pd_clone, &pd_clone.d_prop, sizeof(d_pd_clone));
        for (int i = 0; i < n0; i++) {
            if(is_limb.h_prop[i])
                pd_clone.h_prop[i] = limb.h_X[i].x;
            else
                pd_clone.h_prop[i] = 0.5 * limb.h_X[i].x;
                // pd_clone.h_prop[i] = 52.f;
        }
    pd_clone.copy_to_device();

    //set up clone-like tracking in the AP axis
    Property<float> ap_clone{n_max, "ap_clone"};
    cudaMemcpyToSymbol(
        d_ap_clone, &ap_clone.d_prop, sizeof(d_ap_clone));
        for (int i = 0; i < n0; i++) {
            if(is_limb.h_prop[i])
                ap_clone.h_prop[i] = limb.h_X[i].y + 20.f;
            else
                ap_clone.h_prop[i] = 0.5 * (limb.h_X[i].y + 20.f);
                // ap_clone.h_prop[i] = 40.f;
        }
    ap_clone.copy_to_device();

    //set up clone-like tracking in the DV axis
    Property<float> dv_clone{n_max, "dv_clone"};
    cudaMemcpyToSymbol(
        d_dv_clone, &dv_clone.d_prop, sizeof(d_dv_clone));
        for (int i = 0; i < n0; i++) {
            if(is_limb.h_prop[i])
                dv_clone.h_prop[i] = limb.h_X[i].z + 20.f;
            else
                dv_clone.h_prop[i] = 0.5 * (limb.h_X[i].z + 20.f);
                // dv_clone.h_prop[i] = 30.f;
        }
    dv_clone.copy_to_device();

    // State for proliferations
    curandState* d_state;
    cudaMalloc(&d_state, n_max * sizeof(curandState));
    auto seed = time(NULL);
    setup_rand_states<<<(n_max + 128 - 1) / 128, 128>>>(n_max, seed, d_state);

    // Calculate centroid (needed to dynamically control AER)
    // float3 centroid {0};
    // for (auto i = 0; i <= n0; i++) {
    //     centroid.x += limb.h_X[i].x;
    //     centroid.y += limb.h_X[i].y;
    //     centroid.z += limb.h_X[i].z;
    // }
    // centroid *= 1.f / float(n0);
    // std::cout<<"centroid pre "<<centroid.x<<" "<<centroid.y<<"
    // "<<centroid.z<<std::endl;


    std::cout << "n_time_steps " << n_time_steps << " write interval "
              << skip_step << std::endl;


    //restricts aer cells to a geometric rule
    float pd_extension = 5.0;//5//8.f;//+ 0.02 * float(time_step);
    set_aer<<<(limb.get_d_n() + 128 - 1) / 128, 128>>>(
        X_fixed, pd_extension, limb.d_X, limb.d_n);

    for (auto time_step = 0; time_step <= 500; time_step++) {
        limb.take_step<only_diffusion, friction>(dt);
    }

    Vtk_output limb_output{output_tag};
    for (auto time_step = 0; time_step <= n_time_steps; time_step++) {
        if (time_step % skip_step == 0 || time_step == n_time_steps) {
            limb.copy_to_host();
            protrusions.copy_to_host();
            type.copy_to_host();
            n_epi_nbs.copy_to_host();
            n_mes_nbs.copy_to_host();
            prolif_rate.copy_to_host();
            is_distal.copy_to_host();
            is_limb.copy_to_host();
            pd_clone.copy_to_host();
            ap_clone.copy_to_host();
            dv_clone.copy_to_host();
        }

        contain_aer<<<(limb.get_d_n() + 128 - 1) / 128, 128>>>(X_fixed, limb.d_X, limb.d_n);

        proliferate<<<(limb.get_d_n() + 128 - 1) / 128, 128>>>(
            max_proliferation_rate, r_min, limb.d_X, limb.d_n, d_state);
        protrusions.set_d_n(limb.get_d_n() * prots_per_cell);
        grid.build(limb, r_protrusion);
        update_protrusions<<<(protrusions.get_d_n() + 32 - 1) / 32, 32>>>(
            dist_x_ratio, dist_y_ratio, prox_x_ratio, prox_y_ratio,
            limb.get_d_n(), grid.d_grid, limb.d_X, protrusions.d_state,
            protrusions.d_link);


        limb.take_step<force, friction>(dt, intercalation);

        // write the output
        if (time_step % skip_step == 0 || time_step == n_time_steps) {
            limb_output.write_positions(limb);
            limb_output.write_links(protrusions);
            limb_output.write_polarity(limb);
            limb_output.write_field(limb, "WNT");
            limb_output.write_field(limb, "FGF", &Cell::f);
            limb_output.write_property(type);
            // limb_output.write_property(n_epi_nbs);
            // limb_output.write_property(n_mes_nbs);
            limb_output.write_property(prolif_rate);
            limb_output.write_property(is_distal);
            limb_output.write_property(is_limb);
            limb_output.write_property(pd_clone);
            limb_output.write_property(ap_clone);
            limb_output.write_property(dv_clone);
        }
    }

    // //float3 centroid {0};
    // centroid.x = 0.f;
    // centroid.y = 0.f;
    // centroid.z = 0.f;
    // for (auto i = 0; i <= n0; i++) {
    //     centroid.x += limb.h_X[i].x;
    //     centroid.y += limb.h_X[i].y;
    //     centroid.z += limb.h_X[i].z;
    // }
    // centroid *= 1.f / float(n0);
    // std::cout<<"centroid post "<<centroid.x<<" "<<centroid.y<<"
    // "<<centroid.z<<std::endl;

    // write down the limb epithelium for later shape comparison
    limb.copy_to_host();
    protrusions.copy_to_host();
    type.copy_to_host();
    Solution<Cell, Grid_solver> epi_Tf{n_max};
    int j = 0;
    for (int i = 0; i < *limb.h_n; i++) {
        if (type.h_prop[i] >= epithelium) {
            epi_Tf.h_X[j].x = limb.h_X[i].x;
            epi_Tf.h_X[j].y = limb.h_X[i].y;
            epi_Tf.h_X[j].z = limb.h_X[i].z;
            epi_Tf.h_X[j].phi = limb.h_X[i].phi;
            epi_Tf.h_X[j].theta = limb.h_X[i].theta;
            j++;
        }
    }
    *epi_Tf.h_n = j;
    Vtk_output epi_Tf_output{output_tag + ".shape"};
    epi_Tf_output.write_positions(epi_Tf);
    epi_Tf_output.write_polarity(epi_Tf);

    return 0;
}
