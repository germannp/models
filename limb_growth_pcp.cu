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
// const auto distal_strength = 0.075f;  //0.15
// const auto proximal_strength = 0.20f; //0.30
const auto flank_strength = 0.10f;
const auto r_protrusion = 2.0f;
const auto distal_threshold = 0.25f;
const auto max_proliferation_rate = 0.0030f; //0.0030f;
const auto clone_ratio = 0.1f;

const auto n_time_steps = 1000;
const auto skip_step = 100;

enum Cell_types { mesenchyme, epithelium, aer };

__device__ Cell_types* d_type;
__device__ int* d_mes_nbs;
__device__ int* d_epi_nbs;
__device__ float* d_prolif_rate;
__device__ bool* d_is_dorsal;
__device__ bool* d_is_distal;
__device__ bool* d_is_limb;
__device__ float* d_pd_clone;
__device__ float* d_ap_clone;
__device__ float* d_dv_clone;
__device__ int* d_sparse_clone;

Property<int> n_mes_nbs{n_max, "n_mes_nbs"};  // defining these here so function
Property<int> n_epi_nbs{n_max, "n_epi_nbs"};  // "neighbour_init" can see them

MAKE_PT(Cell, w, f, theta, phi);

__device__ Cell force(Cell Xi, Cell r, float dist, int i, int j)
{
    Cell dF{0};

    if (i == j) {
        // linear gradient diffusion
        dF.w = -0.01 * (d_type[i] == mesenchyme) * Xi.w;
        dF.f = -0.01 * (d_type[i] == mesenchyme) * Xi.f;
        // if (Xi.w <= 0.f or !d_is_limb[i]) dF.w = 0.f;
        if (Xi.f <= 0.f or !d_is_limb[i]) dF.f = 0.f;
        // if (Xi.w < 0.f or !d_is_limb[i]) Xi.w = 0.f;
        if (Xi.f < 0.f or !d_is_limb[i]) Xi.f = 0.f;
        return dF;
    }

    if (dist > r_max) return dF;

    // linear gradient diffusion
    if(r.f<0.f)
    dF.f = 0.05 * (Xi.f -r.f) * (d_type[i] == mesenchyme);
    else if(r.f>0.f)
    dF.f = -0.05 * Xi.f * (d_type[i] == mesenchyme);

    dF.w = -r.w * (d_type[i] == mesenchyme) * 0.5f; //0.5
    // dF.f = -r.f * (d_type[i] == mesenchyme) * 0.5f; //0.5

    if (d_type[i] == mesenchyme and !d_is_limb[i] and d_is_limb[j]) return dF;


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


    //"diffusion" of spatial cues (intended to reduce noise)
    // if(d_is_limb[i] == true and d_is_limb[j] == true){
    //     d_pd_clone[i] += (d_pd_clone[j] - d_pd_clone[i]) * 0.0005;
    //     d_ap_clone[i] += (d_ap_clone[j] - d_ap_clone[i]) * 0.0005;
    //     d_dv_clone[i] += (d_dv_clone[j] - d_dv_clone[i]) * 0.0005;
    // }

    if (d_type[i] >= epithelium && d_type[j] >= epithelium)
        dF += bending_force(Xi, r, dist) * 0.1f;

    if(d_is_limb[i] and d_type[i] == mesenchyme and d_type[j] == mesenchyme){
        // U_WNT = - ΣXj.w*(n_i . r_ij/r)^2/2 to bias along w
        Polarity rhat{acosf(-r.z / dist), atan2(-r.y, -r.x)};
        // if(d_is_distal[i]) {
            dF -= fabs(r.w / Xi.w) * bidirectional_polarization_force(Xi, rhat);
            dF -= fabs(r.f / Xi.f) * bidirectional_polarization_force(Xi, rhat);
        // } else {
        //     dF -= fabs(r.w / Xi.w) * bidirectional_polarization_force(Xi, rhat);
        //     dF -= fabs(r.f / Xi.f) * bidirectional_polarization_force(Xi, rhat);
        // }
        // dF.theta = 0.f;
    }

    if (d_type[j] >= epithelium)
        atomicAdd(&d_epi_nbs[i], 1);
    else
        atomicAdd(&d_mes_nbs[i], 1);

        // if (Xi.w < 0.f or !d_is_limb[i]) dF.w = 0.f;
        if (Xi.f < 0.f or !d_is_limb[i]) dF.f = 0.f;
        // if (Xi.w < 0.f or !d_is_limb[i]) Xi.w = 0.f;
        if (Xi.f < 0.f or !d_is_limb[i]) Xi.f = 0.f;

        if (Xi.w > 2.f) dF.w = 0.f;
        if (Xi.f > 2.f) dF.f = 0.f;
        if (Xi.w > 2.f) Xi.w = 2.f;
        if (Xi.f > 2.f) Xi.f = 2.f;

    return dF;
}

__device__ Cell only_diffusion(Cell Xi, Cell r, float dist, int i, int j)
{
    Cell dF{0};

    if (i == j) {

        // linear gradient diffusion

        dF.w = -0.01 * (d_type[i] == mesenchyme) * Xi.w;
        // dF.f = -0.01 * (d_type[i] == mesenchyme) * Xi.f;
        // if (Xi.w <= 0.f or !d_is_limb[i]) dF.w = 0.f;
        if (Xi.f <= 0.f or !d_is_limb[i]) dF.f = 0.f;
        // if (Xi.w < 0.f or !d_is_limb[i]) Xi.w = 0.f;
        if (Xi.f < 0.f or !d_is_limb[i]) Xi.f = 0.f;
        return dF;
    }

    if (dist > r_max) return dF;

    if(r.f<0.f)
        dF.f = 0.05 * (Xi.f -r.f) * (d_type[i] == mesenchyme);
    else if(r.f>0.f)
        dF.f = -0.05 * Xi.f * (d_type[i] == mesenchyme);

    dF.w = -r.w * (d_type[i] == mesenchyme) * 0.5f;

    // if (Xi.w < 0.f or !d_is_limb[i]) dF.w = 0.f;
    if (Xi.f < 0.f or !d_is_limb[i]) dF.f = 0.f;
    // if (Xi.w < 0.f or !d_is_limb[i]) Xi.w = 0.f;
    if (Xi.f < 0.f or !d_is_limb[i]) Xi.f = 0.f;

    if (Xi.w > 2.f) dF.w = 0.f;
    if (Xi.f > 2.f) dF.f = 0.f;
    if (Xi.w > 2.f) Xi.w = 2.f;
    if (Xi.f > 2.f) Xi.f = 2.f;


    return dF;
}

__device__ float friction(Cell Xi, Cell r, float dist, int i, int j)
{
    if (i == j) return 0;
    return 1;
}

__device__ void link_force(const Cell* __restrict__ d_X, const int a,
    const int b, const float* d_strength, Cell* d_dX)
{

    // if(d_X[a].f + d_X[b].f > 2 * distal_threshold)
    //     pd_strength = 2.f * distal_strength;
    float pd_strength;
    if (!d_is_limb[a])
        pd_strength = 2.f * flank_strength;
    else
        pd_strength = d_strength[a] + d_strength[b];

    // if(d_is_dorsal[a])
    //     pd_strength *= 1.2;
    // else
    //     pd_strength *= 0.8;

    auto r = d_X[a] - d_X[b];
    auto dist = norm3df(r.x, r.y, r.z);

    atomicAdd(&d_dX[a].x, -pd_strength * r.x / dist);
    atomicAdd(&d_dX[a].y, -pd_strength * r.y / dist);
    atomicAdd(&d_dX[a].z, -pd_strength * r.z / dist);
    atomicAdd(&d_dX[b].x, pd_strength * r.x / dist);
    atomicAdd(&d_dX[b].y, pd_strength * r.y / dist);
    atomicAdd(&d_dX[b].z, pd_strength * r.z / dist);
}

__global__ void update_protrusions(float dist_y_ratio,
    float prox_y_ratio, float dist_strength, float prox_strength, const int n_cells,
    const Grid* __restrict__ d_grid, const Cell* __restrict d_X,
    curandState* d_state, Link* d_link, float* d_strength)
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

    auto distal = d_X[a].f + d_X[b].f > 2 * distal_threshold;


    auto x_ratio = 0.25f; //0.25f;
    auto y_ratio = 0.75f; //0.50f;
    if(d_is_limb[a]){
        if(distal){
            x_ratio = 0.0f;
            y_ratio = dist_y_ratio;
        }else{
            x_ratio = 0.0f;
            y_ratio = prox_y_ratio;
        }
    }

    int x_y_or_z;
    float dice = curand_uniform(&d_state[i]);
    if(dice < x_ratio)
        x_y_or_z = 0;
    else if(dice < y_ratio)
        x_y_or_z = 1;
    else
        x_y_or_z = 2;

    auto more_along_x = false;
    auto more_along_y = false;
    auto more_along_z = false;
    more_along_x = fabs(new_r.x / new_dist) > fabs(old_r.x / old_dist) * (1.f - noise);
    more_along_y = fabs(new_r.y / new_dist) > fabs(old_r.y / old_dist) * (1.f - noise);
    more_along_z = fabs(new_r.z / new_dist) > fabs(old_r.z / old_dist) * (1.f - noise);

    float3 polarisation{sinf(d_X[a].theta) * cosf(d_X[a].phi),
        sinf(d_X[a].theta) * sinf(d_X[a].phi),
        cosf(d_X[a].theta)};
    auto old_dotprod = (old_r.x * polarisation.x + old_r.y * polarisation.y +
        old_r.z * polarisation.z) / old_dist;
    auto new_dotprod = (new_r.x * polarisation.x + new_r.y * polarisation.y +
        new_r.z * polarisation.z) / new_dist;

    auto more_along_polarity = fabs(old_dotprod) < fabs(new_dotprod) * (1.f - noise);
    auto normal_to_polarity = fabs(old_dotprod) > fabs(new_dotprod) * (1.f - noise);

    auto normal_to_f =
        fabs(new_r.f / new_dist) < fabs(old_r.f / old_dist) * (1.f - noise);
    auto more_along_f =
        fabs(new_r.f / new_dist) > fabs(old_r.f / old_dist) * (1.f - noise);

    auto normal_to_w =
        fabs(new_r.w / new_dist) < fabs(old_r.w / old_dist) * (1.f - noise);
    auto more_along_w =
        fabs(new_r.w / new_dist) > fabs(old_r.w / old_dist) * (1.f - noise);

    if(d_is_limb[i]){
        more_along_y = more_along_polarity;
        more_along_z = more_along_w;
    }

    if (distal){
        d_is_distal[a] = true;
        d_strength[a] = dist_strength;
    } else {
        d_is_distal[a] = false;
        d_strength[a] = prox_strength;
    }

    if (not_initialized or (x_y_or_z == 0 and more_along_x)
        or (x_y_or_z == 1 and more_along_y) or
        (x_y_or_z == 2 and more_along_z)) {
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
    // if(d_is_limb[i] and !d_is_distal[i])
    //     rate *= 0.75f;

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
        // d_X[n].f = d_X[i].f / 2;
        // d_X[i].f = d_X[i].f / 2;
    } else {
        d_X[n].w = d_X[i].w;
    }
    d_X[n].f = d_X[i].f;
    d_X[n].theta = d_X[i].theta;
    d_X[n].phi = d_X[i].phi;
    d_type[n] = d_type[i];
    d_prolif_rate[n] = d_prolif_rate[i];
    d_is_distal[n] = false;
    d_is_dorsal[n] = d_is_dorsal[i];
    d_is_limb[n] = d_is_limb[i];
    d_pd_clone[n] = d_pd_clone[i];
    d_ap_clone[n] = d_ap_clone[i];
    d_dv_clone[n] = d_dv_clone[i];
    d_sparse_clone[n] = d_sparse_clone[i];
}

__global__ void set_aer(float3 centroid, float pd_extension, Cell* d_X,
    int* d_n_cells)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > *d_n_cells) return;

    if(d_X[i].z > centroid.z + 0.4f)
        d_is_dorsal[i] = true;

    if(d_type[i] == epithelium) {
            // d_X[i].f = 0.0f;
        if(d_X[i].x > centroid.x - pd_extension and d_X[i].z < centroid.z + 4.4f //0.4 //4.4
            and d_X[i].z > centroid.z - 4.4f) { //8.4 //4.4
            d_type[i] = aer;
            d_X[i].f = 1.f;
            if(d_is_dorsal[i])
                d_X[i].w = 1.0f;
            else
                d_X[i].w = 0.f;
        } else {
            d_type[i] = epithelium;
            d_X[i].f = 0.f;
            if(d_X[i].x > centroid.x - 18.f and d_is_dorsal[i]) // wnt gradient only dorsal
            // if(d_X[i].x > centroid.x - 18.f) // wnt gradient throughout epi.
                d_X[i].w = 1.0f;
        }
    }

}

__global__ void contain_aer(float3 centroid, Cell* d_X, int* d_n_cells, float progress)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i > *d_n_cells) return;


    float ventral_boundary = 4.4f;// 4.4 - 4.f * progress;

    if(d_type[i] == aer and
        (d_X[i].z > centroid.z + 4.4f or d_X[i].z < centroid.z - ventral_boundary)) {
        d_type[i] = epithelium;
        d_X[i].f = 0.f;
        if(d_is_dorsal[i]) // wnt gradient only dorsal
            d_X[i].w = 1.f;
    }

}
// Double step solver means we have to initialise n_neighbours before every
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
        d_X, d_dX, links.d_link, links.get_d_n(), links.d_strength);
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
        "_dry_" + param1 + "_pry_" + param2 +
        "_ds_" + param3 + "_ps_" + param4;
    float dist_y_ratio = std::stof(param1);
    float prox_y_ratio = std::stof(param2);
    float distal_strength = std::stof(param3);
    float proximal_strength = std::stof(param4);


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

    float y_min = -12.5f;
    float y_max = 15.f;
    float z_min = -6.f;
    float z_max = 9.f;

    for (int i = 0; i < n0; i++) {
        if(limb.h_X[i].y > y_min and limb.h_X[i].y < y_max and
            limb.h_X[i].z > z_min and limb.h_X[i].z < z_max){
            if((type.h_prop[i] == mesenchyme and limb.h_X[i].x < -2.0f) or //-6
                (type.h_prop[i] == epithelium and limb.h_X[i].x < -4.0f))  //-4
                is_limb.h_prop[i] = false;
            else
                is_limb.h_prop[i] = true;
        }else
            is_limb.h_prop[i] = false;

        limb.h_X[i].theta = M_PI / 2.f;
        limb.h_X[i].phi = 2. * M_PI * rand() / (RAND_MAX + 1.);
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

    Property<bool> is_dorsal{n_max, "is_dorsal"};
    cudaMemcpyToSymbol(
        d_is_dorsal, &is_dorsal.d_prop, sizeof(d_is_dorsal));

    for (int i = 0; i < n0; i++) {
        prolif_rate.h_prop[i] = max_proliferation_rate;
        is_distal.h_prop[i] = false;
        is_dorsal.h_prop[i] = false;
    }
    prolif_rate.copy_to_device();
    is_distal.copy_to_device();
    is_dorsal.copy_to_device();

    //set up clone-like tracking (gradient-like labelling)
    Property<float> pd_clone{n_max, "pd_clone"};
    cudaMemcpyToSymbol(
        d_pd_clone, &pd_clone.d_prop, sizeof(d_pd_clone));
    Property<float> ap_clone{n_max, "ap_clone"};
    cudaMemcpyToSymbol(
        d_ap_clone, &ap_clone.d_prop, sizeof(d_ap_clone));
    Property<float> dv_clone{n_max, "dv_clone"};
    cudaMemcpyToSymbol(
        d_dv_clone, &dv_clone.d_prop, sizeof(d_dv_clone));

    //set up clone-like tracking (sparse labelling)
    Property<int> sparse_clone{n_max, "sparse_clone"};
    cudaMemcpyToSymbol(
        d_sparse_clone, &sparse_clone.d_prop, sizeof(d_sparse_clone));

    //spheric clone
    float cx = 42.0f;
    float cy = 10.0f;
    float cz = 0.0f;
    float radius = 2.5f;

    auto n_clones = 0;
    for (int i = 0; i < n0; i++) {
        if(is_limb.h_prop[i]){
            pd_clone.h_prop[i] = limb.h_X[i].x;
            ap_clone.h_prop[i] = limb.h_X[i].y + 20.f;
            dv_clone.h_prop[i] = limb.h_X[i].z + 20.f;
            //point clones
            // if(rand() / (RAND_MAX + 1.) < clone_ratio){
            //     n_clones++;
            //     sparse_clone.h_prop[i] = n_clones;
            // } else
            //     sparse_clone.h_prop[i] = 0;
            //spheric clone
            if(pow(cx - limb.h_X[i].x, 2) + pow(cy - limb.h_X[i].y, 2)
                + pow(cz - limb.h_X[i].z, 2) < radius*radius)
                sparse_clone.h_prop[i] = 1;

        }else{
            pd_clone.h_prop[i] = 0.5 * limb.h_X[i].x;
            ap_clone.h_prop[i] = 0.5 * (limb.h_X[i].y + 20.f);
            dv_clone.h_prop[i] = 0.5 * (limb.h_X[i].z + 20.f);
            // pd_clone.h_prop[i] = 52.f;
        }
    }
    pd_clone.copy_to_device();
    ap_clone.copy_to_device();
    dv_clone.copy_to_device();
    sparse_clone.copy_to_device();

    std::cout<<"sparse clone number "<<n_clones<<std::endl;


    // State for proliferations
    curandState* d_state;
    cudaMalloc(&d_state, n_max * sizeof(curandState));
    auto seed = time(NULL);
    setup_rand_states<<<(n_max + 128 - 1) / 128, 128>>>(n_max, seed, d_state);

    std::cout << "n_time_steps " << n_time_steps << " write interval "
              << skip_step << std::endl;

    //restricts aer cells to a geometric rule
    float pd_extension = 5.0;//5//8.f;//+ 0.02 * float(time_step);
    set_aer<<<(limb.get_d_n() + 128 - 1) / 128, 128>>>(
        X_fixed, pd_extension, limb.d_X, limb.d_n);

    std::cout<<"setting up initial gradients"<<std::endl;
    for (auto time_step = 0; time_step <= 1000; time_step++) {
        limb.take_step<only_diffusion, friction>(dt);
    }
    std::cout<<"initial gradients done"<<std::endl;

    Vtk_output limb_output{output_tag};
    for (auto time_step = 0; time_step <= n_time_steps; time_step++) {
        if (time_step % skip_step == 0 || time_step == n_time_steps) {
            limb.copy_to_host();
            protrusions.copy_to_host();
            type.copy_to_host();
            // n_epi_nbs.copy_to_host();
            // n_mes_nbs.copy_to_host();
            prolif_rate.copy_to_host();
            is_distal.copy_to_host();
            is_dorsal.copy_to_host();
            is_limb.copy_to_host();
            pd_clone.copy_to_host();
            ap_clone.copy_to_host();
            dv_clone.copy_to_host();
            sparse_clone.copy_to_host();
        }

        contain_aer<<<(limb.get_d_n() + 128 - 1) / 128, 128>>>(X_fixed, limb.d_X, limb.d_n, float(time_step)/float(n_time_steps));

        proliferate<<<(limb.get_d_n() + 128 - 1) / 128, 128>>>(
            max_proliferation_rate, r_min, limb.d_X, limb.d_n, d_state);
        protrusions.set_d_n(limb.get_d_n() * prots_per_cell);
        grid.build(limb, r_protrusion);
        update_protrusions<<<(protrusions.get_d_n() + 32 - 1) / 32, 32>>>(
            dist_y_ratio, prox_y_ratio, distal_strength, proximal_strength,
            limb.get_d_n(), grid.d_grid, limb.d_X, protrusions.d_state,
            protrusions.d_link, protrusions.d_strength);

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
            limb_output.write_property(is_dorsal);
            limb_output.write_property(is_limb);
            limb_output.write_property(pd_clone);
            limb_output.write_property(ap_clone);
            limb_output.write_property(dv_clone);
            limb_output.write_property(sparse_clone);
        }
    }

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
