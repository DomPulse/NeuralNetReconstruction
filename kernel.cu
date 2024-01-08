#include <iostream>
#include <vector>
#include <curand_kernel.h>
#include <random>
#include <chrono>
#include <fstream>
using namespace std;

template <typename T>
__global__ void synapse_current(T* curr, bool* fire, T* syn_weights, int* post_syn_idx, int num_neurons, int num_syns)
{
    //this is now vestigial code :)
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_idx < num_neurons)
    {
        if (fire[neuron_idx] == 1)
        {
            for (int syn_idx = 0; syn_idx < num_syns; syn_idx++)
            {
                int tot_idx = neuron_idx * num_syns + syn_idx; //remember this so i can actuall reference shit later
                atomicAdd(&curr[post_syn_idx[tot_idx]], syn_weights[tot_idx]);
            }
        }
    }
}


template <typename T>
__global__ void psuedo_noisy_current(T* curr, bool* exin, T* start_current, int offset, int num_neurons)
{

    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_idx < num_neurons)
    {
        curr[neuron_idx] = (2 + (exin[neuron_idx] == 1) * 3) * start_current[((neuron_idx + offset) * (neuron_idx + offset)) % num_neurons];
    }
}

template <typename T>
__global__ void update_neuron(T* volt, T* rec, T* curr, bool* fire, bool* exin, T* fit_param, int num_neurons)
{
    //indexed by taking the neuron_idx + some offset
    //neuron_idx+0 -> V
    //neuron_idx+1 -> u
    //neuron_idx+2 -> I
    //neuron_idx+3 -> jF
    //bc of this, neuron_idx counts by 4
    //same scheme for fit parameters
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_idx < num_neurons)
    {
        float a = 0;
        float b = 0;
        float c = 0;
        float d = 0;
        if (exin[neuron_idx] == 0)
        {
            a = 0.02 + 0.08 * fit_param[neuron_idx];
            b = 0.25 - 0.05 * fit_param[neuron_idx];
            c = -65.0;
            d = 2;

        }
        else
        {
            a = 0.02;
            b = 0.2;
            c = -65.0 + 15 * fit_param[neuron_idx] * fit_param[neuron_idx];
            d = 8 - 6 * fit_param[neuron_idx] * fit_param[neuron_idx];
        }

        fire[neuron_idx] = 0;
        volt[neuron_idx] += 0.5 * (0.04 * volt[neuron_idx] * volt[neuron_idx] + 5 * volt[neuron_idx] + 140 - rec[neuron_idx] + curr[neuron_idx]);
        volt[neuron_idx] += 0.5 * (0.04 * volt[neuron_idx] * volt[neuron_idx] + 5 * volt[neuron_idx] + 140 - rec[neuron_idx] + curr[neuron_idx]);
        rec[neuron_idx] += a * (b * volt[neuron_idx] - rec[neuron_idx]);
        curr[neuron_idx] = 0;
        if (volt[neuron_idx] >= 30)
        {
            volt[neuron_idx] = c;
            rec[neuron_idx] += d;
            fire[neuron_idx] = 1;
        }
    }
    
}

template <typename T>
__global__ void define_exin_array(T* temp_exin_array, unsigned long long seed, unsigned long long offset, int num_neurons)
{
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_idx < num_neurons)
    {
        curandState state;
        curand_init(seed, neuron_idx, offset, &state);
        if (curand_uniform(&state) < 0.2) //this 0.2 is frac inhibit, i'm hardcoding it like a dumb dumb
        {
            temp_exin_array[neuron_idx] = 0;
        }
        else
        {
            temp_exin_array[neuron_idx] = 1;
        }
    }
}

template <typename T>
__global__ void define_start_curr(T* start_current, unsigned long long seed, unsigned long long offset, int num_neurons)
{
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_idx < num_neurons)
    {
        curandState state;
        curand_init(seed, neuron_idx, offset, &state);
        start_current[neuron_idx] = curand_normal(&state);
    }
}

template <typename T>
__global__ void define_fit_params(T* temp_fit_param, unsigned long long seed, unsigned long long offset, int num_neurons)
{
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_idx < num_neurons)
    {
        curandState state;
        curand_init(seed, neuron_idx, offset, &state);
        temp_fit_param[neuron_idx] = curand_uniform(&state);
        /*
        if (exin[neuron_idx] == 0)
        {
            float ri = curand_uniform(&state);
            temp_fit_param
            a[neuron_idx] = 0.02 + 0.08 * ri;
            b[neuron_idx] = 0.25 - 0.05 * ri;
            c[neuron_idx] = -65.0;
            d[neuron_idx] = 2;

        }
        else
        {
            float re = curand_uniform(&state);
            a[neuron_idx] = 0.02;
            b[neuron_idx] = 0.2;
            c[neuron_idx] = -65.0 + 15 * re * re;
            d[neuron_idx] = 8 - 6 * re * re;
        }
        */
    }
}

template <typename T>
__global__ void initialize_neurons(T* volt, T* rec, T* curr, bool* fire, bool* exin, T* fit_param, int num_neurons)
{
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_idx < num_neurons)
    {
        float b = 0.2;
        if (exin[neuron_idx] == 0)
        {
            b = 0.25 - 0.05 * fit_param[neuron_idx];
        }
        float u = -65.0 * b; //cringe double work around
        volt[neuron_idx] = -65.0;
        rec[neuron_idx] = u;
        curr[neuron_idx] = 0.0;
        fire[neuron_idx] = 0.0;
    }
}

template <typename T>
__global__ void define_synapses(T* rand_syn_weights, int* post_syn_idx, bool* exin, int num_neurons, int num_syns, unsigned long long seed, unsigned long long offset)
{
    //neurons are all given as pre_synaptic
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_idx < num_neurons)
    {
        curandState state;
        curand_init(seed, neuron_idx, offset, &state);

        for (int syn_idx = 0; syn_idx < num_syns; syn_idx++)
        {
            int tot_idx = neuron_idx * num_syns + syn_idx; //remember this so i can actuall reference shit later
            int potential_psi = curand(&state) % num_neurons;
            bool used_already = false;
            for (int j = 0; j < syn_idx; j++)
            {
                int compare_tot_idx = neuron_idx * num_syns + j;
                if (post_syn_idx[compare_tot_idx] == potential_psi)
                {
                    used_already = true;
                }
            }
            if (used_already)
            {
                syn_idx -= 1;
            }
            else
            {
                post_syn_idx[tot_idx] = potential_psi; //assigns the post synaptic index, far too lazy to make sure there are no repeats (really need to work on that)
                if (exin[neuron_idx])
                {
                    rand_syn_weights[tot_idx] = (curand_uniform(&state)) * 0.5; //assigns the actual synaptic weight
                }
                else
                {
                    rand_syn_weights[tot_idx] = (curand_uniform(&state)) * (-1); //assigns the actual synaptic weight
                }
            }
        }
    }
}

bool writeCSV(vector<vector<float>> mat)
{
    std::ofstream file;
    file.open("volts.csv", ios_base::app);
    for (auto& row : mat) {
        for (auto col : row)
            file << col << ',';
        file << '\n';
    }
    file.close();

    return true;
}

int main() {
    unsigned long long seed = 1234;  // Random seed

    int num_neurons = 1048576;
    //num_neurons = 1024;
    int syns_per_neur = 999;
    int sim_time{ 15000 };

    // Host vector to store the result
    vector<float> h_volt(num_neurons);
    vector<float> h_fit_param(num_neurons);
    vector<float> h_synapses(num_neurons * syns_per_neur * 2);
    vector<bool> h_exin_array(num_neurons);
    //vector<vector<float>> h_all_volts(sim_time, vector<float>(num_neurons));

    // Launch kernel to generate random numbers on the GPU
    int threadsPerBlock = 32; //was formerly 256
    int blocksPerGrid = (num_neurons + threadsPerBlock - 1) / threadsPerBlock;

    dim3 threads(threadsPerBlock);
    dim3 blocks(blocksPerGrid);

    // Device vector
    float* d_volt;
    float* d_rec;
    float* d_curr;
    bool* d_fire;
    float* d_fit_param;
    int* d_syn_idxs;
    float* d_syn_weights;
    bool* d_exin_array;
    float* psuedo_rand_curr;

    cudaMalloc((void**)&d_volt, num_neurons * sizeof(float));
    cudaMalloc((void**)&d_rec, num_neurons * sizeof(float));
    cudaMalloc((void**)&d_curr, num_neurons * sizeof(float));
    cudaMalloc((void**)&d_fire, num_neurons * sizeof(bool));
    cudaMalloc((void**)&d_fit_param, num_neurons * sizeof(float));
    cudaMalloc((void**)&d_syn_idxs, num_neurons * syns_per_neur * sizeof(int));
    cudaMalloc((void**)&d_syn_weights, num_neurons * syns_per_neur * sizeof(float));
    cudaMalloc((void**)&d_exin_array, num_neurons * sizeof(bool));
    cudaMalloc((void**)&psuedo_rand_curr, num_neurons * sizeof(float));

    //initializing data on gpu
    define_exin_array << <blocks, threads >> > (d_exin_array, seed, 0.0, num_neurons);
    define_start_curr << <blocks, threads >> > (psuedo_rand_curr, seed, 0.0, num_neurons);
    define_fit_params << <blocks, threads >> > (d_fit_param, seed, 0.0, num_neurons);
    initialize_neurons << <blocks, threads >> > (d_volt, d_rec, d_curr, d_fire, d_exin_array, d_fit_param, num_neurons);
    define_synapses << <blocks, threads >> > (d_syn_weights, d_syn_idxs, d_exin_array, num_neurons, syns_per_neur, seed, 0.0);

    //update the neurons with 1ms time step
    cout << "started" << endl;
    auto start = chrono::high_resolution_clock::now();
    for (int t = 0; t < sim_time; t++)
    {
        //take action brother!
        update_neuron << <blocks, threads >> > (d_volt, d_rec, d_curr, d_fire, d_exin_array, d_fit_param, num_neurons);

        psuedo_noisy_current << <blocks, threads >> > (d_curr, d_exin_array, psuedo_rand_curr, t, num_neurons);
        synapse_current << <blocks, threads >> > (d_curr, d_fire, d_syn_weights, d_syn_idxs, num_neurons, syns_per_neur);
 
        //cudaMemcpy(h_volt.data(), d_volt, num_neurons * sizeof(float), cudaMemcpyDeviceToHost);
        //h_all_volts[t] = h_volt;
    }


    // Copy the result back to the host
    cudaMemcpy(h_volt.data(), d_volt, num_neurons * sizeof(float), cudaMemcpyDeviceToHost);
    cout << h_volt[0] << endl;
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::seconds>(stop - start);
    cout << duration.count() << endl;
    cout << "done" << endl;
    //writeCSV(h_all_volts);

    // Cleanup
    cudaFree(d_volt);
    cudaFree(d_rec);
    cudaFree(d_curr);
    cudaFree(d_fire);
    cudaFree(d_syn_idxs);
    cudaFree(d_syn_weights);
    cudaFree(d_exin_array);

    return 0;
}