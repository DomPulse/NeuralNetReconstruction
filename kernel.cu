﻿#include <iostream>
#include <vector>
#include <curand_kernel.h>
#include <random>
#include <chrono>
using namespace std;

template <typename T>
__global__ void synapse_current(T* curr, bool* fire, T* synapses, int num_neurons, int num_syns) 
{
    //should probably interchange the order of pre and post syn idx so i can just check if it fired once and avoid the loop if it didn't
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (fire[neuron_idx] == 1) 
    {
        for (int syn_idx = 0; syn_idx < num_syns; syn_idx++)
        {
            int tot_idx = (neuron_idx * num_syns + syn_idx) * 2; //remember this so i can actuall reference shit later
            int post_syn_idx = (int)synapses[tot_idx];
            curr[post_syn_idx] += synapses[tot_idx+1];
        }
    }
    
}

template <typename T>
__global__ void noisy_current(T* curr, bool* exin, unsigned long long seed, unsigned long long offset) {//yeah idk how the whoel T* thing works, it had a breakdown when i tried to pass a float and an int, crazy stuff
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(seed, neuron_idx, offset, &state);

    curr[neuron_idx] = (2 + 3 * (exin[neuron_idx] == 1)) * curand_normal(&state); //this is probably the slow down, should probably have 4 vector instead of 1 vector with 4x the length  
}

template <typename T>
__global__ void update_neuron(T* volt, T* rec, T* curr, bool* fire, bool* exin, T* fit_param)
{
    //indexed by taking the neuron_idx + some offset
    //neuron_idx+0 -> V
    //neuron_idx+1 -> u
    //neuron_idx+2 -> I
    //neuron_idx+3 -> jF
    //bc of this, neuron_idx counts by 4
    //same scheme for fit parameters
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
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

template <typename T>
__global__ void define_exin_array(T* temp_exin_array, unsigned long long seed, unsigned long long offset)
{
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
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

template <typename T>
__global__ void define_fit_params(T* temp_fit_param, unsigned long long seed, unsigned long long offset)
{
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
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

template <typename T>
__global__ void initialize_neurons(T* volt, T* rec, T* curr, bool* fire, bool* exin, T* fit_param)
{
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
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

template <typename T>
__global__ void define_synapses(T* rand_synapses, bool* exin, int num_neurons, int num_syns, unsigned long long seed, unsigned long long offset)
{
    //neurons are all given as pre_synaptic
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(seed, neuron_idx, offset, &state);

    for (int syn_idx = 0; syn_idx < num_syns; syn_idx++)
    {
        int tot_idx = (neuron_idx * num_syns + syn_idx) * 2; //remember this so i can actuall reference shit later
        rand_synapses[tot_idx] = (curand_uniform(&state) * num_neurons); //assigns the post synaptic index, far too lazy to make sure there are no repeats (really need to work on that)
        rand_synapses[tot_idx + 1] = (curand_uniform(&state)) * (exin[neuron_idx] * 2 - 1) * (1 - 0.5 * (exin[neuron_idx] == 1)); //assigns the actual synaptic weight

    }
}

int main() {
    unsigned long long seed = 1234;  // Random seed

    int num_neurons = 32768*32;
    num_neurons = 2048*64*2;
    int syns_per_neur = 1023;
    int sim_time{ 1000 };

    // Host vector to store the result
    vector<float> h_volt(num_neurons);
    vector<float> h_rec(num_neurons);
    vector<float> h_curr(num_neurons);
    vector<bool> h_fire(num_neurons);
    vector<float> h_fit_param(num_neurons);
    vector<float> h_synapses(num_neurons * syns_per_neur * 2);
    vector<bool> h_exin_array(num_neurons);

    // Launch kernel to generate random numbers on the GPU
    int threadsPerBlock = 32; //was formerly 256
    int blocksPerGrid = (num_neurons + threadsPerBlock - 1) / threadsPerBlock;

    // Device vector
    float* d_volt;
    float* d_rec;
    float* d_curr;
    bool* d_fire;
    float* d_fit_param;
    float* d_synapses;
    bool* d_exin_array;

    cudaMalloc((void**)&d_volt, num_neurons * sizeof(float));
    cudaMalloc((void**)&d_rec, num_neurons * sizeof(float));
    cudaMalloc((void**)&d_curr, num_neurons * sizeof(float));
    cudaMalloc((void**)&d_fire, num_neurons * sizeof(bool));
    cudaMalloc((void**)&d_fit_param, num_neurons * sizeof(float));
    cudaMalloc((void**)&d_synapses, num_neurons * syns_per_neur * 2 * sizeof(float));
    cudaMalloc((void**)&d_exin_array, num_neurons * sizeof(bool));

    //initializing data on gpu
    define_exin_array << <blocksPerGrid, threadsPerBlock >> > (d_exin_array, seed, 0.0);
    define_fit_params << <blocksPerGrid, threadsPerBlock >> > (d_fit_param, seed, 0.0);
    initialize_neurons << <blocksPerGrid, threadsPerBlock >> > (d_volt, d_rec, d_curr, d_fire, d_exin_array, d_fit_param);
    define_synapses << <blocksPerGrid, threadsPerBlock >> > (d_synapses, d_exin_array, num_neurons, syns_per_neur, seed, 0.0);

    //update the neurons with 1ms time step
    cout << "started" << endl;
    auto start = chrono::high_resolution_clock::now();
    for (int t = 0; t < sim_time; t++)
    {
        //take action brother!
        update_neuron << <blocksPerGrid, threadsPerBlock >> > (d_volt, d_rec, d_curr, d_fire, d_exin_array, d_fit_param);
        noisy_current << <blocksPerGrid, threadsPerBlock >> > (d_curr, d_exin_array, seed, 0.0);
        synapse_current << <blocksPerGrid, threadsPerBlock >> > (d_curr, d_fire, d_synapses, num_neurons, syns_per_neur);
        //yeah so now i just need the currents implimented so thats cool

    }
    

    // Copy the result back to the host
    cudaMemcpy(h_volt.data(), d_volt, num_neurons * sizeof(float), cudaMemcpyDeviceToHost);
    cout << h_volt[0] << endl;
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::seconds>(stop - start);
    cout << duration.count() << endl;
    cout << "done" << endl;

    // Cleanup
    cudaFree(d_volt);
    cudaFree(d_rec);
    cudaFree(d_curr);
    cudaFree(d_fire);
    cudaFree(d_synapses);
    cudaFree(d_exin_array);

    return 0;
}