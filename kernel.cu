#include <iostream>
#include <vector>
#include <curand_kernel.h>
#include <random>
#include <chrono>
using namespace std;

template <typename T>
__global__ void synapse_current(T* curr, T*fire, T* synapses, int num_neurons) {//yeah we be working on its
    //should probably interchange the order of pre and post syn idx so i can just check if it fired once and avoid the loop if it didn't
    int post_syn_idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int pre_syn_idx = 0; pre_syn_idx < num_neurons; pre_syn_idx++)
    {
        int tot_idx = post_syn_idx * num_neurons + pre_syn_idx;
        curr[post_syn_idx] += synapses[tot_idx] * fire[pre_syn_idx];
    }
}

template <typename T>
__global__ void noisy_current(T* curr, int* exin, unsigned long long seed, unsigned long long offset) {//yeah idk how the whoel T* thing works, it had a breakdown when i tried to pass a float and an int, crazy stuff
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(seed, neuron_idx, offset, &state);

    curr[neuron_idx] = (2 + 3 * (exin[neuron_idx] == 1)) * curand_normal(&state); //this is probably the slow down, should probably have 4 vector instead of 1 vector with 4x the length  
}

template <typename T>
__global__ void update_neuron(T* volt, T* rec, T* curr, T* fire, T* a, T* b, T* c, T* d)
{
    //indexed by taking the neuron_idx + some offset
    //neuron_idx+0 -> V
    //neuron_idx+1 -> u
    //neuron_idx+2 -> I
    //neuron_idx+3 -> jF
    //bc of this, neuron_idx counts by 4
    //same scheme for fit parameters
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    fire[neuron_idx] = 0;
    volt[neuron_idx] += 0.5 * (0.04 * volt[neuron_idx] * volt[neuron_idx] + 5 * volt[neuron_idx] + 140 - rec[neuron_idx] + curr[neuron_idx]);
    volt[neuron_idx] += 0.5 * (0.04 * volt[neuron_idx] * volt[neuron_idx] + 5 * volt[neuron_idx] + 140 - rec[neuron_idx] + curr[neuron_idx]);
    rec[neuron_idx] += a[neuron_idx] * (b[neuron_idx] * volt[neuron_idx] - rec[neuron_idx]);
    curr[neuron_idx] = 0;
    if (volt[neuron_idx] >= 30)
    {
        volt[neuron_idx] = c[neuron_idx];
        rec[neuron_idx] += d[neuron_idx];
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
        temp_exin_array[neuron_idx] = -1;
    }
    else
    {
        temp_exin_array[neuron_idx] = 1;
    }
}

template <typename T>
__global__ void define_fit_params(T* a, T*b, T*c, T*d, int* exin, unsigned long long seed, unsigned long long offset)
{
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(seed, neuron_idx, offset, &state);
    if (exin[neuron_idx] == -1)
    {
        float ri = curand_uniform(&state);
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
}

template <typename T>
__global__ void initialize_neurons(T* volt, T* rec, T* curr, T* fire, T*b)
{
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    float u = -65.0 * b[neuron_idx]; //cringe double work around
    volt[neuron_idx] = -65.0;
    rec[neuron_idx] = u;
    curr[neuron_idx] = 0.0;
    fire[neuron_idx] = 0.0;
}

template <typename T>
__global__ void define_synapses(T* rand_synapses, int* exin, int num_neurons, float frac_conect, unsigned long long seed, unsigned long long offset)
{
    int post_syn_idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(seed, post_syn_idx, offset, &state);
    for (int pre_syn_idx = 0; pre_syn_idx < num_neurons; pre_syn_idx++)
    {
        int tot_idx = post_syn_idx * num_neurons + pre_syn_idx; //remember this so i can actuall reference shit later
        rand_synapses[tot_idx] = (curand_uniform(&state)) * exin[pre_syn_idx] * (1 - 0.5 * (exin[pre_syn_idx] == 1));
        if (post_syn_idx == pre_syn_idx || curand_uniform(&state) > frac_conect)
        {
            rand_synapses[tot_idx] = 0;
        }
    }
}

int main() {
    unsigned long long seed = 1234;  // Random seed

    int num_neurons = 1024;
    int sim_time{ 1000 };
    float frac_inhib{ 0.2 };

    // Host vector to store the result
    vector<float> h_volt(num_neurons);
    vector<float> h_rec(num_neurons);
    vector<float> h_curr(num_neurons);
    vector<float> h_fire(num_neurons);
    vector<float> h_a(num_neurons);
    vector<float> h_b(num_neurons);
    vector<float> h_c(num_neurons);
    vector<float> h_d(num_neurons);
    vector<float> h_synapses(num_neurons * num_neurons);
    vector<int> h_exin_array(num_neurons);

    // Launch kernel to generate random numbers on the GPU
    int threadsPerBlock = 256; //was formerly 256
    int blocksPerGrid = (num_neurons + threadsPerBlock - 1) / threadsPerBlock;

    // Device vector
    float* d_volt;
    float* d_rec;
    float* d_curr;
    float* d_fire;
    float* d_a;
    float* d_b;
    float* d_c;
    float* d_d;
    float* d_synapses;
    int* d_exin_array;

    cudaMalloc((void**)&d_volt, num_neurons * sizeof(float));
    cudaMalloc((void**)&d_rec, num_neurons * sizeof(float));
    cudaMalloc((void**)&d_curr, num_neurons * sizeof(float));
    cudaMalloc((void**)&d_fire, num_neurons * sizeof(float));
    cudaMalloc((void**)&d_a, num_neurons * sizeof(float));
    cudaMalloc((void**)&d_b, num_neurons * sizeof(float));
    cudaMalloc((void**)&d_c, num_neurons * sizeof(float));
    cudaMalloc((void**)&d_d, num_neurons * sizeof(float));
    cudaMalloc((void**)&d_synapses, num_neurons * num_neurons * sizeof(float));
    cudaMalloc((void**)&d_exin_array, num_neurons * sizeof(int));

    //initializing data on gpu
    define_exin_array << <blocksPerGrid, threadsPerBlock >> > (d_exin_array, seed, 0.0);
    define_fit_params << <blocksPerGrid, threadsPerBlock >> > (d_a, d_b, d_c, d_d, d_exin_array, seed, 0.0);
    initialize_neurons << <blocksPerGrid, threadsPerBlock >> > (d_volt, d_rec, d_curr, d_fire, d_b);
    float frac_conect = 1024/num_neurons;
    define_synapses << <blocksPerGrid, threadsPerBlock >> > (d_synapses, d_exin_array, num_neurons, frac_conect, seed, 0.0);

    //update the neurons with 1ms time step
    cout << "started" << endl;
    auto start = chrono::high_resolution_clock::now();
    for (int t = 0; t < sim_time; t++)
    {
        //take action brother!
        update_neuron << <blocksPerGrid, threadsPerBlock >> > (d_volt, d_rec, d_curr, d_fire, d_a, d_b, d_c, d_d);
        noisy_current << <blocksPerGrid, threadsPerBlock >> > (d_curr, d_exin_array, seed, 0.0);
        synapse_current << <blocksPerGrid, threadsPerBlock >> > (d_curr, d_fire, d_synapses, num_neurons);
        //yeah so now i just need the currents implimented so thats cool

    }
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    cout << duration.count() << endl;
    cout << "done" << endl;

    // Copy the result back to the host
    cudaMemcpy(h_volt.data(), d_volt, num_neurons * sizeof(float), cudaMemcpyDeviceToHost);
    cout << h_volt[0] << endl;

    // Cleanup
    cudaFree(d_volt);
    cudaFree(d_rec);
    cudaFree(d_curr);
    cudaFree(d_fire);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_d);
    cudaFree(d_synapses);
    cudaFree(d_exin_array);

    return 0;
}