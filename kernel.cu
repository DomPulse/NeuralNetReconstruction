//I would like to thank chatGPT for writing a lot of the gpu stuff
//I handled the logic of updating neurons but chatGPT really did me a solid

#include <iostream>
#include <vector>
#include <curand_kernel.h>
#include <random>
//so i linearized all the arrays and that may have been completely unneccessay,so when this works I'll try and go back to do it with arrays
using namespace std;

template <typename T>
__global__ void synapse_current(T* neuron, T* synapses, int num_neurons) {//yeah we be working on its
    //should probably interchange the order of pre and post syn idx so i can just check if it fired once and avoid the loop if it didn't
    int post_syn_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    for (int pre_syn_idx = 0; pre_syn_idx < num_neurons; pre_syn_idx++)
    {
        int tot_idx = post_syn_idx * num_neurons / 4 + pre_syn_idx; 
        neuron[post_syn_idx + 2] += synapses[tot_idx]*neuron[pre_syn_idx * 4 + 3];
    }
}

template <typename T>
__global__ void noisy_current(T* neuron, int* exin, unsigned long long seed, unsigned long long offset) {//yeah idk how the whoel T* thing works, it had a breakdown when i tried to pass a float and an int, crazy stuff
    int neuron_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    curandState state;
    curand_init(seed, neuron_idx, offset, &state);

    neuron[neuron_idx + 2] = (2 + 3 * (exin[neuron_idx / 4] == 1)) * curand_normal(&state); //this is probably the slow down, should probably have 4 vector instead of 1 vector with 4x the length  
}

template <typename T>
__global__ void update_neuron(T* neuron, T* fit_params)
{
    //indexed by taking the neuron_idx + some offset
    //neuron_idx+0 -> V
    //neuron_idx+1 -> u
    //neuron_idx+2 -> I
    //neuron_idx+3 -> jF
    //bc of this, neuron_idx counts by 4
    //same scheme for fit parameters
    int neuron_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    neuron[neuron_idx + 3] = 0;
    neuron[neuron_idx] += 0.5 * (0.04 * neuron[neuron_idx] * neuron[neuron_idx] + 5 * neuron[neuron_idx] + 140 - neuron[neuron_idx + 1] + neuron[neuron_idx + 2]);
    neuron[neuron_idx] += 0.5 * (0.04 * neuron[neuron_idx] * neuron[neuron_idx] + 5 * neuron[neuron_idx] + 140 - neuron[neuron_idx + 1] + neuron[neuron_idx + 2]);
    neuron[neuron_idx + 1] += fit_params[neuron_idx] * (fit_params[neuron_idx + 1] * neuron[neuron_idx] - neuron[neuron_idx + 1]);
    neuron[neuron_idx + 2] = 0;
    if (neuron[neuron_idx] >= 30)
    {
        neuron[neuron_idx] = fit_params[neuron_idx + 2];
        neuron[neuron_idx + 1] += fit_params[neuron_idx + 3];
        neuron[neuron_idx + 3] = 1;
    }
}

template <typename T>
void printVector(const vector<T>& vec) {
    for (const auto& element : vec) {
        cout << element << " ";
    }
    cout << endl;
}

vector<int> define_exin_array(int num_neurons, float frac_inhib)
{
    vector<int> temp_exin_array(num_neurons);
    for (int n = 0; n < num_neurons; n++)
    {
        if (rand() / static_cast<float>(RAND_MAX) < frac_inhib)
        {
            temp_exin_array[n] = -1;
        }
        else
        {
            temp_exin_array[n] = 1;
        }
    }
    return temp_exin_array;
}

vector<float> define_fit_params(vector<int> exin, int num_neurons)
{
    vector<float> temp_fit_params(4 * num_neurons);
    for (int n = 0; n < num_neurons * 4; n += 4)
    {
        if (exin[n / 4] == -1)
        {
            float ri = rand() / static_cast<float>(RAND_MAX);
            temp_fit_params[n] = 0.02 + 0.08 * ri;
            temp_fit_params[n + 1] = 0.25 - 0.05 * ri;
            temp_fit_params[n + 2] = -65.0;
            temp_fit_params[n + 3] = 2;

        }
        else
        {
            float re = rand() / static_cast<float>(RAND_MAX);
            temp_fit_params[n] = 0.02;
            temp_fit_params[n + 1] = 0.2;
            temp_fit_params[n + 2] = -65.0 + 15 * re * re;
            temp_fit_params[n + 3] = 8 - 6 * re * re;
        }
    }
    return temp_fit_params;
}

vector<float> initialize_neurons(vector<float> temp_fit_params, int num_neurons)
{
    vector<float> temp_neurons(4 * num_neurons);
    for (int n = 0; n < num_neurons; n += 4)
    {
        float u = -65.0 * temp_fit_params[n + 1]; //cringe double work around
        temp_neurons[n] = -65.0;
        temp_neurons[n + 1] = u;
        temp_neurons[n + 2] = 0.0;
        temp_neurons[n + 3] = 0.0;
    }
    return temp_neurons;
}

vector<float> define_synapses(vector<int> exin, int num_neurons)
{
    vector<float> rand_synapses(num_neurons * num_neurons);
    for (int post_syn_idx = 0; post_syn_idx < num_neurons; post_syn_idx++)
    {
        for (int pre_syn_idx = 0; pre_syn_idx < num_neurons; pre_syn_idx++)
        {
            int tot_idx = post_syn_idx * num_neurons + pre_syn_idx; //remember this so i can actuall reference shit later
            rand_synapses[tot_idx] = (rand() / static_cast<float>(RAND_MAX)) * exin[pre_syn_idx] * (1 - 0.5 * (exin[pre_syn_idx] == 1));
            if (post_syn_idx == pre_syn_idx)
            {
                rand_synapses[tot_idx] = 0;
            }
        }
    }
    return rand_synapses;

}

int main() {
    unsigned long long seed = 1234;  // Random seed

    int num_neurons = 1024;  
    int sim_time{ 1000 };
    float frac_inhib{ 0.2 };

    // Host vector to store the result
    vector<float> h_neurons(4 * num_neurons); //membrane voltage V, recovery u, current I, just fired 0 means not fired, 1 means just fired, repeating in order
    vector<float> h_alphabet(4 * num_neurons); //fit params a, b, c, d repeating
    vector<float> h_synapses(num_neurons * num_neurons);
    vector<int> h_exin_array(num_neurons);

    //initializing the host variables
    h_exin_array = define_exin_array(num_neurons, frac_inhib);
    h_alphabet = define_fit_params(h_exin_array, num_neurons);
    h_neurons = initialize_neurons(h_alphabet, num_neurons);
    h_synapses = define_synapses(h_exin_array, num_neurons);

    // Device vector
    float *d_neurons; 
    float *d_alphabet;
    float *d_synapses;
    int *d_exin_array;
    cudaMalloc((void**)&d_neurons, 4 * num_neurons * sizeof(float));
    cudaMalloc((void**)&d_alphabet, 4 * num_neurons * sizeof(float));
    cudaMalloc((void**)&d_synapses, num_neurons * num_neurons * sizeof(float));
    cudaMalloc((void**)&d_exin_array, num_neurons * sizeof(int));
    
    //transfering initializaed data to gpu
    cudaMemcpy(d_neurons, h_neurons.data(), 4 * num_neurons * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_alphabet, h_alphabet.data(), 4 * num_neurons * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_synapses, h_synapses.data(), num_neurons * num_neurons * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_exin_array, h_exin_array.data(), num_neurons * sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel to generate random numbers on the GPU
    int threadsPerBlock = 32; //was formerly 256
    int blocksPerGrid = (num_neurons + threadsPerBlock - 1) / threadsPerBlock;

    //update the neurons with 1ms time step
    for (int t = 0; t < sim_time; t++)
    {
        //take action brother!
        update_neuron << <blocksPerGrid, threadsPerBlock >> >(d_neurons, d_alphabet);
        noisy_current << <blocksPerGrid, threadsPerBlock >> > (d_neurons, d_exin_array, seed, 0.0);
        synapse_current << <blocksPerGrid, threadsPerBlock >> > (d_neurons, d_synapses, num_neurons);
        //yeah so now i just need the currents implimented so thats cool
        
    }
    // Copy the result back to the host
    cudaMemcpy(h_neurons.data(), d_neurons, num_neurons * 4 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_alphabet.data(), d_alphabet, num_neurons * 4 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_synapses.data(), d_synapses, num_neurons * num_neurons * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_exin_array.data(), d_exin_array, num_neurons * sizeof(int), cudaMemcpyDeviceToHost);
    cout << h_neurons[4] << endl;

    // Cleanup
    cudaFree(d_neurons);
    cudaFree(d_alphabet);
    cudaFree(d_synapses);
    cudaFree(d_exin_array);

    return 0;
}
