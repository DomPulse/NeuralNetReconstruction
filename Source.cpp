#include <iostream>
#include <vector>
#include <random>

using namespace std;

int num_neurons{ 1000 };
int sim_time{ 1000 };
float frac_inhib{ 0.2 };
//yeah so used to be a 2D array but that's not jiving with the 
vector<float> neurons(4 * num_neurons); //membrane voltage V, recovery u, current I, just fired 0 means not fired, 1 means just fired repeating
vector<float> alphabet(4 * num_neurons); //fit params a, b, c, d repeating
vector<float> synapses(num_neurons*num_neurons);
vector<int> exin_array(num_neurons);

vector<float> update_neuron(vector<float> neuron, vector<float> fit_params, int neuron_idx)
{
	//indexed by taking the neuron_idx + some offset
	//neuron_idx+0 -> V
	//neuron_idx+1 -> u
	//neuron_idx+2 -> I
	//neuron_idx+3 -> jF
	//bc of this, neuron_idx counts by 4
	//same scheme for fit parameters
	neuron[neuron_idx+3] = 0;
	neuron[neuron_idx] += 0.5 * (0.04 * neuron[neuron_idx] * neuron[neuron_idx] + 5 * neuron[neuron_idx] + 140 - neuron[neuron_idx+1] + neuron[neuron_idx+2]);
	neuron[neuron_idx] += 0.5 * (0.04 * neuron[neuron_idx] * neuron[neuron_idx] + 5 * neuron[neuron_idx] + 140 - neuron[neuron_idx+1] + neuron[neuron_idx + 2]);
	neuron[neuron_idx+1] += fit_params[neuron_idx] * (fit_params[neuron_idx+1] * neuron[neuron_idx] - neuron[neuron_idx+1]);
	neuron[neuron_idx+2] = 5;
	if (neuron[neuron_idx] >= 30)
	{
		neuron[neuron_idx] = fit_params[neuron_idx+2];
		neuron[neuron_idx+1] += fit_params[neuron_idx+3];
		neuron[neuron_idx+3] = 1;
	}
	return neuron;
}

vector<int> define_exin_array()
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

vector<float> define_fit_params(vector<int> exin)
{
	vector<float> temp_fit_params(4 * num_neurons);
	for (int n = 0; n < num_neurons * 4; n+=4)
	{
		if (exin[n/4] == -1) 
		{
			float ri = rand() / static_cast<float>(RAND_MAX);
			temp_fit_params[n] = 0.02 + 0.08 * ri;
			temp_fit_params[n+1] = 0.25 - 0.05 * ri;
			temp_fit_params[n+2] = -65.0;
			temp_fit_params[n+3] = 2;

		}
		else
		{
			float re = rand() / static_cast<float>(RAND_MAX);
			temp_fit_params[n] = 0.02;
			temp_fit_params[n+1] = 0.2;
			temp_fit_params[n+2] = -65.0+15*re*re;
			temp_fit_params[n+3] = 8-6*re*re;
		}
	}
	return temp_fit_params;
}

vector<float> initialize_neurons(vector<float> temp_fit_params)
{
	vector<float> temp_neurons(4 * num_neurons);
	for (int n = 0; n < num_neurons; n+=4)
	{
		float u = -65.0 * temp_fit_params[n+1]; //cringe double work around
		temp_neurons[n] = -65.0;
		temp_neurons[n+1] = u;
		temp_neurons[n+2] = 0.0;
		temp_neurons[n+3] = 0.0;
	}
	return temp_neurons;
}

vector<float> define_synapses(vector<int> exin)
{
	vector<float> rand_synapses(num_neurons * num_neurons);
	for (int post_syn_idx = 0; post_syn_idx < num_neurons; post_syn_idx++)
	{
		for (int pre_syn_idx = 0; pre_syn_idx < num_neurons; pre_syn_idx++)
		{
			int tot_idx = post_syn_idx * num_neurons + pre_syn_idx; //remember this so i can actuall reference shit later
			rand_synapses[tot_idx] = (rand() / static_cast<float>(RAND_MAX))*exin[pre_syn_idx]*(1-0.5*(exin[pre_syn_idx]==1));
			if (post_syn_idx == pre_syn_idx)
			{
				rand_synapses[tot_idx] = 0;
			}
		}
	}
	return rand_synapses;

}

vector<float> firing_current(vector<float> neuron, vector<float> temp_synapses)
{
	for (int post_syn_idx = 0; post_syn_idx < num_neurons; post_syn_idx++)
	{
		for (int pre_syn_idx = 0; pre_syn_idx < num_neurons; pre_syn_idx++)
		{
			//I know this would be faster with an if statement, but eventually it will be bitwise and hopefully that makes up for it, idk
			int tot_idx = post_syn_idx * num_neurons + pre_syn_idx;
			neuron[post_syn_idx+2] += temp_synapses[tot_idx] * neuron[pre_syn_idx+3];
		}
	}
	return neuron;
}

vector<float> noisy_current(vector<float> neuron, vector<int> exin)
{
	//this is gonna be uber slow, when I get every parallelized it'll hopefully be fast af
	default_random_engine gen;
	normal_distribution<float> d(0, 1);
	for (int n = 0; n < num_neurons*4; n+=4)
	{
		if (exin[n/4] == 1)
		{
			
			neuron[n+2] = 5 * d(gen);
		}
		else
		{
			neuron[n+2] = 2 * d(gen);
		}
	}
	return neuron;
}

int main()
{
	exin_array = define_exin_array();
	alphabet = define_fit_params(exin_array);
	neurons = initialize_neurons(alphabet);
	synapses = define_synapses(exin_array);

	for (int t = 0; t < sim_time; t++)
	{
		for (int n = 0; n < num_neurons*4; n+=4)
		{
			neurons = update_neuron(neurons, alphabet, n); //custom function but each neuron is independent so it should be parallelizable

		}

		//turn this into a highly parallel vector addition big facts
		neurons = noisy_current(neurons, exin_array); //picking random numbers, hopefully parallelizeble
		neurons = firing_current(neurons, synapses); //essentially matrix multiplication then vector addition, both parallelizable
	}
	return 0;
}