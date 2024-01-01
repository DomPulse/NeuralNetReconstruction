#include <iostream>
#include <vector>
#include <random>

using namespace std;

int num_neurons{ 1000 };
int sim_time{ 1000 };
float frac_inhib{ 0.2 };
vector<vector<float>> neurons(4, vector<float>(num_neurons)); //membrane voltage V, recovery u, current I, just fired 0 means not fired, 1 means just fired
vector<vector<float>> alphabet(4, vector<float>(num_neurons)); //fit params a, b, c, d
vector<vector<float>> synapses(num_neurons, vector<float>(num_neurons));
vector<int> exin_array(num_neurons);

vector<vector<float>> update_neuron(vector<vector<float>> neuron, vector<vector<float>> fit_params, int neuron_idx)
{
	neuron[3][neuron_idx] = 0;
	neuron[0][neuron_idx] += 0.5 * (0.04 * neuron[0][neuron_idx] * neuron[0][neuron_idx] + 5 * neuron[0][neuron_idx] + 140 - neuron[1][neuron_idx] + neuron[2][neuron_idx]);
	neuron[0][neuron_idx] += 0.5 * (0.04 * neuron[0][neuron_idx] * neuron[0][neuron_idx] + 5 * neuron[0][neuron_idx] + 140 - neuron[1][neuron_idx] + neuron[2][neuron_idx]);
	neuron[1][neuron_idx] += fit_params[0][neuron_idx] * (fit_params[1][neuron_idx] * neuron[0][neuron_idx] - neuron[1][neuron_idx]);
	neuron[2][neuron_idx] = 0;
	if (neuron[0][neuron_idx] >= 30)
	{
		neuron[0][neuron_idx] = fit_params[2][neuron_idx];
		neuron[1][neuron_idx] += fit_params[3][neuron_idx];
		neuron[3][neuron_idx] = 1;
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

vector<vector<float>> define_fit_params(vector<int> exin)
{
	vector<vector<float>> temp_fit_params(4, vector<float>(num_neurons));
	for (int n = 0; n < num_neurons; n++)
	{
		if (exin[n] == -1) 
		{
			float ri = rand() / static_cast<float>(RAND_MAX);
			temp_fit_params[0][n] = 0.02 + 0.08 * ri;
			temp_fit_params[1][n] = 0.25 - 0.05 * ri;
			temp_fit_params[2][n] = -65.0;
			temp_fit_params[3][n] = 2;

		}
		else
		{
			float re = rand() / static_cast<float>(RAND_MAX);
			temp_fit_params[0][n] = 0.02;
			temp_fit_params[1][n] = 0.2;
			temp_fit_params[2][n] = -65.0+15*re*re;
			temp_fit_params[3][n] = 8-6*re*re;
		}
	}
	return temp_fit_params;
}

vector<vector<float>> initialize_neurons(vector<vector<float>> temp_fit_params)
{
	vector<vector<float>> temp_neurons(4, vector<float>(num_neurons));
	for (int n = 0; n < num_neurons; n++)
	{
		float u = -65.0 * temp_fit_params[1][n]; //cringe double work around
		temp_neurons[0][n] = -65.0;
		temp_neurons[1][n] = u;
		temp_neurons[2][n] = 0.0;
		temp_neurons[3][n] = 0.0;
	}
	return temp_neurons;
}

vector<vector<float>> define_synapses(vector<int> exin)
{
	vector<vector<float>> rand_synapses(num_neurons, vector<float>(num_neurons));
	for (int post_syn_idx = 0; post_syn_idx < num_neurons; post_syn_idx++)
	{
		for (int pre_syn_idx = 0; pre_syn_idx < num_neurons; pre_syn_idx++)
		{
			rand_synapses[post_syn_idx][pre_syn_idx] = (rand() / static_cast<float>(RAND_MAX))*exin[pre_syn_idx]*(1-0.5*(exin[pre_syn_idx]==1));
			if (post_syn_idx == pre_syn_idx)
			{
				rand_synapses[post_syn_idx][pre_syn_idx] = 0;
			}
		}
	}
	return rand_synapses;

}

vector<float> firing_current(vector<float> currents, vector<float> fires, vector<vector<float>> temp_synapses)
{
	for (int post_syn_idx = 0; post_syn_idx < num_neurons; post_syn_idx++)
	{
		for (int pre_syn_idx = 0; pre_syn_idx < num_neurons; pre_syn_idx++)
		{
			//I know this would be faster with an if statement, but eventually it will be bitwise and hopefully that makes up for it, idk
			currents[post_syn_idx] += temp_synapses[post_syn_idx][pre_syn_idx] * fires[pre_syn_idx];
		}
	}
	return currents;
}

vector<float> noisy_current(vector<int> exin)
{
	//this is gonna be uber slow, when I get every parallelized it'll hopefully be fast af
	default_random_engine gen;
	normal_distribution<float> d(0, 1);
	vector<float> temp_current(num_neurons);
	for (int n = 0; n < num_neurons; n++)
	{
		if (exin[n] == 1)
		{
			
			temp_current[n] = 5 * d(gen);
		}
		else
		{
			temp_current[n] = 2 * d(gen);
		}
	}
	return temp_current;
}

int main()
{
	exin_array = define_exin_array();
	alphabet = define_fit_params(exin_array);
	neurons = initialize_neurons(alphabet);
	synapses = define_synapses(exin_array);

	for (int t = 0; t < sim_time; t++)
	{
		for (int n = 0; n < num_neurons; n++)
		{
			neurons = update_neuron(neurons, alphabet, n); //custom function but each neuron is independent so it should be parallelizable

			/*
			if (neurons[n][3] == 1)
			{
				cout << "+";
			}
			else
			{
				cout << "-";
			}
			*/

		}
		//cout << endl;
		//turn this into a highly parallel vector addition big facts
		neurons[2] = noisy_current(exin_array); //picking random numbers, hopefully parallelizeble
		neurons[2] = firing_current(neurons[2], neurons[3], synapses); //essentially matrix multiplication then vector addition, both parallelizable
	}
	return 0;
}