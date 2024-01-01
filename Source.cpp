#include <iostream>
#include <vector>
#include <random>

using namespace std;

int num_neurons{ 1000 };
int sim_time{ 1000 };
float frac_inhib{ 0.2 };
vector<vector<float>> neurons(num_neurons, vector<float>(4)); //membrane voltage V, recovery u, current I, just fired 0 means not fired, 1 means just fired
vector<vector<float>> alphabet(num_neurons, vector<float>(4)); //fit params a, b, c, d
vector<vector<float>> synapses(num_neurons, vector<float>(num_neurons));
vector<int> exin_array(num_neurons);

vector<float> update_neuron(vector<float> neuron, vector<float> fit_params)
{
	neuron[3] = 0;
	neuron[0] += 0.5 * (0.04 * neuron[0] * neuron[0] + 5 * neuron[0] + 140 - neuron[1] + neuron[2]);
	neuron[0] += 0.5 * (0.04 * neuron[0] * neuron[0] + 5 * neuron[0] + 140 - neuron[1] + neuron[2]);
	neuron[1] += fit_params[0] * (fit_params[1] * neuron[0] - neuron[1]);
	neuron[2] = 0;
	if (neuron[0] >= 30)
	{
		neuron[0] = fit_params[2];
		neuron[1] += fit_params[3];
		neuron[3] = 1;
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
	vector<vector<float>> temp_fit_params(num_neurons, vector<float>(4));
	for (int n = 0; n < num_neurons; n++)
	{
		if (exin[n] == -1) 
		{
			float ri = rand() / static_cast<float>(RAND_MAX);
			float a = 0.02 + 0.08 * ri;
			float b = 0.25 - 0.05 * ri;
			float c = -65.0;
			float d = 2;
			temp_fit_params[n] = { a,b,c,d };

		}
		else
		{
			float re = rand() / static_cast<float>(RAND_MAX);
			float a = 0.02;
			float b = 0.2;
			float c = -65.0+15*re*re;
			float d = 8-6*re*re;
			temp_fit_params[n] = { a,b,c,d };
		}
	}
	return temp_fit_params;
}

vector<vector<float>> initialize_neurons(vector<vector<float>> temp_fit_params)
{
	vector<vector<float>> temp_neurons(num_neurons, vector<float>(4));
	for (int n = 0; n < num_neurons; n++)
	{
		float u = -65.0 * temp_fit_params[n][1]; //cringe double work around
		temp_neurons[n] = { -65.0, u, 0.0, 0.0 };
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

vector<vector<float>> firing_current(vector<vector<float>> temp_neurons, vector<vector<float>> temp_synapses)
{
	//this is gonna be uber slow, when I get every parallelized it'll hopefully be fast af


	
	for (int post_syn_idx = 0; post_syn_idx < num_neurons; post_syn_idx++)
	{
		for (int pre_syn_idx = 0; pre_syn_idx < num_neurons; pre_syn_idx++)
		{
			//I know this would be faster with an if statement, but eventually it will be bitwise and hopefully that makes up for it, idk
			temp_neurons[post_syn_idx][2] += temp_synapses[post_syn_idx][pre_syn_idx] * temp_neurons[pre_syn_idx][3];
		}
	}
	return temp_neurons;
}

vector<vector<float>> noisy_current(vector<vector<float>> temp_neurons, vector<int> exin)
{
	//this is gonna be uber slow, when I get every parallelized it'll hopefully be fast af
	default_random_engine gen;
	normal_distribution<float> d(0, 1);
	for (int n = 0; n < num_neurons; n++)
	{
		if (exin[n] == 1)
		{
			
			temp_neurons[n][2] = 5 * d(gen);
		}
		else
		{
			temp_neurons[n][2] = 2 * d(gen);
		}
	}
	return temp_neurons;
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
			neurons[n] = update_neuron(neurons[n], alphabet[n]);
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
		neurons = noisy_current(neurons, exin_array);
		neurons = firing_current(neurons, synapses);
	}
	return 0;
}