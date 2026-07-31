#!/bin/bash
# Token Bias Perturbation — boltz2

cache_dir="/path/to/cache_dir"
BOLTZ2WEIGHT="/path/to/boltz2_conf.ckpt"
yaml_dir="/path/to/yaml_dir"
out_dir="/path/to/out_dir"

recycles=3
num_workers=3
num_samples=5
sampling_steps=200
step_scale=1.638
seed=42

noise_scale=5.0
token_bias_t_start=1.0
token_bias_t_end=0.0
token_bias_psi=0.7
token_bias_mode=cross_and_ll

boltz predict "$yaml_dir" \
	--cache "$cache_dir" \
	--out_dir "$out_dir" \
	--diffusion_samples $num_samples \
	--num_workers $num_workers \
	--step_scale $step_scale \
	--seed $seed \
	--recycling_steps $recycles \
	--sampling_steps $sampling_steps \
	--devices 1 \
	--checkpoint "$BOLTZ2WEIGHT" \
	--noise_token_bias \
	--token_bias_noise_scale $noise_scale \
	--token_bias_t_start $token_bias_t_start \
	--token_bias_t_end $token_bias_t_end \
	--token_bias_psi $token_bias_psi \
	--token_bias_mode $token_bias_mode
