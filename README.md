# DiffusionSAT

This repository is the official implementation of DiffusionSAT, a pure GNN SAT solver.
- For training, it uses Belief Propagation and QuerySAT-based approach.
- For sampling solutions, it uses multinomial denoising diffusion.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training New Models

Edit the `config.py` file and set these settings in the `Config` class:

| parameter name                                 | sample value           | description                                                  |
| ---------------------------------------------- | ---------------------- | ------------------------------------------------------------ |
| train_steps                                    | 167_000                | number of iterations to perform during the training process; <br />if `use_cosine_decay==False` (see below), the number of iterations should be 100_000 or more |
| train_min_vars                                 | 3 (30)                 | the min number of variables in a random SAT formula to generate |
| train_max_vars                                 | 30 (100)               | the max number of variables in a random SAT formula to generate |
| train_size                                     | 100_000                | the number of SAT instances to generate                      |
| test_size                                      | 10_000                 | the number of SAT instances to use for validation            |
| use_hard_3sat                                  | True                   | whether hard 3-SAT instances should be generated with clause/variable ratio ~ 4.3 |
| desired_multiplier_for_the_number_of_solutions | 20                     | [ony if `use_hard_3sat==False`] remove some clauses of random SAT instances to increate the number of possible solutions by this factor |
| max_nodes_per_batch                            | 20_000 (for NVidia T4) | the upper limit on how many literals and clauses to use in one batch |
| use_cosine_decay                               | True                   | use CosineDecay instead of fixed rate learning schedule      |
| learning_rate                                  | 0.0003                 | [only if `use_cosine_decay==False`] the fixed learning rate  |
| use_unigen                                     | True                   | use Unigen solver for computing sample solutions; if False, Glucose will be used;<br />see also: `data/diffusion_sat_instances.py#get_sat_solution` |

Then launch
```bash
python3 ./diffusion_training.py
```

## Using Existing Models

Existing (pre-trained) models are available in the Releases section.

## Sampling SAT Solutions
You can use diffusion_sample.py. There you need to define two variables:
- model_path -- specifies where the pre-trained model is located
- dimacs_filename -- specifies the name of the SAT problem specified in the DIMACS format

```python
diffusion_dict = DiffusionSampler(model_path, dimacs_filename).samples(n_samples)
```

The result is a python dictionary in the format: `[solution_as_int: number_of_occurrences]`.
The `solution_as_int` is an int with binary representation corresponding to bit values for x1, x2,...xN (the right-most bit is for x1, bit 0 means "False", bit 1 means "True").
The `number_of_occurrences` shows how many times that particular solution was generated.



## Evaluation

Some evaluation (e.g., uniformity) can be obtained by launching:
```bash
python3 diffusion_evaluation.py
```

## Results

Papers: