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

Edit the config.py file and set these settings:
```python
    train_steps = 167_000 #1_000 #5_000 #10_000 #25_000 #50_000 #75_000 #167_000
    train_min_vars = 30 #3 30
    train_max_vars = 100 #30 100
    test_size=10_000
    train_size=100_000
    use_hard_3sat = True 
       # ^^^ if True, use hard 3-SAT instances with 4.3 clause/variable ratio for training;
       #     if False, use NeuroSAT-based k-SAT generation algorithm
    desired_multiplier_for_the_number_of_solutions = 20
       # ^^^ only if use_hard_3sat==False
       # remove some clauses to multiply the number of samples by desired_multiplier_for_the_number_of_solutions
    max_nodes_per_batch = 20_000 # 20_000 for Nvidia T4, 60_000 for more advanced cards (setting by SK)
    use_cosine_decay = True # use CosineDecay instead of fixed rate schedule
    learning_rate = 0.0003
       # ^^^ the fixed learning rate, if use_cosine_decay==False
    use_unigen = True
       # ^^^ Unigen() or Glucose() for computing samples used for training;
       #     see: data/diffusion_sat_instances.py#get_sat_solution
```

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