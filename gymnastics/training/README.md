## Training

This part of the repo provides some tools to train the networks we have searched for. 

Idealised usage:

```python
for i in range(num_models_to_collect):
    model = search_space.sample_random_model()

    final_test_error = full_training_run(model)

    add_to_table(model, final_test_error)
```

