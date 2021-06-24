import yaml


def test_resnet18_genotype_generator():

    import os

    absolute_path = os.path.abspath(__file__)

    with open("bloks/test/genotypes/resnet18_baseline.yaml", "r") as config_file:
        ground_truth_config = yaml.safe_load(config_file)

    from bloks.genotypes import generate_resnet_genotypes

    generated_configs = generate_resnet_genotypes()
    generated_config = generated_configs["resnet18"]

    assert ground_truth_config == generated_config
