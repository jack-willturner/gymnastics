from .nds import NDSSearchSpace

__all__ = ["get_benchmark"]


def get_benchmark(benchmark_name: str, path_to_api: str):

    if benchmark_name == "NASBench101":
        from .nasbench101 import NASBench101SearchSpace

        return NASBench101SearchSpace(path_to_api=path_to_api)
    elif benchmark_name == "NASBench201":
        from .nasbench201 import NASBench201SearchSpace

        return NASBench201SearchSpace(path_to_api=path_to_api)
    elif benchmark_name == "NATSBench":
        from .natsbench import NATSBenchSearchSpace

        return NATSBenchSearchSpace(path_to_api=path_to_api)
    elif benchmark_name == "NDS_resnet":
        return NDSSearchSpace("ResNet")
    elif benchmark_name == "NDS_amoeba":
        return NDSSearchSpace("Amoeba")
    elif benchmark_name == "NDS_amoeba_in":
        return NDSSearchSpace("Amoeba_in")
    elif benchmark_name == "NDS_darts_in":
        return NDSSearchSpace("DARTS_in")
    elif benchmark_name == "NDS_darts":
        return NDSSearchSpace("DARTS")
    elif benchmark_name == "NDS_darts_fix-w-d":
        return NDSSearchSpace("DARTS_fix-w-d")
    elif benchmark_name == "NDS_darts_lr-wd":
        return NDSSearchSpace("DARTS_lr-wd")
    elif benchmark_name == "NDS_enas":
        return NDSSearchSpace("ENAS")
    elif benchmark_name == "NDS_enas_in":
        return NDSSearchSpace("ENAS_in")
    elif benchmark_name == "NDS_enas_fix-w-d":
        return NDSSearchSpace("ENAS_fix-w-d")
    elif benchmark_name == "NDS_pnas":
        return NDSSearchSpace("PNAS")
    elif benchmark_name == "NDS_pnas_fix-w-d":
        return NDSSearchSpace("PNAS_fix-w-d")
    elif benchmark_name == "NDS_pnas_in":
        return NDSSearchSpace("PNAS_in")
    elif benchmark_name == "NDS_nasnet":
        return NDSSearchSpace("NASNet")
    elif benchmark_name == "NDS_nasnet_in":
        return NDSSearchSpace("NASNet_in")
    elif benchmark_name == "NDS_resnext-a":
        return NDSSearchSpace("ResNeXt-A")
    elif benchmark_name == "NDS_resnext-a_in":
        return NDSSearchSpace("ResNeXt-A_in")
    elif benchmark_name == "NDS_resnext-b":
        return NDSSearchSpace("ResNeXt-B")
    elif benchmark_name == "NDS_resnext-b_in":
        return NDSSearchSpace("ResNeXt-B_in")
    elif benchmark_name == "NDS_vanilla":
        return NDSSearchSpace("Vanilla")
    elif benchmark_name == "NDS_vanilla_lr-wd":
        return NDSSearchSpace("Vanilla_lr-wd")
    elif benchmark_name == "NDS_vanilla_lr-wd_in":
        return NDSSearchSpace("Vanilla_lr-wd_in")
