import gymnastics

proxy = gymnastics.proxies.NASWOT()
dataset = gymnastics.datasets.CIFAR10Loader(path="~/datasets/cifar10", download=False)

minibatch = dataset.sample_minibatch()

best_score = 0.0
best_model = None

# try out 10 random architectures and save the best one
for i in range(10):

    # generate a random network configuration and
    # initialise a ResNet backbone with it
    genotype = gymnastics.genotypes.generate_random_genotype("resnet26")
    model = gymnastics.models.ResNet26(genotype)

    score = proxy.score(model, minibatch)

    if score > best_score:
        best_score = score
        best_model = model
