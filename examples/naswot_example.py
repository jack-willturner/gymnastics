import gymnastics

# use the 101 Skeleton with the 201 cell space
skeleton = gymnastics.searchspace.NASBench101Skeleton()
cell_space = gymnastics.searchspace.NASBench201CellSpace()

search_space = gymnastics.searchspace.SearchSpace(skeleton, cell_space)

# create an accuracy predictor
proxy = gymnastics.proxies.NASWOT()
train, _ = gymnastics.datasets.CIFAR10()

minibatch = train.sample_minibatch()

best_score = 0.0
best_model = None

# try out 10 random architectures and save the best one
for i in range(10):

    model = search_space.sample_random_architecture()

    score = proxy.score(model, minibatch)

    if score > best_score:
        best_score = score
        best_model = model
