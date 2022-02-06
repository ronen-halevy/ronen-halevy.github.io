https://stackoverflow.com/questions/49164230/deep-neural-network-skip-connection-implemented-as-summation-vs-concatenation :

Basically, the difference relies on the different way in which the final layer is influenced by middle features.

Standard architectures with skip-connection using element-wise summation (e.g. ResNet) can be viewed as an iterative estimation procedure to some extent (see for instance this work), where the features are refined through the various layers of the network. The main benefits of this choice are that it works and is a compact solution (it keeps the number of features fixed across a block).

Architectures with concatenated skip-connections (e.g. DenseNet), allow the subsequent layers to re-use middle representations, maintaining more information which can lead to better performances. Apart from the feature re-use, another consequence is the implicit deep supervision (as in this work) which allow better gradient propagation across the network, especially for deep ones (in fact it has been used for the Inception architecture).

Obviously, if not properly designed, concatenating features can lead to an exponential growth of the parameters (this explains, in part, the hierarchical aggregation used in the work you pointed out) and, depending on the problem, using a lot of information could lead to overfitting.
