def parse_weights(weights, primitives):
    gene = []
    for i in range(weights.shape[0]):
        idx = weights[i].argmax().item()
        gene.append(primitives[idx])
    return gene