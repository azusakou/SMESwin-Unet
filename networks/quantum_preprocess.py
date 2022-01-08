from pennylane.templates import RandomLayers
n_layers = 1
dev = qml.device("default.qubit", wires=4)
# Random circuit parameters
rand_params = np.random.uniform(high=2 * np.pi, size=(n_layers, 4))

@qml.qnode(dev)
def circuit(phi):
    # Encoding of 4 classical input values
    for j in range(4):
        qml.RY(np.pi * phi[j], wires=j)

    # Random quantum circuit
    RandomLayers(rand_params, wires=list(range(4)))

    # Measurement producing 4 classical output values
    return [qml.expval(qml.PauliZ(j)) for j in range(4)]

def quanv(image):
    """Convolves the input image with many applications of the same quantum circuit."""
    i_size=image.shape[2]
    out = np.zeros((4,int(i_size/2),int(i_size/2)))

    # Loop over the coordinates of the top-left pixel of 2X2 squares
    for j in range(0, int(i_size), 2):
        for k in range(0, int(i_size), 2):
            # Process a squared 2x2 region of the image with a quantum circuit
            q_results = circuit(
                [
                    image[0, j, k],
                    image[0, j, k + 1],
                    image[0, j + 1, k],
                    image[0, j + 1, k + 1]
                ]
            )
            # Assign expectation values to different channels of the output pixel (j/2, k/2)
            for c in range(4):
                out[c, j//2, k//2] = q_results[c]
    return out

def q_data(x):
    count = 0
    listvar = []
    for idx, img in enumerate(x):
        img = quanv(img)
        exec('x_n{}=img'.format(count))
        exec('listvar.append(x_n{})'.format(count))
        count+=1
    x = np.asarray(listvar)
    #x = torch.stack(listvar)#; print (x)
    return torch.tensor(x, requires_grad=True)