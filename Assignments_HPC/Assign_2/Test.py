from mpi4py import MPI
import numpy as np
import time

def compute_gradient(data, labels, weight):
    errors = labels @ weight - data
    gradient = labels.T.dot(errors) * (2/ len(data))
    return gradient
def generate_synthetic_data(num_samples, noise_std=0.1):
    np.random.seed(42)
    X = np.random.rand(num_samples, 1)
    Y = 2 * X + noise_std * np.random.randn(num_samples, 1)
    return X, Y

def update_weights(weights, learning_rate, gradient):
    # Update weights using SGD
    return weights - learning_rate * gradient

def parallel_sgd(X, Y, learning_rate, num_iterations, batch_size):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    num_samples = len(Y)
    num_features = X.shape[1]

    # Distribute data among processes
    local_size = num_samples // size
    local_start = rank * local_size
    local_end = (rank + 1) * local_size if rank != size - 1 else num_samples

    X_local = X[local_start:local_end]
    Y_local = Y[local_start:local_end]

    # Initialize weights on each process
    weights = np.random.rand(num_features, 1)

    for iteration in range(num_iterations):
        # Compute local gradient
        local_gradient = compute_gradient(X_local, Y_local, weights)

        # Synchronize weights periodically
        if iteration % 10 == 0:
            comm.Allreduce(MPI.IN_PLACE, weights, op=MPI.SUM)

        # Update weights using SGD
        weights = update_weights(weights, learning_rate, local_gradient)

    return weights

if __name__ == "__main__":
    num_samples = 1000
    noise_std = 0.1
    learning_rate = 0.01
    num_iterations = 100
    batch_size = 32

    start_time = time.time()
    X, Y = generate_synthetic_data(num_samples, noise_std)

    # Perform parallel SGD
    weights = parallel_sgd(X, Y, learning_rate, num_iterations, batch_size)

    if MPI.COMM_WORLD.Get_rank() == 0:
        print("Final weights:", weights)
        print("Execution time:", time.time() - start_time)