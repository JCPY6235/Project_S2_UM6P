from mpi4py import MPI
import numpy as np

# Fonction pour mettre à jour l'état de la grille en fonction des règles du jeu de la vie
def update_grid(local_grid):
    new_grid = np.zeros(local_grid.shape, dtype=int)
    for i in range(1, local_grid.shape[0]-1):
        for j in range(1, local_grid.shape[1]-1):
            # Compter le nombre de voisins vivants
            neighbor_count = np.sum(local_grid[i-1:i+2, j-1:j+2]) - local_grid[i, j]
            # Appliquer les règles du jeu de la vie
            if local_grid[i, j] == 1 and (neighbor_count < 2 or neighbor_count > 3):
                new_grid[i, j] = 0  # Sous-population ou surpopulation
            elif local_grid[i, j] == 0 and neighbor_count == 3:
                new_grid[i, j] = 1  # Reproduction
            else:
                new_grid[i, j] = local_grid[i, j]  # Conserver l'état actuel
    return new_grid

# Créer une grille initiale aléatoire sur chaque processus
def create_initial_grid(local_size):
    return np.random.randint(2, size=local_size, dtype=int)

# Initialiser MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Définir la taille de la grille globale
global_rows, global_cols = 10, 10

# Calculer la taille locale de chaque processus
local_rows = global_rows // size
local_cols = global_cols

# Créer la grille cartésienne de processus MPI
dims = [size, 1]
periods = [True, False]
cart_comm = comm.Create_cart(dims, periods)

# Créer la sous-grille locale pour chaque processus
local_grid = np.zeros((local_rows + 2, local_cols), dtype=int)

# Remplir la grille locale avec des valeurs aléatoires
local_grid[1:-1, :] = create_initial_grid((local_rows, local_cols))

# Boucle principale pour les itérations du jeu de la vie
num_iterations = 5
for it in range(num_iterations):
    # Envoyer et recevoir les bords de la grille locale avec les voisins
    north_neighbor, south_neighbor = cart_comm.Shift(0, 1)
    cart_comm.Sendrecv([local_grid[-2, :], MPI.INT], dest=north_neighbor, recvbuf=[local_grid[0, :], MPI.INT], source=south_neighbor)
    cart_comm.Sendrecv([local_grid[1, :], MPI.INT], dest=south_neighbor, recvbuf=[local_grid[-1, :], MPI.INT], source=north_neighbor)
    
    # Mettre à jour l'état de la grille
    local_grid = update_grid(local_grid)

    # Afficher la grille locale après chaque itération
    print("Process", rank, "iteration", it)
    print(local_grid[1:-1, :])

# Terminer MPI
MPI.Finalize()
