import gmsh

# Initialiser Gmsh
gmsh.initialize()
gmsh.model.add("Rectangular Domain with BCs")

# Créer les points du rectangle (exemple : 2 x 1)
length = 2.0
height = 1.0
lc = 0.1  # Taille caractéristique du maillage

gmsh.model.geo.addPoint(0, 0, 0, lc, 1)
gmsh.model.geo.addPoint(length, 0, 0, lc, 2)
gmsh.model.geo.addPoint(length, height, 0, lc, 3)
gmsh.model.geo.addPoint(0, height, 0, lc, 4)

# Créer les lignes du rectangle
gmsh.model.geo.addLine(1, 2, 1)
gmsh.model.geo.addLine(2, 3, 2)
gmsh.model.geo.addLine(3, 4, 3)
gmsh.model.geo.addLine(4, 1, 4)

# Créer une boucle et une surface plane
gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
gmsh.model.geo.addPlaneSurface([1], 1)

# Synchroniser la géométrie
gmsh.model.geo.synchronize()

# Définir les conditions aux limites sur les bords
# Par exemple : Dirichlet sur les bords gauche et droit, Neumann sur les bords haut et bas
gmsh.model.addPhysicalGroup(1, [1], 1)  # Bord gauche
gmsh.model.addPhysicalGroup(1, [3], 2)  # Bord droit
gmsh.model.addPhysicalGroup(1, [2], 3)  # Bord haut
gmsh.model.addPhysicalGroup(1, [4], 4)  # Bord bas

# Nommer les groupes physiques
gmsh.model.setPhysicalName(1, 1, "Dirichlet Left")
gmsh.model.setPhysicalName(1, 2, "Dirichlet Right")
gmsh.model.setPhysicalName(1, 3, "Neumann Top")
gmsh.model.setPhysicalName(1, 4, "Neumann Bottom")

# Générer le maillage
gmsh.model.mesh.generate(2)

# Sauvegarder le maillage dans un fichier
gmsh.write("rectangular_domain_with_bcs.msh")

# Finaliser Gmsh
gmsh.finalize()


def get_coord(k):
    ax, ay = Cells_Cord[k][0][0], Cells_Cord[k][0][1]
    bx, by = Cells_Cord[k][1][0], Cells_Cord[k][1][1]
    cx, cy = Cells_Cord[k][2][0], Cells_Cord[k][2][1]
    return [(ax,ay), (bx,by), (cx,cy)]


def dist_modi(edge,k):
    triangles = faces.cellid[edge]
    vect1= get_coord(list(triangles).index(k))
    vect2 = get_coord(abs(1-list(triangles).index(k)))
    common_vertix = list(set(vect1) & set(vect2))
    if common_vertix !=[]:
        B1 = barycentre(vect1)
        B2 = barycentre(vect2)
        dist1 = distance_point_segment(B1,common_vertix[0],common_vertix[1])
        dist2 = distance_point_segment(B2, common_vertix[0],common_vertix[1])
        distance = (dist1 + dist2)
    else:
        B1 = barycentre(vect1)
        edge_node = faces._nodeid[edge]
        edge_vect =  [nodes._vertex[edge_node[h]][:-2] for h in range(2)]
        dist1 = distance_point_segment(B1,edge_vect[0],edge_vect[1])
        dist2 = 0
        distance = (dist1 + dist2)
    return distance



u = np.array(u)

# Récupération des indices des orthocentres avec ordonnées nulles
indices = np.where(Orthocentre[:, 1] == 0.0)[0]

# Récupération des orthocentres et des valeurs de u correspondantes
orthocentres_ordonnees_nulles = Orthocentre[indices]
u_correspondant = u[indices]

# Affichage des résultats
print("Orthocentres avec ordonnées nulles :", orthocentres_ordonnees_nulles)
print("Valeurs de u correspondantes :", u_correspondant)

# Plot des résultats
plt.figure()
plt.scatter(orthocentres_ordonnees_nulles[:, 0], u_correspondant, s=20, label='Orthocentres avec ordonnées nulles')
plt.xlabel('Coordonnée x des orthocentres')
plt.ylabel('Valeurs de u')
plt.legend()
plt.grid(True)
plt.title('Plot des orthocentres avec ordonnées nulles et leurs valeurs correspondantes')
plt.show()