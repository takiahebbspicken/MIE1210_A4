# Interior nodes
        if i > mesh.ygrid:
            # East face velocity
            mesh.vel_face[i, 0] = (mesh.velocities[i] + mesh.velocites[i + mesh.ygrid]) / 2 \
                                  + (1 / 2)\
                                  * ((mesh.volumes[i]/mesh.a_momentum[i]) *
                                     ((mesh.pressure[i + mesh.ygrid*2]-mesh.pressure[i])
                                      / (2*mesh.areas[i + mesh.ygrid, 1])) +
                                     (mesh.volumes[i+mesh.ygrid] / mesh.a_momentum[i + mesh.ygrid]) *
                                     ((mesh.pressure[i + mesh.ygrid] - mesh.pressure[i - mesh.ygrid]) /
                                      (2 * mesh.areas[i, 1]))) \
                                  - (mesh.volumes[i] / 2 + mesh.volumes[i+mesh.ygrid] / 2) * \
                                  (1 / mesh.a_momentum[i + mesh.ygrid] + 1 / mesh.a_momentum[i + mesh.ygrid]) * \
                                  ((mesh.pressure[i+mesh.ygrid] - mesh.pressure[i]) /
                                   (2*mesh.dx[mesh.column + 1]))

            # North face velocity
            mesh.vel_face[i, 1] = (mesh.velocities[i] + mesh.velocites[i + mesh.ygrid]) / 2 + (1 / 2)
        # Nodes next to left boundary

        # Nodes two from right boundary

        # Nodes two from top boundary