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


def correct_face_velocities(mesh):
    for idx in range(mesh.num_nodes - mesh.ygrid):
        if idx >= mesh.ygrid * mesh.xgrid - mesh.ygrid:  # Right boundary adjacent
            mesh.vel_face_correction[idx, 0] = 0
            mesh.vel_face_correction[idx, 2] = (1 / mesh.a_momentum[idx - mesh.ygrid, 0] + 1 / mesh.a_momentum[
                idx, 0]) * (mesh.pressure_correction[idx-mesh.ygrid]
                            - mesh.pressure_correction[idx]) / 2 * mesh.areas[idx, 0]
        elif idx < mesh.ygrid:  # Left boundary adjacent
            mesh.vel_face_correction[idx, 0] = (1 / mesh.a_momentum[idx + mesh.ygrid, 0] + 1 / mesh.a_momentum[
                idx, 0]) * (mesh.pressure_correction[idx]
                            - mesh.pressure_correction[idx + mesh.ygrid]) / 2 * mesh.areas[idx, 0]
            mesh.vel_face_correction[idx, 2] = 0
        else:
            mesh.vel_face_correction[idx, 0] = (1 / mesh.a_momentum[idx + mesh.ygrid, 0] + 1 / mesh.a_momentum[
                idx, 0]) * (mesh.pressure_correction[idx]
                            - mesh.pressure_correction[idx + mesh.ygrid]) / 2 * mesh.areas[idx, 0]
            mesh.vel_face_correction[idx, 2] = (1 / mesh.a_momentum[idx - mesh.ygrid, 0] + 1 / mesh.a_momentum[
                idx, 0]) * (mesh.pressure_correction[idx - mesh.ygrid]
                            - mesh.pressure_correction[idx]) / 2 * mesh.areas[idx, 0]
        if idx % mesh.ygrid == (mesh.ygrid - 1):  # Top boundary adjacent
            mesh.vel_face_correction[idx, 1] = 0
            mesh.vel_face_correction[idx, 3] = (1 / mesh.a_momentum[idx - 1, 0] + 1 / mesh.a_momentum[idx, 0]) * \
                                               (mesh.pressure_correction[idx-1]
                                                - mesh.pressure_correction[idx]) / 2 * mesh.areas[idx, 1]
        elif idx % mesh.ygrid == 0 and idx != 0:  # Bottom boundary adjacent
            mesh.vel_face_correction[idx, 1] = (1 / mesh.a_momentum[idx + 1, 0] + 1 / mesh.a_momentum[idx, 0]) * \
                                               (mesh.pressure_correction[idx]
                                                - mesh.pressure_correction[idx + 1]) / 2 * mesh.areas[idx, 1]
            mesh.vel_face_correction[idx, 3] = 0
        else:
            mesh.vel_face_correction[idx, 1] = (1 / mesh.a_momentum[idx + 1, 0] + 1 / mesh.a_momentum[idx, 0]) * \
                                               (mesh.pressure_correction[idx]
                                                - mesh.pressure_correction[idx + 1]) / 2 * mesh.areas[idx, 1]
            mesh.vel_face_correction[idx, 3] = (1 / mesh.a_momentum[idx - 1, 0] + 1 / mesh.a_momentum[idx, 0]) * \
                                               (mesh.pressure_correction[idx-1]
                                                - mesh.pressure_correction[idx]) / 2 * mesh.areas[idx, 1]

        # mesh.vel_face_correction[idx + mesh.ygrid, 2] = mesh.vel_face_correction[idx, 0]
        # mesh.vel_face_correction[idx + 1, 3] = mesh.vel_face_correction[idx, 1]
    mesh.vel_face[:, 0] += mesh.vel_face_correction[:, 0]
    mesh.vel_face[:, 1] += mesh.vel_face_correction[:, 1]
    mesh.vel_face[:, 2] += mesh.vel_face_correction[:, 2]
    mesh.vel_face[:, 3] += mesh.vel_face_correction[:, 3]