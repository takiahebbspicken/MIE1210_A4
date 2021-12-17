def pressure_derivatives(mesh, idx):  # TODO: pressure_boundary variable and check top calculations
    _ = None
    if idx >= mesh.num_nodes - mesh.ygrid:  # Right boundary
        if idx == mesh.num_nodes - 1:  # Top right corner
            dpx_p = (2 * mesh.pressure_boundaries[idx - mesh.ygrid * mesh.column[idx], 2] -
                     mesh.pressure[idx - mesh.ygrid] -
                     mesh.pressure[idx]) / (2 * mesh.areas[idx, 1])
            dpy_p = (2 * mesh.pressure_boundaries[mesh.column[idx], 1] - mesh.pressure[idx]
                     - mesh.pressure[idx - 1]) / (2 * mesh.areas[idx, 0])
        elif idx == mesh.num_nodes - mesh.ygrid:  # Bottom right corner
            dpx_p = (2 * mesh.pressure_boundaries[idx - mesh.ygrid * mesh.column[idx], 2] -
                     mesh.pressure[idx - mesh.ygrid] -
                     mesh.pressure[idx]) / (2 * mesh.areas[idx, 1])
            dpy_p = (2 * mesh.pressure_boundaries[mesh.column[idx], 3] - mesh.pressure[idx]
                     - mesh.pressure[idx - 1]) / (2 * mesh.areas[idx, 0])
        else:
            dpx_p = (2 * mesh.pressure_boundaries[idx - mesh.ygrid * mesh.column[idx], 2] -
                     mesh.pressure[idx - mesh.ygrid] -
                     mesh.pressure[idx]) / (2 * mesh.areas[idx, 1])
            dpy_p = (mesh.pressure[idx + 1] - mesh.pressure[idx - 1]) / (2 * mesh.areas[idx, 0])
        return _, dpx_p, _, _, dpy_p, _
    elif idx % mesh.ygrid == (mesh.ygrid - 1):  # Top boundary
        if idx == mesh.num_nodes - 1:  # Top left corner
            dpx_p = (mesh.pressure[idx + mesh.ygrid] + mesh.pressure[idx] - 2 * mesh.pressure_boundaries[idx, 0]) \
                    / (2 * mesh.areas[idx, 1])
            dpy_p = (2 * mesh.pressure_boundaries[mesh.column[idx], 1] - mesh.pressure[idx]
                     - mesh.pressure[idx - 1]) / (2 * mesh.areas[idx, 0])
        else:
            dpx_p = (mesh.pressure[idx + mesh.ygrid] - mesh.pressure[idx - mesh.ygrid]) / (2 * mesh.areas[idx, 1])
            dpy_p = (2 * mesh.pressure_boundaries[mesh.column[idx], 1] - mesh.pressure[idx]
                     - mesh.pressure[idx - 1]) / (2 * mesh.areas[idx, 0])
        return _, dpx_p, _, _, dpy_p, _
    else:  # Added interpolations for face velocity calculations
        if idx == 0:  # Bottom left boundary corner adjacent
            dpx_E = (mesh.pressure[idx + mesh.ygrid * 2] - mesh.pressure[idx]) / \
                    (2 * mesh.areas[idx + mesh.ygrid, 1])
            dpx_p = (mesh.pressure[idx + mesh.ygrid] + mesh.pressure[idx] - 2 * mesh.pressure_boundaries[idx, 0]) \
                    / (2 * mesh.areas[idx, 1])
            dpy_p = (mesh.pressure[idx + 1] + mesh.pressure[idx] - 2 * mesh.pressure_boundaries[idx, 3]) / (
                    2 * mesh.areas[idx, 0])
            dpx_e = (mesh.pressure[idx + mesh.ygrid] - mesh.pressure[idx]) / (2 * mesh.dx[mesh.column[idx] + 1])
            dpy_N = (mesh.pressure[idx + 2] - mesh.pressure[idx]) / (2 * mesh.areas[idx + mesh.ygrid, 0])
            dpy_n = (mesh.pressure[idx + 1] - mesh.pressure[idx]) / (
                    2 * mesh.dy[idx - mesh.column[idx] * mesh.ygrid + 1])
            # elif idx > mesh.num_nodes - 2 * mesh.ygrid:
        elif idx < mesh.ygrid:  # Left boundary adjacent
            if idx < mesh.num_nodes - 2 * mesh.ygrid:
                dpx_E = (mesh.pressure[idx + mesh.ygrid * 2] - mesh.pressure[idx]) \
                        / (2 * mesh.areas[idx + mesh.ygrid, 1])
            dpx_p = (mesh.pressure[idx + mesh.ygrid] + mesh.pressure[idx] - 2
                     * mesh.pressure_boundaries[idx, 0]) \
                    / (2 * mesh.areas[idx, 1])
            dpx_e = (mesh.pressure[idx + mesh.ygrid] - mesh.pressure[idx]) / (2 * mesh.dx[mesh.column[idx] + 1])
            dpy_N = (mesh.pressure[idx + 2] - mesh.pressure[idx]) / (2 * mesh.areas[idx + mesh.ygrid, 0])
            dpy_p = (mesh.pressure[idx + 1] - mesh.pressure[idx - 1]) / (2 * mesh.areas[idx, 0])
            dpy_n = (mesh.pressure[idx + 1] - mesh.pressure[idx]) / (
                    2 * mesh.dy[idx - mesh.column[idx] * mesh.ygrid + 1])
        elif idx >= mesh.num_nodes - 2 * mesh.ygrid:  # Two from right boundary
            dpx_E = (2 * mesh.pressure_boundaries[idx % mesh.ygrid + 1, 2] -
                     mesh.pressure[idx] - mesh.pressure[idx + mesh.ygrid]) \
                    / (2 * mesh.areas[idx + mesh.ygrid, 1])
            dpx_p = (mesh.pressure[idx + mesh.ygrid] - mesh.pressure[idx - mesh.ygrid]) / (2 * mesh.areas[idx, 1])
            dpx_e = (mesh.pressure[idx + mesh.ygrid] - mesh.pressure[idx]) / (2 * mesh.dx[mesh.column[idx] + 1])
            dpy_N = (mesh.pressure[idx + 2] - mesh.pressure[idx]) / (2 * mesh.areas[idx + mesh.ygrid, 0])
            dpy_p = (mesh.pressure[idx + 1] - mesh.pressure[idx - 1]) / (2 * mesh.areas[idx, 0])
            dpy_n = (mesh.pressure[idx + 1] - mesh.pressure[idx]) / (
                    2 * mesh.dy[idx - mesh.column[idx] * mesh.ygrid + 1])
        elif idx % mesh.ygrid == 0 and idx != 0:  # Bottom boundary adjacent
            dpx_E = (mesh.pressure[idx + mesh.ygrid * 2] - mesh.pressure[idx]) \
                    / (2 * mesh.areas[idx + mesh.ygrid, 1])
            dpy_p = (mesh.pressure[idx + 1] + mesh.pressure[idx] -
                     2 * mesh.pressure_boundaries[mesh.column[idx], 3]) / (2 * mesh.areas[idx, 0])
            dpx_p = (mesh.pressure[idx + mesh.ygrid] - mesh.pressure[idx - mesh.ygrid]) / (2 * mesh.areas[idx, 1])
            dpx_e = (mesh.pressure[idx + mesh.ygrid] - mesh.pressure[idx]) / (2 * mesh.dx[mesh.column[idx] + 1])
            dpy_N = (mesh.pressure[idx + 2] - mesh.pressure[idx]) / (2 * mesh.areas[idx + mesh.ygrid, 0])
            dpy_n = (mesh.pressure[idx + 1] - mesh.pressure[idx]) / (
                    2 * mesh.dy[idx - mesh.column[idx] * mesh.ygrid + 1])
        elif idx % mesh.ygrid == (mesh.ygrid - 2):  # Two from top boundary
            dpx_E = (mesh.pressure[idx + mesh.ygrid * 2] - mesh.pressure[idx]) \
                    / (2 * mesh.areas[idx + mesh.ygrid, 1])
            dpy_N = (2 * mesh.pressure_boundaries[mesh.column[idx], 1] - mesh.pressure[idx] - mesh.pressure[
                idx + 1]) / \
                    (2 * mesh.areas[idx + mesh.ygrid, 0])
            dpx_p = (mesh.pressure[idx + mesh.ygrid] - mesh.pressure[idx - mesh.ygrid]) / (2 * mesh.areas[idx, 1])
            dpx_e = (mesh.pressure[idx + mesh.ygrid] - mesh.pressure[idx]) / (2 * mesh.dx[mesh.column[idx] + 1])
            dpy_p = (mesh.pressure[idx + 1] - mesh.pressure[idx - 1]) / (2 * mesh.areas[idx, 0])
            dpy_n = (mesh.pressure[idx + 1] - mesh.pressure[idx]) / (
                    2 * mesh.dy[idx - mesh.column[idx] * mesh.ygrid + 1])
        else:
            dpx_E = (mesh.pressure[idx + mesh.ygrid * 2] - mesh.pressure[idx]) / (2 * mesh.areas[idx + mesh.ygrid, 1])
            dpx_p = (mesh.pressure[idx + mesh.ygrid] - mesh.pressure[idx - mesh.ygrid]) / (2 * mesh.areas[idx, 1])
            dpx_e = (mesh.pressure[idx + mesh.ygrid] - mesh.pressure[idx]) / (2 * mesh.dx[mesh.column[idx] + 1])
            dpy_N = (mesh.pressure[idx + 2] - mesh.pressure[idx]) / (2 * mesh.areas[idx + mesh.ygrid, 0])
            dpy_p = (mesh.pressure[idx + 1] - mesh.pressure[idx - 1]) / (2 * mesh.areas[idx, 0])
            dpy_n = (mesh.pressure[idx + 1] - mesh.pressure[idx]) / (
                    2 * mesh.dy[idx - mesh.column[idx] * mesh.ygrid + 1])
        return dpx_E, dpx_p, dpx_e, dpy_N, dpy_p, dpy_n

    if idx == 0:  # Bottom left boundary corner
        dpx_E =
        dpx_p =
        dpx_e =
        dpy_N =
        dpy_p =
        dpy_n =
    elif idx == mesh.ygrid - 1:  # Top left corner boundary
        dpx_E =
        dpx_p =
        dpx_e =
        dpy_N =
        dpy_p =
        dpy_n =

    elif idx == mesh.num_nodes - mesh.ygrid:  # Bottom right corner boundary
        dpx_E =
        dpx_p =
        dpx_e =
        dpy_N =
        dpy_p =
        dpy_n =

    elif idx == mesh.num_nodes - 1:  # Top right corner boundary
        dpx_E =
        dpx_p =
        dpx_e =
        dpy_N =
        dpy_p =
        dpy_n =

    elif idx < mesh.ygrid:  # Left boundary
        dpx_E =
        dpx_p =
        dpx_e =
        dpy_N =
        dpy_p =
        dpy_n =

    elif idx >= mesh.num_nodes - mesh.ygrid:  # Right boundary
        dpx_E =
        dpx_p =
        dpx_e =
        dpy_N =
        dpy_p =
        dpy_n =

    elif idx % mesh.ygrid == 0 and idx != 0:  # Bottom boundary
        dpx_E =
        dpx_p =
        dpx_e =
        dpy_N =
        dpy_p =
        dpy_n =

    elif idx % mesh.ygrid == (mesh.ygrid - 1):  # Top boundary
        dpx_E =
        dpx_p =
        dpx_e =
        dpy_N =
        dpy_p =
        dpy_n =

    else: