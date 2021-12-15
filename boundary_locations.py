for i in range(mesh.num_nodes):
    if idx == 0:  # Bottom left boundary corner

    elif idx == mesh.ygrid - 1:  # Top left corner boundary

    elif idx == mesh.num_nodes - mesh.ygrid:  # Bottom right corner boundary

    elif idx == mesh.num_nodes - 1:  # Top right corner boundary

    elif idx < mesh.ygrid:  # Left boundary

    elif idx >= mesh.num_nodes - mesh.ygrid:  # Right boundary

    elif idx % mesh.ygrid == 0 and idx != 0:  # Bottom boundary

    elif idx % mesh.ygrid == (mesh.ygrid - 1):  # Top boundary

    else:
