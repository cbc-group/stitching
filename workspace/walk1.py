direction  = (1,1)
i_cursor   = [0,0]
tile_shape = [2,3]
snake      = False
axis_order = "yx"

print(axis_order)
print(tile_shape)

if (len(tile_shape) <= 2):
    axis_order = axis_order.replace("z", "")
    axisid = { 'x':1, 'y':0 }
else:
    axisid = { 'x':2, 'y':1, 'z':0 }
axis_order = [ axisid[lab] for lab in list(axis_order) ]

# convert tile_shape from (Nx,Ny,Nz) to (Nz,Ny,Nx)
tile_shape = tile_shape[::-1]

def step(index, axis):
    overflow = True
    if direction[axis] < 0 and index[axis] <= 0:
        # negative overflow
        if snake:
            # toggle direction
            index[axis] = 0
            direction[axis] *= -1
        else:
            index[axis] = tile_shape[axis] - 1
    elif direction[axis] > 0 and index[axis] >= tile_shape[axis] - 1:
        # positive overflow
        if snake:
            # toggle direction
            index[axis] = tile_shape[axis] - 1
            direction[axis] *= -1
        else:
            index[axis] = 0
    else:
        # .. next step in current axis
        index[axis] += direction[axis]
        overflow = False
    return index, overflow

def walk(index, axis_order, aidx):
    while True:
        yield tuple(index)
        index, overflow = step(index, axis_order[aidx])
        if overflow:
            # current axis overflow
            for _aidx in range(1, len(axis_order)):
                index, overflow = step(index, axis_order[_aidx])
                if not overflow:
                    break
            else:
                return


aidx    = 0
indices = [c for c in walk(i_cursor, axis_order, aidx)]

print(indices)
