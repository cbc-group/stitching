direction  = (1,1,1)
i_cursor   = [0,0,0]
tile_shape = [2,3,4]
snake      = False
axis_order = "xzy"

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

def walk(index, axis):
    while True:
        yield tuple(index)
        index, overflow = step(index, axis)
        if overflow:
            # current axis overflow
            for _axis in range(axis-1, -1, -1):
                index, overflow = step(index, _axis)
                if not overflow:
                    break
            else:
                return

print(axis_order)
print(tile_shape)

if (len(tile_shape) <= 2):
    axis_order = str.replace("z", "")
axisid     = { 'x':0, 'y':1, 'z':2 }
axis_order = [ axisid[lab] for lab in list(axis_order) ]

indices = [c for c in walk(i_cursor, len(tile_shape)-1)]

print(indices)
