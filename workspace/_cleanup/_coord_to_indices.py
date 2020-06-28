coords = [ (1,0), (2,99), (4,201), (51,2), (49,102), (53,199) ]


def _coord_to_indices(_coords, _maxshift):
    ndim = len(_coords[0])
    idxs = []
    for i in range(ndim):
        cs = sorted([coord[i] for coord in _coords])
        c0 = cs[0]
        ds = []
        i0 = 0
        for cc in cs:
            i0 += (cc-c0 > _maxshift)
            ds.append(i0)
            c0 = cc
        idx = tuple(ds[cs.index(coord[i])] for coord in _coords)
        idxs.append(idx)
    return [i for i in zip(*idxs)]

idx = _coord_to_indices(coords, 20)
print(idx)
