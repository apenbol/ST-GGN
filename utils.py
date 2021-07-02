
def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

import numpy as np
def get_sample_index(end,num_sample,start = 0):
    '''

    Args:
        end: length of non-zero sequence
        num_sample:
        start: default from zero

    Returns:
        uniform index
    '''

    groupstep = int(np.floor(end/num_sample))
    temp = []
    for index,i in enumerate(range(start,end,groupstep)):
        if index == num_sample-1:
            temp.append((i,end))
            break
        else:
            temp.append((i,i+groupstep))
    res = [np.random.randint(i[0],i[1]) for i in temp]

    return res
def test_get_sample_index():
    out = get_sample_index(10,3)
    print(out)
    print(len(out))