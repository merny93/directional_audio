import numpy as np
from functools import reduce
from bresenham import bresenhamline
N = 50
BC = np.zeros((N,N), dtype=np.bool_)
BC[:,0] = True
BC[:,-1] = True
# BC[0,:] = True
BC[-1,:] = True
BC[N//2,:] = True
BC[N//2 +1,:] = True


def do_wave(BC, wave_origin, t0 = 0.0):
    assert(type(wave_origin) is tuple)#will save heartach later

    ##set speed of sound
    speed_sound = 1
    #get dimensionality of space
    dims = len(wave_origin)

    #generate the time grid and set it to the initial time
    time_grid = np.ones_like(BC, dtype=float)*t0

    #generate basis with zero around the origin
    basis = [np.arange(-wave_origin[i], BC.shape[i] - wave_origin[i]) for i in range(dims)]
    #create a mesh
    m_grid = np.meshgrid(*basis[::-1])
    #stack it to another dimension
    m_grid_ar = np.stack(m_grid)
    #get disntance
    dists = np.linalg.norm(m_grid_ar, axis=0)
    #and use speed of sound to add on
    time_grid = time_grid + (dists/speed_sound)

    ##now we worry about what get hit
    visited = np.zeros_like(BC, dtype=np.bool_)
    #generate the rays that land in boundary
    #get coordinates of all the boundary
    edge_idx = []
    edge_idx.append([(0,i) for i in range(BC.shape[1])]) #top
    edge_idx.append([(i,BC.shape[1]) for i in range(0,BC.shape[1])]) #right
    edge_idx.append([(BC.shape[0],i) for i in range(BC.shape[1])]) #bottom
    edge_idx.append([(i,-1) for i in range(BC.shape[1])]) #left
    edge_idx = [x for y in edge_idx for x in y]

    #loop through all the points
    for idx in edge_idx:
        #simplify the rise over run
        sp = np.array(wave_origin)[np.newaxis, :]
        idx= np.array(idx)[np.newaxis, :]

        #init the move list
        #it contains the id of the basis vector to take
        move_list = bresenhamline(sp,idx)
        move_list = [tuple(move_list[s,:]) for s in range(move_list.shape[0])]
        for cur_pos in move_list:
            try:
                visited[cur_pos]= True
            except:
                break
            if BC[cur_pos]:
                #hit a boundary!!!
                break
    visited[wave_origin] = True
    return visited, time_grid


def simulate_bounces(BC, source, mics, n_bounce = 2):
    sources = [(source, 0.0)]
    res = [[] for _ in range(len(mics))]
    all_t = []
    for bounce_n in range(n_bounce):
        new_s = []
        for s in sources:
            v,t =do_wave(BC, s[0], s[1])
            all_t.append(t*v)
            # print(s)
            # plt.imshow(v*t)
            # plt.show()
            for mic_n,mic_loc in enumerate(mics):
                if v[mic_loc]:
                    res[mic_n].append(t[mic_loc])
            for idx, val in np.ndenumerate(v):
                if val and BC[idx]:
                    new_s.append(tuple([idx, t[idx]]))
        sources = new_s
    all_t = np.array(all_t)
    frames = []
    plt.ion()
    plt.clf()
    for t_step in range(100):
        frame = np.count_nonzero(np.logical_and(all_t>t_step, all_t<(t_step+1)), axis=0)
        frames.append(frame)
        plt.clf()
        plt.imshow(frame)
        plt.show()
        plt.pause(0.2)
        print(t_step)
    return res
               



import matplotlib.pyplot as plt

plt.imshow(BC)
plt.show()
r = simulate_bounces(BC, (10,10), [(4,3), (4,5)])
# print(r)



