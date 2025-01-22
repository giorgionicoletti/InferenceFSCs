import numpy as np
from numba import njit, prange
import utils

# Global parameters
taur = 0.9
D = 0.1
speed = 30.0
koff = 18.0 
kon = 3000 #kon / 0.0062
alpha = 2.0 #OK
m0 = 1 #0.5
ntar = 6.0
kr = 0.1 #0.07
kb = 0.2 #0.14
h = 10.0 #OK
kz = 2.0 #OK
ka = 3.0 #OK


# def gradx_ecoli3D_FSC(FSC, NRep, NSteps, dt=1e-3, NBurn=10000, seeds=None, c0=400.0, grad=0.1,):
#     total_steps = NBurn + NSteps
#     pos_xyz = np.zeros((NRep, 3, NSteps))
#     actions = np.zeros((NRep, NSteps))
#     concentrations = np.zeros((NRep, NSteps))
#     memories = np.zeros((NRep, NSteps))

#     for idx_rep in prange(NRep):
#         np.random.seed(seeds[idx_rep])
#         x, y, z = 0., 0., 0.
#         nx, ny, nz = np.random.randn(3)
#         r = np.sqrt(nx**2 + ny**2 + nz**2)
#         nx, ny, nz = nx / r, ny / r, nz / r
#         c = c0
#         m = utils.numba_random_choice(FSC.generator.InternalMemSpace, FSC.rho)

#         it = 0
#         while it <= total_steps:

#             f = np.array([1.0, c])

#             a, m = FSC._nb_trajectory_step(FSC.generator._nb_get_TMat, FSC.generator.InternalMemSpace,
#                                                    FSC.generator.InternalMemActSpace, FSC.theta, m, f)
            
#             if a == 1:
#                 nx, ny, nz = np.random.randn(3)
#                 r = np.sqrt(nx**2 + ny**2 + nz**2)
#                 nx, ny, nz = nx / r, ny / r, nz / r
#             else:
#                 nx += np.random.randn() * np.sqrt(2.0 * D * dt)
#                 ny += np.random.randn() * np.sqrt(2.0 * D * dt)
#                 nz += np.random.randn() * np.sqrt(2.0 * D * dt)
#                 r = np.sqrt(nx**2 + ny**2 + nz**2)
#                 nx, ny, nz = nx / r, ny / r, nz / r
#                 vx, vy, vz = speed * nx, speed * ny, speed * nz
#                 x += vx * dt
#                 y += vy * dt
#                 z += vz * dt

#             c = c0 + grad * x

#             if it > NBurn and it <= total_steps:
#                 idx = it - NBurn - 1

#                 pos_xyz[idx_rep, 0, idx] = x
#                 pos_xyz[idx_rep, 1, idx] = y
#                 pos_xyz[idx_rep, 2, idx] = z

#                 actions[idx_rep, idx] = a
#                 concentrations[idx_rep, idx] = c
#                 memories[idx_rep, idx] = m
            
#             it += 1


def gradx_ecoli3D(NRep, NSteps, dt=1e-3, NBurn=10000, seeds=None, c0=400.0, grad=0.1, ttumble=0.1):
    if seeds is None:
        seeds = np.random.randint(0, 1000000, size=NRep)
    elif len(seeds) != NRep:
        raise ValueError("The length of seeds must be equal to NRep")
    
    values = gradx_simulate_ecoli3D(NRep, NSteps, dt, NBurn, seeds, c0, grad, ttumble)
    keys = ["pos_xyz", "actions", "concentrations", "methylation", "activity", "chey"]
    
    results = []
    for i in range(NRep):
        traj_dict = {key: values[j][i] for j, key in enumerate(keys)}
        traj_dict["seed"] = seeds[i]
        results.append(traj_dict)
    
    return results

@njit(parallel=True)
def gradx_simulate_ecoli3D(NRep, NSteps, dt, NBurn, seeds, c0, grad, ttumble, a0 = 1/3):
    total_steps = NBurn + NSteps
    pos_xyz = np.zeros((NRep, 3, NSteps))
    actions = np.zeros((NRep, NSteps))
    concentrations = np.zeros((NRep, NSteps))
    methylation = np.zeros((NRep, NSteps))
    activity = np.zeros((NRep, NSteps))
    chey = np.zeros((NRep, NSteps))

    for idx_rep in prange(NRep):
        np.random.seed(seeds[idx_rep])
        a = a0
        c = c0
        m = m0 - np.log(2) / (ntar * alpha) + (np.log(1 + c / koff) - np.log(1 + c / kon)) / alpha
        yp = ka * a / (kz + ka * a)
        yp0 = yp * (taur / ttumble)**(1 / h)
        tr = taur

        x, y, z = 0., 0., 0.
        nx, ny, nz = np.random.randn(3)
        r = np.sqrt(nx**2 + ny**2 + nz**2)
        nx, ny, nz = nx / r, ny / r, nz / r

        it = 0
        while it <= total_steps:
            a = 1.0 / (1.0 + np.exp(ntar * (alpha * (m0 - m) + np.log(1.0 + c / koff) - np.log(1.0 + c / kon))))
            m += dt * (kr * (1.0 - a) - kb * a)
            yp += dt * (ka * a * (1.0 - yp) - kz * yp)
            tr = ttumble * (yp0 / yp)**h

            if np.random.rand() < dt / tr:
                nx, ny, nz = np.random.randn(3)
                r = np.sqrt(nx**2 + ny**2 + nz**2)
                nx, ny, nz = nx / r, ny / r, nz / r
                time = 0
                tumble_duration = np.random.exponential(ttumble)

                while time <= tumble_duration:
                    if it > NBurn and it <= total_steps:
                        idx = it - NBurn - 1
                        pos_xyz[idx_rep, 0, idx] = x
                        pos_xyz[idx_rep, 1, idx] = y
                        pos_xyz[idx_rep, 2, idx] = z
                        concentrations[idx_rep, idx] = c

                        actions[idx_rep, idx] = 1

                        methylation[idx_rep, idx] = m
                        activity[idx_rep, idx] = a
                        chey[idx_rep, idx] = yp

                    a = 1. / (1. + np.exp(ntar * (alpha * (m0 - m) + np.log(1 + c / koff) - np.log(1 + c / kon))))
                    m += dt * (kr * (1 - a) - kb * a)
                    yp += dt * (ka * a * (1 - yp) - kz * yp)
                    tr = ttumble * (yp0 / yp)**h
                    c = c0 + grad * x

                    time += dt
                    it += 1
                    

            if it > NBurn and it <= total_steps:
                idx = it - NBurn - 1
                actions[idx_rep, idx] = 0

                pos_xyz[idx_rep, 0, idx] = x
                pos_xyz[idx_rep, 1, idx] = y
                pos_xyz[idx_rep, 2, idx] = z
                concentrations[idx_rep, idx] = c

                methylation[idx_rep, idx] = m
                activity[idx_rep, idx] = a
                chey[idx_rep, idx] = yp

            nx += np.random.randn() * np.sqrt(2.0 * D * dt)
            ny += np.random.randn() * np.sqrt(2.0 * D * dt)
            nz += np.random.randn() * np.sqrt(2.0 * D * dt)
            r = np.sqrt(nx**2 + ny**2 + nz**2)
            nx, ny, nz = nx / r, ny / r, nz / r
            vx, vy, vz = speed * nx, speed * ny, speed * nz
            x += vx * dt
            y += vy * dt
            z += vz * dt
            c = c0 + grad * x
            
            it += 1


    return pos_xyz, actions, concentrations, methylation, activity, chey

def cswitch_ecoli3D(NRep, NSteps1, NSteps2, dt=1e-3, NBurn=10000, seeds=None, c0=400.0, c1=40.0, ttumble=0.1):
    if seeds is None:
        seeds = np.random.randint(0, 1000000, size=NRep)
    elif len(seeds) != NRep:
        raise ValueError("The length of seeds must be equal to NRep")
    
    values = cswitch_simulate_ecoli3D(NRep, NSteps1, NSteps2, dt, NBurn, seeds, c0, c1, ttumble)
    keys = ["pos_xyz", "actions", "concentrations", "methylation", "activity", "chey"]
    
    results = []
    for i in range(NRep):
        traj_dict = {key: values[j][i] for j, key in enumerate(keys)}
        traj_dict["seed"] = seeds[i]
        results.append(traj_dict)
    
    return results

@njit(parallel=True)
def cswitch_simulate_ecoli3D(NRep, NSteps1, NSteps2, dt, NBurn, seeds, c0, c1, ttumble, a0 = 1/3):
    NSteps = NSteps1 + NSteps2
    total_steps = NBurn + NSteps
    pos_xyz = np.zeros((NRep, 3, NSteps))
    actions = np.zeros((NRep, NSteps))
    concentrations = np.zeros((NRep, NSteps))
    methylation = np.zeros((NRep, NSteps))
    activity = np.zeros((NRep, NSteps))
    chey = np.zeros((NRep, NSteps))

    for idx_rep in prange(NRep):
        np.random.seed(seeds[idx_rep])
        a = a0
        c = c0
        m = m0 - np.log(2) / (ntar * alpha) + (np.log(1 + c / koff) - np.log(1 + c / kon)) / alpha
        yp = ka * a / (kz + ka * a)
        yp0 = yp * (taur / ttumble)**(1 / h)
        tr = taur

        x, y, z = 0., 0., 0.
        nx, ny, nz = np.random.randn(3)
        r = np.sqrt(nx**2 + ny**2 + nz**2)
        nx, ny, nz = nx / r, ny / r, nz / r

        it = 0
        while it <= total_steps:
            a = 1.0 / (1.0 + np.exp(ntar * (alpha * (m0 - m) + np.log(1.0 + c / koff) - np.log(1.0 + c / kon))))
            m += dt * (kr * (1.0 - a) - kb * a)
            yp += dt * (ka * a * (1.0 - yp) - kz * yp)
            tr = ttumble * (yp0 / yp)**h

            if np.random.rand() < dt / tr:
                nx, ny, nz = np.random.randn(3)
                r = np.sqrt(nx**2 + ny**2 + nz**2)
                nx, ny, nz = nx / r, ny / r, nz / r
                time = 0
                tumble_duration = np.random.exponential(ttumble)

                while time <= tumble_duration:
                    if it > NBurn and it <= total_steps:
                        idx = it - NBurn - 1
                        pos_xyz[idx_rep, 0, idx] = x
                        pos_xyz[idx_rep, 1, idx] = y
                        pos_xyz[idx_rep, 2, idx] = z
                        concentrations[idx_rep, idx] = c

                        actions[idx_rep, idx] = 1

                        methylation[idx_rep, idx] = m
                        activity[idx_rep, idx] = a
                        chey[idx_rep, idx] = yp

                    a = 1. / (1. + np.exp(ntar * (alpha * (m0 - m) + np.log(1 + c / koff) - np.log(1 + c / kon))))
                    m += dt * (kr * (1 - a) - kb * a)
                    yp += dt * (ka * a * (1 - yp) - kz * yp)
                    tr = ttumble * (yp0 / yp)**h
                    time += dt
                    it += 1

            if it > NBurn and it <= total_steps:
                idx = it - NBurn - 1
                actions[idx_rep, idx] = 0

                pos_xyz[idx_rep, 0, idx] = x
                pos_xyz[idx_rep, 1, idx] = y
                pos_xyz[idx_rep, 2, idx] = z
                concentrations[idx_rep, idx] = c

                methylation[idx_rep, idx] = m
                activity[idx_rep, idx] = a
                chey[idx_rep, idx] = yp

            nx += np.random.randn() * np.sqrt(2.0 * D * dt)
            ny += np.random.randn() * np.sqrt(2.0 * D * dt)
            nz += np.random.randn() * np.sqrt(2.0 * D * dt)
            r = np.sqrt(nx**2 + ny**2 + nz**2)
            nx, ny, nz = nx / r, ny / r, nz / r
            vx, vy, vz = speed * nx, speed * ny, speed * nz
            x += vx * dt
            y += vy * dt
            z += vz * dt

            # Switch concentration after NBurn + NSteps1 timesteps
            if it == NBurn + NSteps1:
                c = c1

            it += 1

    return pos_xyz, actions, concentrations, methylation, activity, chey