import numpy as np
from numba import njit, prange

# Global parameters
taur = 0.9
rho = 0.1 / (0.9 + 0.1)
D = 0.1
speed = 30.0
kon = 18.0
koff = kon / 0.0062
alpha = 2.0
m0 = 1.0
ntar = 6.0
kr = 0.07
kb = 0.14
h = 10.0
kz = 2.0
ka = 3.0

def gradx_ecoli3D(NRep, NSteps, dt=1e-3, NBurn=10000, seeds=None, c0=400.0, grad=0.1, ttumble=0.1):
    if seeds is None:
        seeds = np.random.randint(0, 1000000, size=NRep)
    elif len(seeds) != NRep:
        raise ValueError("The length of seeds must be equal to NRep")
    
    values = gradx_simulate_ecoli3D(NRep, NSteps, dt, NBurn, seeds, c0, grad, ttumble)
    keys = ["pos_xyz", "actions", "concentrations", "methylation", "activity"]
    
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

    for idx_rep in prange(NRep):
        np.random.seed(seeds[idx_rep])
        x, y, z = 0., 0., 0.
        nx, ny, nz = np.random.randn(3)
        r = np.sqrt(nx**2 + ny**2 + nz**2)
        nx, ny, nz = nx / r, ny / r, nz / r
        a = a0
        c = c0
        m = m0 - np.log(2) / (ntar * alpha) + (np.log(1 + c / kon) - np.log(1 + c / koff)) / alpha
        yp = ka * a / (kz + ka * a)
        yp0 = yp * (taur / ttumble)**(1 / h)
        tr = taur

        it = 0
        while it <= total_steps:
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

            a = 1.0 / (1.0 + np.exp(ntar * (alpha * (m0 - m) + np.log(1.0 + c / kon) - np.log(1.0 + c / koff))))
            m += dt * (kr * (1.0 - a) - kb * a)
            yp += dt * (ka * a * (1.0 - yp) - kz * yp)
            tr = ttumble * (yp0 / yp)**h

            if it > NBurn and it <= total_steps:
                idx = it - NBurn - 1

                pos_xyz[idx_rep, 0, idx] = x
                pos_xyz[idx_rep, 1, idx] = y
                pos_xyz[idx_rep, 2, idx] = z

                actions[idx_rep, idx] = 0
                concentrations[idx_rep, idx] = c

                methylation[idx_rep, idx] = m
                activity[idx_rep, idx] = a
            
            it += 1

            if np.random.rand() < dt / tr:
                nx, ny, nz = np.random.randn(3)
                r = np.sqrt(nx**2 + ny**2 + nz**2)
                nx, ny, nz = nx / r, ny / r, nz / r
                time = 0
                tumble_duration = np.random.exponential(ttumble)
                while time < tumble_duration:
                    a = 1. / (1. + np.exp(ntar * (alpha * (m0 - m) + np.log(1 + c / kon) - np.log(1 + c / koff))))
                    m += dt * (kr * (1 - a) - kb * a)
                    yp += dt * (ka * a * (1 - yp) - kz * yp)
                    tr = ttumble * (yp0 / yp)**h
                    c = c0 + grad * x

                    time += dt

                    if it > NBurn and it <= total_steps:
                        idx = it - NBurn - 1
                        pos_xyz[idx_rep, 0, idx] = x
                        pos_xyz[idx_rep, 1, idx] = y
                        pos_xyz[idx_rep, 2, idx] = z

                        actions[idx_rep, idx] = 1
                        concentrations[idx_rep, idx] = c

                        methylation[idx_rep, idx] = m
                        activity[idx_rep, idx] = a

                    it += 1

    return pos_xyz, actions, concentrations, methylation, activity

def gradx_ecoli2D(NRep, NSteps, dt=1e-3, NBurn=10000, seeds=None, c0=400.0, grad=0.1, ttumble=0.1):
    if seeds is None:
        seeds = np.random.randint(0, 1000000, size=NRep)
    elif len(seeds) != NRep:
        raise ValueError("The length of seeds must be equal to NRep")
    
    values = gradx_simulate_ecoli2D(NRep, NSteps, dt, NBurn, seeds, c0, grad, ttumble)
    keys = ["pos_xy", "actions", "concentrations", "methylation", "activity"]
    
    results = []
    for i in range(NRep):
        traj_dict = {key: values[j][i] for j, key in enumerate(keys)}
        results.append(traj_dict)
    
    return results

@njit(parallel=True)
def gradx_simulate_ecoli2D(NRep, NSteps, dt, NBurn, seeds, c0, grad, ttumble):
    total_steps = NBurn + NSteps
    pos_xy = np.zeros((NRep, 2, NSteps + 1))
    actions = np.zeros((NRep, NSteps + 1))
    concentrations = np.zeros((NRep, NSteps + 1))
    methylation = np.zeros((NRep, NSteps + 1))
    activity = np.zeros((NRep, NSteps + 1))

    for ire in prange(NRep):
        np.random.seed(seeds[ire])
        x, y = 0.0, 0.0
        nx, ny = np.random.randn(2)
        r = np.sqrt(nx**2 + ny**2)
        nx, ny = nx / r, ny / r
        a = 1.0 / 3.0
        c = c0
        m = m0 - np.log(2.0) / (ntar * alpha) + (np.log(1.0 + c / kon) - np.log(1.0 + c / koff)) / alpha
        yp = ka * a / (kz + ka * a)
        yp0 = yp * (taur / ttumble)**(1.0 / h)
        tr = taur

        it = 0

        while it <= total_steps:
            it += 1

            nx += np.random.randn() * np.sqrt(2.0 * D * dt)
            ny += np.random.randn() * np.sqrt(2.0 * D * dt)
            r = np.sqrt(nx**2 + ny**2)
            nx, ny = nx / r, ny / r
            vx, vy = speed * nx, speed * ny
            x += vx * dt
            y += vy * dt
            c = c0 + grad * x

            if it > NBurn:
                idx = it - NBurn - 1
                pos_xy[ire, 0, idx] = x
                pos_xy[ire, 1, idx] = y

                actions[ire, idx] = 0
                concentrations[ire, idx] = c

                a = 1.0 / (1.0 + np.exp(ntar * (alpha * (m0 - m) + np.log(1.0 + c / kon) - np.log(1.0 + c / koff))))
                m += dt * (kr * (1.0 - a) - kb * a)
                yp += dt * (ka * a * (1.0 - yp) - kz * yp)
                tr = ttumble * (yp0 / yp)**h

                methylation[ire, idx] = m
                activity[ire, idx] = a
            
            r = np.random.uniform()
            if r < dt / tr:
                nx, ny = np.random.randn(2)
                r = np.sqrt(nx**2 + ny**2)
                nx, ny = nx / r, ny / r
                time = 0.0
                tumble_duration = np.random.exponential(ttumble)
                while time < tumble_duration:
                    it += 1

                    a = 1.0 / (1.0 + np.exp(ntar * (alpha * (m0 - m) + np.log(1.0 + c / kon) - np.log(1.0 + c / koff))))
                    m += dt * (kr * (1.0 - a) - kb * a)
                    yp += dt * (ka * a * (1.0 - yp) - kz * yp)
                    time += dt
                    tr = ttumble * (yp0 / yp)**h
                    c = c0 + grad * x

                    if it > NBurn:
                        idx = it - NBurn - 1
                        pos_xy[ire, 0, idx] = x
                        pos_xy[ire, 1, idx] = y

                        actions[ire, idx] = 1
                        concentrations[ire, idx] = c

                        methylation[ire, idx] = m
                        activity[ire, idx] = a

    return pos_xy, actions, concentrations, methylation, activity