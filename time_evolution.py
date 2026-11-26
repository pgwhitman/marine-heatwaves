import numpy as np

def ssprk3_step(state, dt, rhs):

    # Stage 1
    k1 = rhs(state)
    stage1 = State_increment(state, k1, dt)

    # Stage 2
    k2 = rhs(stage1)
    temp = State_increment(stage1, k2, dt)
    stage2 = state.combine(temp, 0.75, 0.25)

    # Stage 3
    k3 = rhs(stage2)
    temp = State_increment(stage2, k3, dt)
    new_state = state.combine(temp, 1.0 / 3.0, 2.0 / 3.0)

    return new_state


def State_increment(state, derivative, dt):
    """Return state + dt*derivative."""
    return state.combine(derivative, 1.0, dt)


def evolve(state, model, config):

    nsteps = int(config.t_end / config.dt)

    initial_mass = np.sum(state.D) * config.dx * config.dy

    for n in range(nsteps):

        state = ssprk3_step(state, config.dt, model.rhs)

        # Print progress every 100 steps
        if (n + 1) % 100 == 0:

            time = (n + 1) * config.dt

            mass = np.sum(state.D) * config.dx * config.dy
            min_depth = np.min(state.D)
            max_depth = np.max(state.D)

            print(
                f"Step: {n+1:6d}, "
                f"Time: {time:.4f}, "
                f"Mass: {mass:.8f}, "
                f"Min D: {min_depth:.6f}, "
                f"Max D: {max_depth:.6f}"
            )

    final_mass = np.sum(state.D) * config.dx * config.dy

    print("\nSimulation completed!")
    print("Initial mass:", initial_mass)
    print("Final mass:", final_mass)
    print("Mass difference:", final_mass - initial_mass)

    return state