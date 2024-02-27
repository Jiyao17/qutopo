

# functions related to purification


def purify_to_fth(f, f_th):
    """
    starting with fidelity f
    purify until target fidelity f_th
    return the # of steps and the fidelity after each step
    """
    if f >= f_th:
        return 0, [f]

    assert f >= 1 / 2
    assert f_th >= 1 / 2


    A = f
    B = (1 - f) / 3
    C = (1 - f) / 3
    D = (1 - f) / 3

    step = 0
    fedilities = [A]
    while A < f_th:
        N = (A + B)**2 + (C + D)**2
        A_ = (A**2 + B**2) / N
        B_ = 2 * C * D / N
        C_ = (C**2 + D**2) / N
        D_ = 2 * A * B / N

        A = A_
        B = B_
        C = C_
        D = D_

        step += 1
        fedilities.append(A)

    return step, fedilities

def purify_by_step(f, step_num):
    """
    starting with fidelity f
    purify step_num iterations
    return the fidelity after each step
    """


    A = f
    B = (1 - f) / 3
    C = (1 - f) / 3
    D = (1 - f) / 3

    step = 0
    fedilities = [A]
    while step < step_num:
        N = (A + B)**2 + (C + D)**2
        A_ = (A**2 + B**2) / N
        B_ = 2 * C * D / N
        C_ = (C**2 + D**2) / N
        D_ = 2 * A * B / N

        A = A_
        B = B_
        C = C_
        D = D_

        step += 1
        fedilities.append(A)

    return step_num, fedilities

def quantum_swap(f1, f2):
    # swap two qubits with fidelity f1 and f2
    # return the fidelity of the new state

    # assert f1 >= 1 / 2
    # assert f2 >= 1 / 2
    
    new_f = 1/4 + 1/12 * (4 * f1 - 1) * (4 * f2 - 1)
    return new_f

def calc_epr_num(f, f_th):
    # calculate the number of qubits needed to purify f to f_th
    # under development, subject to change

    step_num, fedilities = purify_to_fth(f, f_th)
    num_eprs = 0
    if step_num == 0:
        num_eprs = 1
    else:
        num_eprs = step_num * 2 # assume pk = 1.0 for all steps
    return num_eprs

def asymetric_purify(f1, f2):
    f = (f1 * f2) / (f1 * f2 + (1 - f1) * (1 - f2))
    return f

def swap_dephased(f1, f2):
    f = f1*f2 + (1-f1)*(1-f2)
    return f

def swap_depolarized(f1, f2):
    f = f1*f2 + (1-f1)*(1-f2)/3
    return f

def swap_dephased_grad(f1, f2, p=1):
    if p == 1:
        grad = f2 - (1-f2)
    elif p == 2:
        grad = f1 - (1-f1)
    else:
        raise ValueError('p must be 1 or 2')
    return grad

def swap_depolarized_grad(f1, f2, p=1):
    if p == 1:
        grad = f2 - (1-f2)/3
    elif p == 2:
        grad = f1 - (1-f1)/3
    else:
        raise ValueError('p must be 1 or 2')
    return grad

def purify_dephased(f1, f2):
    f = (f1 * f2) / (f1 * f2 + (1 - f1) * (1 - f2))
    return f

def purify_depolarized(f1, f2):
    nume = f1*f2 + (1-f1)*(1-f2)/9
    deno = f1*f2 + f1*(1-f2)/3 + (1-f1)*f2/3 + (1-f1)*(1-f2)*5/9
    return nume / deno

def purify_dephased_grad(f1, f2, p):
    deno = ((f1 * f2 + (1 - f1) * (1 - f2)))**2
    if p == 1:
        nume = f2 * (f1 * f2 + (1 - f1) * (1 - f2)) - f1*f2 * (f2 - (1 - f2))
    elif p == 2:
        nume = f1 * (f1 * f2 + (1 - f1) * (1 - f2)) - f1*f2 * (f1 - (1 - f1))
    else:
        raise ValueError('p must be 1 or 2')

    return nume / deno

def purify_depolarized_grad(f1, f2, p):
    nume_purify = f1*f2 + (1-f1)*(1-f2)/9
    deno_purify = f1*f2 + f1*(1-f2)/3 + (1-f1)*f2/3 + (1-f1)*(1-f2)*5/9
    deno = deno_purify ** 2
    if p == 1:
        nume = (f2 - (1-f2)/9)*deno_purify - nume_purify*(f2 + (1-f2)/3 - f2/3 - (1-f2)*5/9)
    elif p == 2:
        nume = (f1 - (1-f1)/9)*deno_purify - nume_purify*(f1 + (1-f1)/3 - f1/3 - (1-f1)*5/9)
    else:
        raise ValueError('p must be 1 or 2')

    return nume / deno

if __name__ == '__main__':
    # the graph here reproduces Fig. 1 in
    # Quantum privacy amplification and the security of quantum cryptography over noisy channels
    
    import matplotlib.pyplot as plt
    import numpy as np

    # plot the 3d surface
    # x: number of purification iterations
    # y: initial fidelity
    # z: target fidelity

    initial_fidelity = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    f_th = 0.99

    results = []
    for f in initial_fidelity:
        step, fedilities = purify_by_step(f, 10)

        print(f, step, fedilities)
        if step < 10:
            fedilities += [fedilities[-1]] * (10 - step)
        elif step > 10:
            fedilities = fedilities[:10]
        results.append((f, 10, fedilities))

    X = np.arange(0, 11, 1)
    Y = np.array(initial_fidelity)
    Z = np.ndarray((len(initial_fidelity), 11))
    for i, (f, step, fedilities) in enumerate(results):
        for j, fidelity in enumerate(fedilities):
            Z[i, j] = fidelity

    X, Y = np.meshgrid(X, Y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('Number of Iterations')
    ax.set_ylabel('Initial Fidelity')
    ax.set_zlabel('Target Fidelity')

    plt.show()


