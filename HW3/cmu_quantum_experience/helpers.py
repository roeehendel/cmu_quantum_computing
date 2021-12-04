import sympy
from sympy import Matrix


def print_state(state: Matrix, n_qubits: int):
    print(''.join([str(x) for x in range(n_qubits)]))
    print('-' * 10)

    for i, x in enumerate(state):
        print(format(i, f'0{n_qubits}b'), sympy.simplify(x))
