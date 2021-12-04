from sympy import SparseMatrix

from HW3.cmu_quantum_experience.circuit import Circuit
from HW3.cmu_quantum_experience.helpers import print_state


def main():
    print('Quantum Circuit Simulator')

    circuit_file = 'e_circuit.txt'
    circuit = Circuit.from_file(circuit_file)
    circuit.verbose = False

    for initial_state in range(2 ** circuit.n_qubits):
        result = circuit.run(SparseMatrix(2 ** circuit.n_qubits, 1, {(initial_state, 0): 1}))

        if circuit.measure:
            print('Measurement Result:', result)
        else:
            print(f'Final State ({format(initial_state, f"0{circuit.n_qubits}b")}):')
            print_state(result, circuit.n_qubits)


if __name__ == '__main__':
    main()
