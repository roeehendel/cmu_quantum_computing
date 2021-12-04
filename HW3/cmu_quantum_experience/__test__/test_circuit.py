import sympy
from sympy import Matrix

from HW3.cmu_quantum_experience.circuit import Circuit


def test_cswap():
    circuit_file = 'test1.txt'
    circuit = Circuit.from_file(circuit_file)
    circuit.verbose = True

    result = circuit.run()

    assert result == Matrix([0, 0, 1 / sympy.sqrt(2), 0, 0, 1 / sympy.sqrt(2), 0, 0])
