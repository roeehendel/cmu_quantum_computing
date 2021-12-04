import sys
from typing import Union, Optional

from HW3.cmu_quantum_experience.gates import *
from HW3.cmu_quantum_experience.helpers import print_state


class Circuit:
    # GATE_NAME_TO_GATE = {
    #     'Not': NotGate,
    #     'Z': ZGate,
    #     'S': SGate,
    #     'CNot': CNotGate,
    #     'Hadamard': HadamardGate,
    #     'Swap': SwapGate,
    #     'CSwap': CSwapGate,
    #     'CCNot': CCNotGate,
    # }

    def __init__(self, gates: List[Gate], measure: bool, n_qubits: int, verbose: bool = False):
        self.n_qubits = n_qubits
        self.gates = gates
        self.measure = measure
        self.verbose = verbose

    @classmethod
    def from_file(cls, circuit_file: str):
        with open(circuit_file, 'r') as f:
            lines = f.read().split('\n')

        lines = cls._filter_lines(lines)

        if len(lines) == 0:
            raise ValueError('')

        arguments_lines, gates_lines, measure = cls._parse_program_parts(lines)

        circuit_arguments = cls._parse_arguments(arguments_lines)

        gates = list(map(cls._parse_gate, gates_lines))

        return cls(gates, measure, **circuit_arguments)

    def run(self, initial_state: Optional[Matrix] = None) -> Union[Matrix, str]:
        final_state = self._compute_final_state(initial_state)

        if self.measure:
            probabilities = list(sympy.matrix_multiply_elementwise(abs(final_state), abs(final_state)))
            result = np.random.choice(np.arange(2 ** self.n_qubits), p=probabilities)
            result = format(result, f'0{self.n_qubits}b')
            return result

        return final_state

    def _compute_final_state(self, initial_state: Optional[Matrix] = None) -> Matrix:
        if initial_state is not None:
            state = initial_state
        else:
            state = sympy.zeros(2 ** self.n_qubits, 1)
            state[0] = 1

        for gate in self.gates:
            state = gate.apply(state)
            if self.verbose:
                self._print_state_after_gate(state, gate)

            if not sympy.simplify(state.conjugate().dot(state)) == 1:
                raise Exception(f'State norm: {sympy.simplify(state.conjugate().dot(state))}')

        return state

    def _print_state_after_gate(self, state: Matrix, gate: Gate):
        print()
        print(f'After {str(gate)}')
        print()
        print_state(state, self.n_qubits)

    @classmethod
    def _filter_lines(cls, lines: List[str]) -> List[str]:
        lines = [line.strip() for line in lines]
        lines = [line for line in lines if not line.startswith('#')]
        lines = [line for line in lines if len(line) > 0]
        return lines

    @classmethod
    def _parse_program_parts(cls, lines: List[str]) -> Tuple[List[str], List[str], bool]:
        measure = False

        arguments_lines = [lines[0]]
        if lines[-1] == 'Measure':
            measure = True
            gates_lines = lines[1:-1]
        else:
            gates_lines = lines[1:]

        return arguments_lines, gates_lines, measure

    @classmethod
    def _parse_arguments(cls, lines):
        first_line = None
        try:
            first_line = lines.pop(0)
            n_qubits = int(first_line)
        except ValueError as e:
            raise ValueError(f'First line should be the number of qubits, got: {first_line}')

        return {
            'n_qubits': n_qubits
        }

    @classmethod
    def _parse_gate(cls, gate_description: str) -> Gate:
        gate_name, *gate_arguments = gate_description.split(' ')

        gate_arguments = list(map(int, gate_arguments))

        # gate_class = cls.GATE_NAME_TO_GATE[gate_name]
        gate_class = getattr(sys.modules[__name__], f'{gate_name}Gate')

        gate = gate_class(gate_arguments)

        return gate
