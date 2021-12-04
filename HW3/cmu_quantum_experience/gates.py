from abc import abstractmethod, ABC
from typing import List, Tuple

import numpy as np
import sympy
from sympy import Matrix, Expr, Integer, SparseMatrix, I, Float


def generate_bit_lists(n_bits: int):
    binary_strings = []

    def genbin(n: int, bs: List[int]):
        if len(bs) == n:
            binary_strings.append(bs)
        else:
            genbin(n, bs + [0])
            genbin(n, bs + [1])

    genbin(n_bits, [])

    return binary_strings


def bit_list_to_int(bit_list: List[int]):
    out = 0
    for bit in bit_list:
        out = (out << 1) | bit
    return out


def parse_bit_list_sparse_vector(sparse_vector: List[Tuple[Expr, List[int]]], vector_size: int) -> SparseMatrix:
    values = {(bit_list_to_int(index), 0): value for value, index in sparse_vector}
    return SparseMatrix(vector_size, 1, values)


class Gate(ABC):
    def __init__(self, arguments: List[int]):
        self.arguments = arguments

    def apply(self, state: List[Expr]) -> Matrix:
        N = len(state)
        n = int(np.log2(N))

        mapped_basis = [self._map_basis_vector(x) for x in generate_bit_lists(n)]

        mapped_basis_parsed = [parse_bit_list_sparse_vector(x, N) for x in mapped_basis]

        gate_matrix = Matrix(mapped_basis_parsed).reshape(N, N).T

        # from sympy import pprint
        # pprint(gate_matrix)

        return gate_matrix * state

    def __str__(self):
        return f'{self.__class__.__name__} {" ".join([str(x) for x in self.arguments])}'

    @abstractmethod
    def _map_basis_vector(self, bits: List[int]) -> List[Tuple[Expr, List[int]]]:
        pass


class SingleQubitMatrixGate(Gate, ABC):
    @abstractmethod
    def _matrix(self) -> Matrix:
        pass

    def _map_basis_vector(self, bits: List[int]) -> List[Tuple[Expr, List[int]]]:
        operand = self.arguments[-1]

        bits_zero = bits.copy()
        bits_one = bits.copy()

        bits_zero[operand] = 0
        bits_one[operand] = 1

        matrix = self._matrix()

        if bits[operand] == 0:
            r = [(matrix[0, 0], bits_zero), (matrix[1, 0], bits_one)]
        else:
            r = [(matrix[0, 1], bits_zero), (matrix[1, 1], bits_one)]

        return r


class NotGate(SingleQubitMatrixGate):
    def _matrix(self) -> Matrix:
        return Matrix([
            [0, 1],
            [1, 0]
        ])


class SqrtNotGate(SingleQubitMatrixGate):
    def _matrix(self) -> Matrix:
        return Integer(1) / Integer(2) * Matrix([
            [1 + I, 1 - I],
            [1 - I, 1 + I]
        ])


class RotateGate(SingleQubitMatrixGate):
    def _matrix(self) -> Matrix:
        degree = self.arguments[0] / 180 * sympy.pi

        return Matrix([
            [sympy.cos(degree), -sympy.sin(degree)],
            [sympy.sin(degree), sympy.cos(degree)]
        ])


class PhaseGate(SingleQubitMatrixGate, ABC):
    _phase: Expr

    def _matrix(self) -> Matrix:
        phase = sympy.exp(sympy.pi * (self._phase / 180) * I)
        return Matrix([
            [1, 0],
            [0, phase]
        ])


class ZGate(PhaseGate):
    _phase = 180


class SGate(PhaseGate):
    _phase = 90


class TGate(PhaseGate):
    _phase = 45


class TDagGate(PhaseGate):
    _phase = -45


class HadamardGate(SingleQubitMatrixGate):
    def _matrix(self) -> Matrix:
        return (1 / sympy.sqrt(2)) * Matrix([
            [1, 1],
            [1, -1]
        ])


class CNotGate(Gate):
    def _map_basis_vector(self, bits: List[int]) -> List[Tuple[Expr, List[int]]]:
        control_bit = self.arguments[0]
        target_bit = self.arguments[1]

        new_bits = bits.copy()
        if new_bits[control_bit] == 1:
            new_bits[target_bit] = 1 - new_bits[target_bit]

        return [(Integer(1), new_bits)]


class SwapGate(Gate):
    def _map_basis_vector(self, bits: List[int]) -> List[Tuple[Expr, List[int]]]:
        bit_1 = self.arguments[0]
        bit_2 = self.arguments[1]

        new_bits = bits.copy()
        new_bits[bit_1] = bits[bit_2]
        new_bits[bit_2] = bits[bit_1]

        return [(Integer(1), new_bits)]


class CSwapGate(Gate):
    def _map_basis_vector(self, bits: List[int]) -> List[Tuple[Expr, List[int]]]:
        control_bit = self.arguments[0]
        bit_1 = self.arguments[1]
        bit_2 = self.arguments[2]

        new_bits = bits.copy()
        if bits[control_bit]:
            new_bits[bit_1] = bits[bit_2]
            new_bits[bit_2] = bits[bit_1]

        return [(Integer(1), new_bits)]


class CCNotGate(Gate):
    def _map_basis_vector(self, bits: List[int]) -> List[Tuple[Expr, List[int]]]:
        control_bit_1 = self.arguments[0]
        control_bit_2 = self.arguments[1]
        target_bit = self.arguments[2]

        new_bits = bits.copy()
        if bits[control_bit_1] & bits[control_bit_2]:
            new_bits[target_bit] = 1 - bits[target_bit]

        return [(Integer(1), new_bits)]


class Measure:
    pass
