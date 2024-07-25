from functools import reduce
from itertools import product
import scipy.sparse as ss
import numpy as np
from binary_matrix import TYPE_BINARY, BinaryMatrix, inv, round_to_binary, rank
from collections import defaultdict

def get_x(num: int, q: int) -> int:
    """
    Get the x part of the integer symplectic representation of a Pauli string
    Args:
        num (int): The integer representation of a Pauli string
        q (int): The number of qubits
    Returns:
        int: The x part of the integer symplectic representation
    """
    return num & ((1 << q) - 1)


def get_z(num: int, q: int) -> int:
    """
    Get the z part of the integer symplectic representation of a Pauli string
    Args:
        num (int): The integer representation of a Pauli string
        q (int): The number of qubits
    Returns:
        int: The z part of the integer symplectic representation
    """
    return num >> q

def count_ones(num: int) -> int:
    """
    Count the number of ones in the binary representation of a number
    Args:
        num (int): The number to count
    Returns:
        int: The number of ones in the binary representation of the number
    """
    return bin(num).count("1")

def parity_ones(num: int) -> int:
    """
    Count the parity of the number of ones in the binary representation of a number
    Args:
        num (int): The number to count
    Returns:
        int: The number of ones in the binary representation of the number
    """
    return bin(num).count("1") % 2


def dot_prod_mod2(num1: int, num2: int) -> int:
    """
    Compute the dot product of the bitstring representation of two numbers, modulo 2
    Args:
        num1 (int): The first number
        num2 (int): The second number
    Returns:
        int: The dot product of the two numbers modulo 2
    """
    return parity_ones(num1 & num2)


class PauliStringSum(defaultdict):
    """
    Representation of linear combinations of Pauli strings
    """
    def __init__(self, n_qubits: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_qubits = n_qubits
    
    def __mul__(self, other):
        if isinstance(other, PauliStringSum):
            if not self:
                return other
            if not other:
                return self

            result = PauliStringSum(self.n_qubits, self.default_factory)
            for s1, coef1 in self.items():
                for s2, coef2 in other.items():
                    phase = -1 if dot_prod_mod2(get_z(s1, self.n_qubits), get_x(s2, self.n_qubits)) else 1
                    prod_value = coef1 * coef2 * phase
                    if prod_value:
                        result[s1 ^ s2] += prod_value
            return result
        else: # scalar multiplication
            result = PauliStringSum(self.n_qubits, self.default_factory)
            for s, coef in self.items():
                result[s] = coef * other
            return result
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __imul__(self, other):
        if isinstance(other, PauliStringSum):
            return self.__mul__(other)
        else: # scalar multiplication
            for s in self:
                self[s] *= other
            return self

    def __add__(self, other):
        return PauliStringSum(self.n_qubits, self.default_factory, {s: self[s] + other[s] for s in set(self) | set(other)})
    
    def __iadd__(self, other):
        for s, coef in other.items():
            self[s] += coef
        return self
        
    def eliminate_zeros(self, tol=1e-10):
        for s, coef in list(self.items()):
            if abs(coef) <= tol:
                del self[s]

class FermionicEncoding:
    """
    This is the ultimate guide to fermionic encoding problems.
    We describe everything in the binary matrix format. In our paper,
    we also discover an extremely efficient way to improve on the known
    Hamiltonian encoding techniques.
    """

    def __init__(self, q: int, B: BinaryMatrix = None):
        self.cache = dict()
        self.q = q

        zeros = np.zeros((q, q), dtype=TYPE_BINARY)
        eye = np.eye(q, dtype=TYPE_BINARY)

        self.lower = np.block([[zeros, zeros], [eye, zeros]])
        self.sym = np.block([[zeros, eye], [eye, zeros]])

        # The B matrix is the generator matrix which transform the
        # fermionic occupation number basis to the qubit basis.
        if B is not None:
            self.B = B
        else:
            self.B = eye

        self.modes = list(range(q))
        
    @property
    def B(self) -> BinaryMatrix:
        return self._B
    
    @B.setter
    def B(self, value: BinaryMatrix):
        self._B = value
        self.B_inv = inv(value)
        self.update_cache()
        
    def update_cache(self):
        """
        Update the cache for the integer symplectic representation of the Pauli strings.
        """
        self.cache['xz'] = np.zeros(self.q, dtype=int)
        self.cache['xzp'] = np.zeros(self.q, dtype=int)

        for i in range(self.q):
            #x_jw = np.zeros(self.q, dtype=TYPE_BINARY)
            #x_jw[i] = 1
            x = self.B[:, i].copy()

            z_jw = np.zeros(self.q, dtype=TYPE_BINARY)
            z_jw[:i] = 1
            z = z_jw @ self.B_inv % 2

            #zp_jw = np.zeros(self.q, dtype=TYPE_BINARY)
            zp_jw = z_jw.copy()
            zp_jw[i:i+1] = 1
            zp = zp_jw @ self.B_inv % 2
            
            self.cache['xz'][i] = self.toint(np.concatenate([x, z]))
            self.cache['xzp'][i] = self.toint(np.concatenate([x, zp]))

    def toint(self, s: np.ndarray[TYPE_BINARY]) -> int:
        """convert your binary binary vector to the integer representation

        Arg:
        ----
        s (array): your symplectic representation, i.e. a binary vector

        Return:
        -------
        num (int): the corresponding basis
        """
        number = 0
        for bit in s[::-1]:
            number = (number << 1) | bit
        return number

    def to2(self, num):
        """Convert an integer back to symplectic representation,
        i.e. binary vector of length 2 * self.q

        Arg:
        ----
        num (int): a positive integer that is less than 2**(2*self.q)

        Return:
        -------
        s (array): an array of 0s and 1s of length 2 * self.q
        """
        # it is the conversion that has some problem in it.

        if num >= 2 ** (2 * self.q):
            raise ValueError(
                f"The number {num} is too large for the given qubit number ({self.q})."
            )

        s = []
        while num:
            s.append(num % 2)
            num = num // 2

        additional_zeros = [0] * (2 * self.q - len(s))
        return np.array(s + additional_zeros, dtype=TYPE_BINARY)

    def symprod(self, a, b):
        """define the symplectic project between Pauli strings
        Arg:
        ----
        a: a tuple of (symplectic vector of length 2*self.q, sign).
        b: same as a

        return:
        -------
        a tuple of (symplectic vector of length 2*self.q, sign)
        """
        a_vec, a_sign = a
        b_vec, b_sign = b

        vec = (a_vec + b_vec) % 2
        sign = (-1) ** (a_vec @ self.lower @ b_vec) * a_sign * b_sign
        return (vec, sign)

    def fermionic_op(self, operators: list[tuple[int, bool]]) -> PauliStringSum:
        """
        Encode a product of fermionic operators into a linear 
        combination of Pauli strings.

        Arg:
        ----
        a (list of tuple): a list of fermionic creation or annihilation
            operators, each specified by a tuple of (location, is_daggered).
            The location is an integer in the range of [0, self.q-1], and 
            is_daggered is a boolean value. 

        Return:
        -------
        op (PauliStringSum): the encoded a linear combination of Pauli strings 
        """
        encoded = PauliStringSum(self.q, float)

        for s, m in operators:
            op = PauliStringSum(self.q, float)
            op[self.cache['xz'][s]] = 0.5
            op[self.cache['xzp'][s]] += -0.5 if m else 0.5
            encoded *= op

        return encoded

    def set_generator(self, name: str):
        """
        Set the generator of the encoding.
        Args:
            name (str): name of the generator.
        """
        if name == "BK":
            self.BK_mat()
        elif name == "JW":
            self.eye()
        elif name == "parity":
            self.parity()
        elif name == "random":
            self.rand_mat()
        else:
            raise ValueError(f"Unknown generator {name}.")
        return self

    def parity(self):
        """
        Set the generator to be the parity encoding matrix.
        """
        self.B = np.triu(np.ones((self.q, self.q), dtype=TYPE_BINARY))
        return self

    def eye(self):
        """
        Set the generator to be the identity matrix, i.e. the Jordan-Wigner matrix.

        return:
        -------
        B: the matrix
        """
        self.B = np.eye(self.q, dtype=TYPE_BINARY)
        return self

    def BK_mat(self):
        """
        Set the generator to be the Bravyi-Kitaev matrix.
        """
        reps = int(np.log2(self.q) + 1)
        B = np.array([[1]])
        for i in range(reps):
            ze = np.zeros(B.shape)
            B1 = np.concatenate((ze, B), axis=1)
            for i in range(B.shape[0]):
                ze[0][i] += 1
            B2 = np.concatenate((B, ze), axis=1)
            B = np.concatenate((B2, B1))

        B = B[2**reps - self.q :, 2**reps - self.q :]
        self.B = round_to_binary(B)
        return self

    def rand_mat(self):
        """
        Set the generator to be a random binary matrix of size (q, q)
        and rank q.
        """
        rk = 0
        while rk < self.q:
            B = np.random.randint(0, 2, size=(self.q, self.q), dtype=TYPE_BINARY)
            rk = rank(B)

        self.B = B
        return self

    def flip(self, sites):
        """
        flip some number of the entires of the encoding matrix `self.B`
        within the upper triangle region.
        This design is specifically for the BK, parity, and JW matrix.

        Arg:
        ----
        sites (int): This is the number of sites which you want to flip.


        Return:
        -------
        C, C_inv: C and its inverse
        """
        if sites == 0:
            return self.B
        C = self.B.copy()
        C_inv = self.B_inv.copy()

        "Just generate changes for upper triangles"
        np.random.shuffle(self.modes)

        B = self.B.copy()

        for i in range(sites):
            # sort the indices so that the flips are always in the upper triangle
            inds = np.sort(self.modes[2 * i : 2 * i + 2])
            # flip the Boolean entry at (inds[0], inds[1])
            B[inds[0], inds[1]] = (B[inds[0], inds[1]] + 1) % 2

        self.B = B
        self.B_inv = np.round(np.linalg.inv(B) * np.linalg.det(B)).astype(int) % 2

        return C, C_inv

    def decoder(self, op):
        """
        Decode a sparse matrix representing a linear combination of Pauli strings
        to their symplectic representation of binary vectors and their coefficients in
        the linear combination.

        Arg:
        ----
        op (defaultdict): A dictionary where the keys are the integer representation of
            the Pauli strings, and the values are the coefficients of the corresponding
            Pauli strings in the linear combination.

        Return:
        -------
        A list of tuple of (symplectic vector, coefficient) of the Pauli strings,
        in the linear combination of Pauli strings.
        """
        return [(self.to2(p), c) for p, c in op.items()]

    def hamiltonian(self, one: np.ndarray, two: np.ndarray, threshold=1e-10) -> PauliStringSum:
        """
        Generate the Hamiltonian from the one and two body integrals.

        Arg:
        ----
        one (array): the first integrals, of shape (n_modes, n_modes)
        two (array): the second integral, of shape (n_modes, n_modes, n_modes, n_modes)
        threshold (float): the threshold under which small entries are ignored

        Return:
        -------
        ham (PauliStringSum): the Hamiltonian as a linear combination of Pauli strings

        """
        # extract the indices of the above threshold entries
        n_modes = one.shape[0]

        ham = PauliStringSum(self.q, float)

        dag_one = (1, 0)
        for i, j in product(range(n_modes), repeat=2):
            if abs(one[i, j]) < threshold:
                continue
            fermi_string = self.fermionic_op(zip((i, j), dag_one))
            fermi_string *= one[i, j]
            ham += fermi_string

        dag_two = (1, 1, 0, 0)
        for i, j, k, l in product(range(n_modes), repeat=4):
            if abs(two[i, j, k, l]) < threshold:
                continue
            fermi_string = self.fermionic_op(zip((i, j, k, l), dag_two))
            fermi_string *= two[i, j, k, l] / 2
            ham += fermi_string

        ham.eliminate_zeros(threshold)
        return ham

    def qiskit_rep(self, op):
        """Convert Encoding to qiskit format

        Arg:
        ----
        op (PauliStringSum): a linear combination of Pauli strings

        Return:
        -------
        ham (): qiskit representation of the linear combination of Pauli strings
        """
        from qiskit.opflow import I, X, Y, Z
        pauli_qiskit = {(0, 0): I, (0, 1): Z, (1, 0): X, (1, 1): -1j*Y}

        op_qiskit = 0
        for s_int, coef in op.items():
            s = self.to2(s_int)
            xs, zs = s[: self.q], s[self.q :]

            s_qiskit = 1
            for x, z in zip(xs, zs):
                s_qiskit ^= pauli_qiskit[(x, z)]

            op_qiskit += coef * s_qiskit
        return op_qiskit