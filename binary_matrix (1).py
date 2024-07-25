import numpy as np

TYPE_BINARY = np.ubyte
BinaryMatrix = np.ndarray[TYPE_BINARY, TYPE_BINARY]

def rank(matrix: BinaryMatrix) -> int:
    """
    Compute the rank of a binary matrix in Z2.
    Note that the Z2 rank is always less than or equal to the real rank.
    """
    matrix = matrix.copy()
    m, n = matrix.shape
    i = 0
    for k in range(n):
        # Look for non-zero entry in col k at or below row i.
        nonzero_rows = np.where(matrix[i:, k] != 0)[0]
        if nonzero_rows.size > 0:
            # Swap rows i and nonzero_rows[0] + i
            matrix[[i, nonzero_rows[0] + i]] = matrix[[nonzero_rows[0] + i, i]]
            # Zero out below pivot
            below = list(range(i + 1, m)) if m != i + 1 else []
            for j in below:
                if matrix[j, k] != 0:
                    matrix[j, :] ^= matrix[i, :]
            i += 1
        if i >= m:
            break
    # The rank is the number of non-zero rows.
    return np.count_nonzero((matrix != 0).any(axis=1))

def _left_inv(matrix: BinaryMatrix) -> BinaryMatrix:
    matrix = matrix.copy()
    m, n = matrix.shape

    # Initialize the inverse matrix as the identity matrix.
    inverse = np.eye(m, dtype=np.ubyte)
    
    # Perform Gaussian elimination.
    for i in range(n):
        if matrix[i, i] == 0:
            for j in range(i+1, m):
                if matrix[j, i] != 0:
                    matrix[[i, j]] = matrix[[j, i]]
                    inverse[[i, j]] = inverse[[j, i]]
                    break
            else:
                raise ValueError("Matrix does not have an inverse in Z2.")
        
        for j in range(i+1, m):
            if matrix[j, i] != 0:
                matrix[j, :] ^= matrix[i, :]
                inverse[j, :] ^= inverse[i, :]
    
    # Back substitution
    for i in range(n-1, -1, -1):
        for j in range(i):
            if matrix[j, i] != 0:
                matrix[j, :] ^= matrix[i, :]
                inverse[j, :] ^= inverse[i, :]

    # select only the first n rows
    return inverse[:n, :]

def _inv_numpy(matrix: np.ndarray) -> BinaryMatrix:
    """
    Compute the Z2 inverse of a binary matrix
    using numpy's linear algebra functions.
    Args:
        matrix (array): a matrix that is close to binary.
    """
    det = np.linalg.det(matrix)
    inverse = np.linalg.inv(matrix)
    adjugate = np.around(det * inverse).astype(TYPE_BINARY) 
    return adjugate % 2

def is_binary(matrix: np.ndarray) -> bool:
    """
    Check if a matrix is binary, i.e. all entries are in {0, 1}.
    Note that the comparison is done using exact equality.
    """
    return np.all((matrix == 0) | (matrix == 1))

def to_binary(matrix: np.ndarray) -> BinaryMatrix:
    """
    Convert a matrix to binary, i.e. convert all entries to integers.
    Args:
        matrix: A matrix of floats.
    Returns:
        A matrix in Z2, i.e. all entries are in {0, 1}.
    Raises:
        ValueError: If the matrix is not binary.
    """
    if not is_binary(matrix):
        raise ValueError("Matrix is not binary.")
    return matrix.astype(TYPE_BINARY)

def is_close_to_binary(matrix: np.ndarray, tol=1e-8) -> bool:
    """
    Check if a matrix is close to binary, i.e. all entries are in {0, 1} up to a tolerance.
    Args:
        matrix: A matrix of floats.
    """
    return np.all((np.abs(matrix) < tol) | (np.abs(matrix - 1) < tol))

def round_to_binary(matrix: np.ndarray, tol=1e-8) -> BinaryMatrix:
    """
    Round a matrix to binary, i.e. round all entries to the nearest integer.
    Args:
        matrix: A matrix of floats.
        tol (float): The tolerance for rounding to binary.
    Returns:
        A matrix rounded to binary, i.e. all entries are in {0, 1}.
    Raises:
        ValueError: If the matrix is not close to binary.
    """
    if not is_close_to_binary(matrix, tol):
        raise ValueError("Matrix is not close to binary.")
    return np.around(matrix).astype(TYPE_BINARY)

def left_inv(matrix: BinaryMatrix) -> BinaryMatrix:
    """
    Compute the left inverse of a binary matrix in Z2.
    The inverse exists if and only if rank(matrix) equals 
    the number of columns of the matrix.
    Args:
        matrix: A binary matrix in Z2.
        The matrix must have no less rows than columns.
    Returns:
        The left inverse of the matrix, such that
        left_inv(matrix) @ matrix % 2 = I
    Raises:
        ValueError: if the matrix is not binary.
        ValueError: if the matrix has less rows than columns.
        ValueError: if the matrix does not have an left inverse.
    """
    m, n = matrix.shape
    if not is_binary(matrix):
        raise ValueError("All entries of the matrix must be in Z2.")
    if not m >= n:
        raise ValueError("Matrix must have no less rows than columns.")
    return _left_inv(matrix)

def right_inv(matrix: BinaryMatrix) -> BinaryMatrix:
    """
    Compute the right inverse of a binary matrix in Z2.
    The inverse exists if and only if rank(matrix) equals
    the number of rows of the matrix.
    Args:
        matrix: A binary matrix in Z2. 
        The matrix must have no less columns than rows.
    Returns:
        The right inverse of the matrix, such that
        matrix @ right_inv(matrix) % 2 = I
    Raises:
        ValueError: if the matrix is not binary.
        ValueError: if the matrix has less columns than rows.
        ValueError: if the matrix does not have an right inverse.
    """
    m, n = matrix.shape
    if not is_binary(matrix):
        raise ValueError("All entries of the matrix must be in Z2.")
    if not m <= n:
        raise ValueError("Matrix must have no less columns than rows.")
    return _left_inv(matrix.T).T

def inv(matrix: BinaryMatrix) -> BinaryMatrix:
    """
    Compute the inverse of a binary square matrix in Z2.
    The inverse exists if and only if rank(matrix) equals
    the size of the square matrix.
    Args:
        matrix: A binary matrix in Z2. Must be square.
    Returns:
        The inverse of the matrix, such that
        matrix @ inv(matrix) % 2 = inv(matrix) @ matrix % 2 = I
    Raises:
        ValueError: if the matrix is not binary.
        ValueError: if the matrix is not square.
        ValueError: if the matrix does not have an inverse.
    """
    m, n = matrix.shape
    if not is_binary(matrix):
        raise ValueError("All entries of the matrix must be in Z2.")
    if not m == n:
        raise ValueError("Matrix must have equal number of rows and columns.")
    return _left_inv(matrix)
