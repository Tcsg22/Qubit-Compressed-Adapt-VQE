import copy
from itertools import product, permutations
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.linalg import expm
#import fqe
import pyscf
from openfermion import MolecularData
import openfermionpyscf as ofpyscf


import openfermion as of

#this code is written for particle conserved diagonalisation.
#In the future, perhaps we can make use of 
r"""NOTE ON PQRS CONVENTION:
  The data structures which hold fermionic operators / integrals /
  coefficients assume a particular convention which depends on how integrals
  are labeled:
  h[p,q]=\int \phi_p(x)* (T + V_{ext}) \phi_q(x) dx
  h[p,q,r,s]=\int \phi_p(x)* \phi_q(y)* V_{elec-elec} \phi_r(y) \phi_s(x) dxdy
  With this convention, the molecular Hamiltonian becomes
  H =\sum_{p,q} h[p,q] a_p^\dagger a_q
    + 0.5 * \sum_{p,q,r,s} h[p,q,r,s] a_p^\dagger a_q^\dagger a_r a_s
"""
'base functionalities'

def permute(n, zeros, ones, current="", permutations=[]):
    
    if len(current) == n:
        permutations.append([int(x) for x in current])
        #return permutations
    
    if zeros > 0:
        permute(n, zeros - 1, ones, current + "0", permutations)
    if ones > 0:
        permute(n, zeros, ones - 1, current + "1", permutations)

    return np.array(permutations)

def meanfield(mols):
    '''
    Generate all mean field energies
    '''
    MF = []
    for mol in mols:
        
        mf = mol.RHF()
        mf.kernel()
        MF.append(mf)
    return MF

def new_hot(mol, freeze_core, orb_choice, hardcore = 0):
    '''new_hot algorithm 
    '''
    spinorb = len(orb_choice)*2

    e = (mol.nelectron-2*freeze_core)//2**((hardcore)%2)
    basis = permute(spinorb, spinorb-e, e, "", []) 
    return basis


def onehot(mol, freeze, hardcore):
    '''basis generator
    generate a set of number conserving basis
    
    mol: the molecular information
    
    freeze: frozen core
    
    return:
    
    base: the quantum basis
    '''
    e = mol.nelectron/2-freeze
    orb = mol.nao-freeze
    basis = list(set(list(permutations(list(np.ones(int(e), dtype = int))+list(np.zeros(int(orb-e), dtype = int))))))
    base = []

    if hardcore == False:
        for i in basis:
            pair =  np.array([list(i), list(i)]).T
            pair = pair.reshape(2*len(list(i)))
            base.append(pair)
        return base

    else:
        return basis
        

def compute_integrals(pyscf_scf):
    """Integrals with diagonalisation
    Args:
        pyscf_molecule: A pyscf molecule instance.
        pyscf_scf: A PySCF "SCF" calculation object.
    Returns:
        one_electron_integrals: An N by N array storing h_{pq}
        two_electron_integrals: An N by N by N by N array storing h_{pqrs}.
    """
    
    from pyscf import gto, scf, ao2mo, fci, ci

    # Get one electrons integrals.
    pyscf_molecule = pyscf_scf.mol
    n_orbitals = pyscf_scf.mo_coeff.shape[1]
    one_electron_compressed = reduce(np.dot, (pyscf_scf.mo_coeff.T,
                                                 pyscf_scf.get_hcore(),
                                                 pyscf_scf.mo_coeff))
    one_electron_integrals = one_electron_compressed.reshape(
        n_orbitals, n_orbitals).astype(float)

    # Get two electron integrals in compressed format.
    # That is indeed what I worried.
    #try this
    second = pyscf_molecule.intor('int2e')
    two_electron_compressed = ao2mo.kernel(second,
                                           pyscf_scf.mo_coeff)

    two_electron_integrals = ao2mo.restore(
        1, # no permutation symmetry
        two_electron_compressed, n_orbitals)
    # See PQRS convention in OpenFermion.hamiltonians._molecular_data
    # h[p,q,r,s] = (ps|qr)
    two_electron_integrals = np.asarray(
        two_electron_integrals.transpose(0, 2, 3, 1), order='C')
    return one_electron_integrals, two_electron_integrals

#now you have gotten this beautiful result.
def raw_integrals(pyscf_scf):
    """integrals without diagonalisation
    Args:
        pyscf_molecule: A pyscf molecule instance.
        pyscf_scf: A PySCF "SCF" calculation object.
    Returns:
        one_electron_integrals: An N by N array storing h_{pq}
        two_electron_integrals: An N by N by N by N array storing h_{pqrs}.
    """
    
    from pyscf import gto, scf, ao2mo, fci, ci
    from functools import reduce
    
    pyscf_molecule = pyscf_scf.mol
    # Get one electrons integrals.
    n_orbitals = pyscf_scf.mo_coeff.shape[1]
    one_electron_compressed = pyscf_scf.get_hcore()
    print(np.round(reduce(np.dot, (pyscf_scf.mo_coeff.T,pyscf_scf.mo_coeff)), 3))
    #one_electron_compressed = reduce(np.dot, (pyscf_scf.mo_coeff.T,
                                                 #pyscf_scf.get_hcore(),
                                                 #pyscf_scf.mo_coeff))
    print(pyscf_scf.get_hcore().shape)
    one_electron_integrals = one_electron_compressed.reshape(
        n_orbitals, n_orbitals).astype(float)

    # Get two electron integrals in compressed format.
    
    two_electron_exact = pyscf_molecule.intor('int2e')
    #two_electron_long = ao2mo.kernel(integral,pyscf_scf.mo_coeff)
    

    #then full integral:
    #two_electron_exact = ao2mo.kernel(pyscf_molecule,pyscf_scf.mo_coeff)
    
    two_electron_exact = ao2mo.restore(
        1, # no permutation symmetry
        two_electron_exact, n_orbitals)
    
    
    # See PQRS convention in OpenFermion.hamiltonians._molecular_data
    # h[p,q,r,s] = (ps|qr)
    two_electron_exact = np.asarray(
        two_electron_exact.transpose(0, 2, 3, 1), order='C')
    return one_electron_integrals, two_electron_exact

def RDM1_compressed(h):
    h_new = np.zeros((h.shape[0]//2, h.shape[1]//2))
    print(h.shape[0]/2)
    for i in range(h.shape[0]//2):
        h_new[i][i] = h[2*i][2*i]+h[2*i+1][2*i+1]
    
    return np.round(h_new, 10)

def spinorb_from_spatial(one_body_integrals, two_body_integrals):
    from openfermion.config import EQ_TOLERANCE
    n_qubits = 2 * one_body_integrals.shape[0]

    # Initialize Hamiltonian coefficients.
    one_body_coefficients = np.zeros((n_qubits, n_qubits))
    two_body_coefficients = np.zeros(
        (n_qubits, n_qubits, n_qubits, n_qubits))
    # Loop through integrals.
    for p in range(n_qubits // 2):
        for q in range(n_qubits // 2):

            # Populate 1-body coefficients. Require p and q have same spin.
            one_body_coefficients[2 * p, 2 * q] = one_body_integrals[p, q]
            one_body_coefficients[2 * p + 1, 2 * q +
                                  1] = one_body_integrals[p, q]
            # Continue looping to prepare 2-body coefficients.
            for r in range(n_qubits // 2):
                for s in range(n_qubits // 2):

                    # Mixed spin
                    two_body_coefficients[2 * p, 2 * q + 1, 2 * r + 1, 2 *
                                          s] = (two_body_integrals[p, q, r, s])
                    two_body_coefficients[2 * p + 1, 2 * q, 2 * r, 2 * s +
                                          1] = (two_body_integrals[p, q, r, s])

                    # Same spin
                    two_body_coefficients[2 * p, 2 * q, 2 * r, 2 *
                                          s] = (two_body_integrals[p, q, r, s])
                    two_body_coefficients[2 * p + 1, 2 * q + 1, 2 * r +
                                          1, 2 * s +
                                          1] = (two_body_integrals[p, q, r, s])

    # Truncate.
    one_body_coefficients[
        np.absolute(one_body_coefficients) < EQ_TOLERANCE] = 0.
    two_body_coefficients[
        np.absolute(two_body_coefficients) < EQ_TOLERANCE] = 0.

    return one_body_coefficients, two_body_coefficients

#first generate a set of molecules, so for example you can first define your molecular function.
#So lets write a lower level function. 
        


def _orthogonal_matrix(plane, theta):
    '''generator of the 3D orthogonal matrix
    plane: list of axis, specifying the plane of rotation, orientation matters.
    
    theta: specifying the angle of rotation
    '''
    mat = np.zeros((3, 3))
    for (i,j) in product(plane, repeat = 2):
        mat[i, j] = np.cos(theta+(j-i)*np.pi/2)
    
    mat[3-plane[0]-plane[1], 3-plane[0]-plane[1]] = 1
    
    return mat


def plane_molecule(names, Rs, thetas, basis, mult):
    '''data of the plane molecule, O at the origin
    names: name of atoms
    
    Rs: list of radius relative to O
    
    thetas: list of angles relative to O
    '''
    mol_data = np.array([names, Rs, thetas]).T
    molecule = ''
    for atom in mol_data:
        coord = float(atom[1])*np.dot(_orthogonal_matrix([1, 2], float(atom[2])), np.array([0,0,1]))
        molecule+=atom[0]+' '+str(np.round(coord[0], 6))+' '+str(np.round(coord[1], 6))+' '+str(np.round(coord[2], 6))+' ; '
    molecule=molecule[:-2]
    mol = pyscf.M(atom = molecule, 
                  basis = basis, 
                  spin = (mult-1)/2,
                  symmetry = True)
    return mol

'I basically need to modify this first quantised functionality.'
def first_quantised_hardcore(mol, one, two, base):
    '''matrix with particles conserved basis
    Use Slater rule. However, dense matrix for now.
    '''    
    mat = np.zeros((len(base), len(base)))
    for i, j in product(list(zip(np.arange(len(base)), base)), repeat = 2):
        #two const.
        h = 0
        g = 0

        #for RHF, the electrons come in pair.
        #surely if you hardcore this is much better. Well, turn out that we will need extra procedure. 
        diff = i[1]-j[1]
        if sum(np.abs(diff)) == 0:
            h=np.trace(one*i[1])
            #pick basis sum. Question: is this code optimal at all?
            orb_vec = [index for index, item in enumerate(i[1]) if item == 1]
            for (m,n) in product(orb_vec, repeat = 2):
                g+=+1/2*two[m,n,n,m]-1/2*two[m,n,m,n]
            mat[i[0], j[0]] = h+g
        #omitted due to RHF
        #apparently you cannot omit that.
        #there is a lot going on but first you should fix this matrix
        elif sum(diff) == 2:
            end = [index for index, item in enumerate(diff) if item == 1]
            start = [index for index, item in enumerate(diff) if item == -1]
            h = one[end[0], start[0]]
            common = i[1]*j[1]
            orb_vec = [index for index, item in enumerate(common) if item == 1]
            for n in orb_vec:
                #hkχiℓχi − hkχiχiℓ
                g+=two[end[0], n, n, start[0]]-two[end[0], n, start[0], n]
            #meanwhile we sum over the same orbitals
            mat[i[0], j[0]] = h+g
            
            #print(diff)
        #define minus as the taken away. 
        elif sum(np.abs(diff)) == 4:
            #only pair transition is considered here, because RHF conserves spin pair.
            end = [index for index, item in enumerate(diff) if item == 1]
            start = [index for index, item in enumerate(diff) if item == -1]
            mat[i[0], j[0]] = +two[end[1], end[0], start[0], start[1]]-two[end[1], end[0], start[1], start[0]]
    
    #after you get this what do you do?
    mat = sparse.csr_matrix(mat)
    return mat

def first_quantised(mol, one, two, base):
    '''matrix with particles conserved basis
    Use Slater rule. However, dense matrix for now.
    Need to put extra constraint.
    '''    
    mat = np.zeros((len(base), len(base)))

    stuff = list(zip(np.arange(len(base)), base))
    #for i, j in product(list(zip(np.arange(len(base)), base)), repeat = 2):
    for i in stuff:
        for j in stuff[i[0]:]:
            #two const.
            h = 0
            g = 0
            'To ensure block diagonalisation.'
            if sum(i[1])==sum(j[1]):
                diff = i[1]-j[1]
                'first terms are all correct.'
                if sum(np.abs(diff)) == 0:
                    h=np.trace(one*i[1])
                    #pick basis sum. Question: is this code optimal at all?
                    orb_vec = [index for index, item in enumerate(i[1]) if item == 1]
        
                    'remember that the actually integral symmetry is not obeyed.'
                    for (m,n) in product(orb_vec, repeat = 2):
                        g+=1/2*(two[n,m,m,n]-two[n,m,n,m])
                    mat[i[0], j[0]] = h+g
        
                elif sum(np.abs(diff)) == 2:
                    o = [i for i in range(len(diff)) if diff[i]!=0]
                    sop = np.zeros(len(diff))
                    modes = np.arange(len(diff))
            
                    for index, gg in enumerate(o):
                        p = gg*np.ones(len(diff))
                        sop+=(modes<p)
                    sop = sop%2
        
                    'define anti-sym op.'
                    end = [index for index, item in enumerate(diff) if item == 1]
                    start = [index for index, item in enumerate(diff) if item == -1]
                    h = one[end[0], start[0]]
                    common = i[1]*j[1]
                    sign = (sop@common)%2
                    orb_vec = [index for index, item in enumerate(common) if item == 1]
                    for n in orb_vec:
                        g+=1/2*(two[end[0], n, n, start[0]]+two[n, end[0], start[0], n]-two[end[0], n, start[0], n]-two[n, end[0], n, start[0]])
                    #meanwhile we sum over the same orbitals
                    mat[i[0], j[0]] = (h+g)*(-1)**(sign)
                    mat[j[0], i[0]] = mat[i[0], j[0]].conj()
        
                elif sum(np.abs(diff)) == 4:
                    
                    o = [i for i in range(len(diff)) if diff[i]!=0]
                    sop = np.zeros(len(diff))
                    modes = np.arange(len(diff))
            
                    for index, gg in enumerate(o):
                        p = gg*np.ones(len(diff))
                        sop+=(modes<p)
                    sop = sop%2
                    'then calculate sign.'
                    common = i[1]*j[1]
                    sign = (sop@common)%2
        
                    #only pair transition is considered here, because RHF conserves spin pair.
                    end = [index for index, item in enumerate(diff) if item == 1]
                    start = [index for index, item in enumerate(diff) if item == -1]
                    mat[i[0], j[0]] = +1/2*(+two[end[1], end[0], start[0], start[1]]+two[end[0], end[1], start[1], start[0]]
                                            -two[end[1], end[0], start[1], start[0]]-two[end[0], end[1], start[0], start[1]])*(-1)**(sign)
                    mat[j[0], i[0]] = mat[i[0], j[0]].conj()
    
    #after you get this what do you do?
    mat = sparse.csr_matrix(mat)
    return mat


#there is a way to improve this first quantised method.

def get_active_space_integrals(one_body_integrals,
                               two_body_integrals,
                               occupied_indices=None,
                               active_indices=None):
    """Restricts a molecule at a spatial orbital level to an active space
    This active space may be defined by a list of active indices and
        doubly occupied indices. Note that one_body_integrals and
        two_body_integrals must be defined
        n an orthonormal basis set.
    Args:
        one_body_integrals: One-body integrals of the target Hamiltonian
        two_body_integrals: Two-body integrals of the target Hamiltonian
        occupied_indices: A list of spatial orbital indices
            indicating which orbitals should be considered doubly occupied.
        active_indices: A list of spatial orbital indices indicating
            which orbitals should be considered active.
    Returns:
        tuple: Tuple with the following entries:
        **core_constant**: Adjustment to constant shift in Hamiltonian
        from integrating out core orbitals
        **one_body_integrals_new**: one-electron integrals over active
        space.
        **two_body_integrals_new**: two-electron integrals over active
        space.
    """
    # Fix data type for a few edge cases
    occupied_indices = [] if occupied_indices is None else occupied_indices
    if (len(active_indices) < 1):
        raise ValueError('Some active indices required for reduction.')

    # Determine core constant
    core_constant = 0.0
    for i in occupied_indices:
        core_constant += 2 * one_body_integrals[i, i]
        for j in occupied_indices:
            core_constant += (2 * two_body_integrals[i, j, j, i] -
                              two_body_integrals[i, j, i, j])

    # Modified one electron integrals
    one_body_integrals_new = np.copy(one_body_integrals)
    for u in active_indices:
        for v in active_indices:
            for i in occupied_indices:
                one_body_integrals_new[u, v] += (
                    2 * two_body_integrals[i, u, v, i] -
                    two_body_integrals[i, u, i, v])

    # Restrict integral ranges and change M appropriately
    return (core_constant,
            one_body_integrals_new[np.ix_(active_indices, active_indices)],
            two_body_integrals[np.ix_(active_indices, active_indices,active_indices, active_indices)])

def RDM1(vec, base):
    '''
    To calculate the 1-RDM, again use Slater rule. This is diagonal in nature?
    This might change due to the exchange correlation matrix, so safe to code the general case.
    It seems like the 1-RDM must be diagonal in the RHF case. 
    Code the simple case first.
    vec: the ground state
    basis: the basis

    '''
    #in fact, the first RDM is diagonal, so we will not need this.
    
    h = np.zeros((len(base[0]), len(base[0])))
    h_diag = np.identity(len(base[0]))
    
    #there is nothing wrong with this!!
    for (coe, config) in zip(vec, base):
        h+=h_diag*config*coe.conj()*coe

    for (i,j) in product(list(zip(vec, base)), repeat = 2):
        diff = i[1]-j[1]
        if sum(np.abs(diff)) == 2:

            'The fermionic sign'
            o = [i for i in range(len(diff)) if diff[i]!=0]
            sop = np.zeros(len(diff))
            modes = np.arange(len(diff))
    
            for index, gg in enumerate(o):
                p = gg*np.ones(len(diff))
                sop+=(modes<p)
            sop = sop%2

            common = i[1]*j[1]
            sign = (sop@common)%2

            end = [index for index, item in enumerate(diff) if item == 1]
            start = [index for index, item in enumerate(diff) if item == -1]
            h[end[0], start[0]]+= +i[0].conj()*j[0]*(-1)**(sign)
            
    
    return h

#you need to know how to transform them together. Then, verify the energy. 
#an important step. Somehow this 
#Or, one can obtain the 2-RDM via the JW transform. This result might not be good enough. 
def RDM2(vec, base):
    '''
    To calculate the 2-RDM. The problem is very likely due to numerical error. 
    '''
    
    g = np.zeros((len(base[0]), len(base[0]), len(base[0]), len(base[0])))
    #theoretically should work. This is really hard, and that is why I would like to focus. 
    g_diag = np.identity(len(base[0]))
    '''
    for i in list(zip(vec, base)):
        mat = g_diag*i[1]*i[0].conj()*i[0]
        ten = np.tensordot(mat, mat, axes = 0)
        g += np.transpose(ten, (0,2,3,1))-np.transpose(ten, (2,0,3,1))
    '''
    
    for (i,j) in product(list(zip(vec, base)), repeat = 2):
        diff = i[1]-j[1]
        #here we might have the sign problem. maybe this is the error. Use old trad method.
        if sum(np.abs(diff)) == 0:
            orb_vec = product([index for index, item in enumerate(i[1]) if item == 1], repeat = 2)
            orb_vec = [i for i in orb_vec if i[0]<i[1]]
            for (m, n) in orb_vec:
                g[m, n, n, m] += i[0].conj()*i[0]
                g[n, m, m, n] += i[0].conj()*i[0]
                g[m, n, m, n] += -i[0].conj()*i[0]
                g[n, m, n, m] += -i[0].conj()*i[0]

        elif sum(np.abs(diff)) == 2:
            o = [i for i in range(len(diff)) if diff[i]!=0]
            sop = np.zeros(len(diff))
            modes = np.arange(len(diff))
    
            for index, gg in enumerate(o):
                p = gg*np.ones(len(diff))
                sop+=(modes<p)
            sop = sop%2

            'define anti-sym op.'
            end = [index for index, item in enumerate(diff) if item == 1]
            start = [index for index, item in enumerate(diff) if item == -1]
            common = i[1]*j[1]
            sign = (sop@common)%2
            orb_vec = [index for index, item in enumerate(common) if item == 1]
            
            for n in orb_vec:
                g[end[0], n, n, start[0]]+=i[0].conj()*j[0]*(-1)**(sign)
                g[n, end[0], start[0], n]+=i[0].conj()*j[0]*(-1)**(sign)
                g[end[0], n, start[0], n]+=-i[0].conj()*j[0]*(-1)**(sign)
                g[n, end[0], n, start[0]]+=-i[0].conj()*j[0]*(-1)**(sign)
                
            
                 
        elif sum(np.abs(diff)) == 4:            
            end = [index for index, item in enumerate(diff) if item == 1]
            start = [index for index, item in enumerate(diff) if item == -1]

            o = [i for i in range(len(diff)) if diff[i]!=0]
            sop = np.zeros(len(diff))
            modes = np.arange(len(diff))
    
            for index, gg in enumerate(o):
                p = gg*np.ones(len(diff))
                sop+=(modes<p)
            sop = sop%2
            'then calculate sign.'
            common = i[1]*j[1]
            sign = (sop@common)%2

            g[end[1], end[0], start[0], start[1]] += +i[0].conj()*j[0]*(-1)**(sign)
            g[end[1], end[0], start[1], start[0]] += -i[0].conj()*j[0]*(-1)**(sign)
            g[end[0], end[1], start[0], start[1]] += -i[0].conj()*j[0]*(-1)**(sign)
            g[end[0], end[1], start[1], start[0]] += +i[0].conj()*j[0]*(-1)**(sign)
    return g

'all spectrum requires modifications.'
def spectrum(mols, functions, freeze_core, orb_choice, active = 0):
    '''Energy spectrum of a molecule
    
    mols: a list of molecules
    
    freeze: mean field approximation at the core
    
    functions: list of functions
    
    return: All energies from your function. With the law row HF energy
    '''
    E = []

    
    GS = []
    for i in range(len(functions)):
        GS.append([])

    MF = []
    for mol in mols:
        
        mf = mol.RHF()
        mf.kernel()
        MF.append(mf)
        naos = np.arange(mol.nao)
        
        energies = []
        gs = []

        'energy with same parameters for different functions for different approximation.'
        for function in functions:
            core = 0
            one, two = function[0](mf)
            
            #active space transformation.
            #print(type(function[1]))
            if callable(function[1]):
                #print('old: ', np.round(one, 3))
                'Freeze not only core but also virtual orbitals for active space.'
                (core, one, two) = get_active_space_integrals(one, two, occupied_indices = list(naos[:freeze_core]), #+list(naos[len(naos)-freeze_vir:]),
                                                              active_indices = orb_choice)
                #print('new: ', np.round(one, 3))
            #to compute the integrals, to save space.
            one, two = spinorb_from_spatial(one,two)
            mat = first_quantised(mf.mol, one, two, function[2])
            evals, evecs = sparse.linalg.eigsh(mat.real, k=2, which = 'SA')
            energies.append(evals[0]+mf.energy_nuc()+core)
            #print(evecs.shape)
            gs.append(evecs.T[0])
            

        'also the RHF energy later.'
        energies.append(mf.e_tot)
        E.append(energies)
        
        for i in range(len(gs)):
            GS[i].append(gs[i])

    #print(GS)
    return np.array(E).T, GS, MF


def integrals(mols, functions, orb_choice):
    '''Compute first and second electronic integrals
    '''
    integrals = []
    MF = []
    for mol in mols:
        
        mf = mol.RHF()
        mf.kernel()
        MF.append(mf)
        naos = np.arange(mol.nao)
        
        'energy with same parameters for different functions for different approximation.'
        for function in functions:
            core = 0
            one, two = function[0](mf)
            
            #active space transformation.
            #print(type(function[1]))
            if callable(function[1]):
                #print('old: ', np.round(one, 3))
                'Freeze not only core but also virtual orbitals for active space.'
                (core, one, two) = get_active_space_integrals(one, two, occupied_indices = list(naos[:orb_choice[0]]), #+list(naos[len(naos)-freeze_vir:]),
                                                              active_indices = orb_choice)
                #print('new: ', np.round(one, 3))
            #to compute the integrals, to save space.
            one, two = spinorb_from_spatial(one,two)
        integrals.append((one, two, core))
        
    return integrals, MF