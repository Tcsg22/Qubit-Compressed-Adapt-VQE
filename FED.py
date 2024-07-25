import numpy as np
import qiskit
from diag import *
from linear import *
from pre_FED import *

'----------------------------------------------------RDM_info----------------------------------------------------'


def Indices(integral, threshold = 10**-15):
    '''
    Compute the non-0 indices indiscriminatively

    Input:
    ------
    mf: your meanfield object

    naos: the atomic orbitals

    freeze: number of orbitals to freeze. (might require better data structure).

    threshold: threshold for accepting the TDM element. 

    Output:
    -------
    BASE: the non-0 dontribution indices
    '''
    from itertools import product

    one = integral[0]
    two = integral[1]
    
    one_base = []
    two_base = []
    for (i, j) in product(range(len(one)), repeat = 2):
        if np.abs(one[i, j])> threshold:
            one_base.append([i, j])


    for (i, j, k, l) in product(range(len(two)), repeat = 4):
        if i!=j and k!=l:
            if np.abs(two[i, j, k, l])> threshold:
                two_base.append([i, j, k, l])

    base = one_base+two_base
    return base


def RDM_grouper(indices):
    '''RDM_grouper: Input the RDM indices, and this function
    groups the RDM together. Convention is [1,1,0,0] for 
    the creation and annhiliation operators. This function still have some bugs.

    Input:
    ------
    indices: a list of indices

    Output:
    -------
    group: measurement group of rdm
    '''

    'max three for four distinctive indices'
    four = []
    two  = []
    one = []
    for i in indices:
        stuff = set(i)
        if  len(stuff) == 4:
            four.append([i,list(sorted(i))])
        elif (len(i) == 4 and len(stuff) == 2) or len(stuff) == 1:
            one.append(i)

        #elif stuff == 2 or stuff == 3:
        else:
            if len(stuff) == 3:
                seti = [m for m in stuff if i.count(m) == 1]
                seti = list(sorted(seti))
            else:
                seti = list(sorted(i))
            two.append([i, seti])

    'four'
    
    'copy could be useless.'
    groups = []
    
    'to debug'
    #print(two)
    #print(four)
    scan = four+two
    while len(scan) !=0:

        'the second element of the 1st four_s element.'
        dist = scan[0][1]

        'let debug'
            
        
        'if eliminate from behind, no need to relabel.'
        elim = scan[::-1]
        
        group = []
        for index, item in enumerate(scan):
            if item[1] == dist:
                group.append(item[0])
                elim.pop(len(scan)-1-index)        
        'update the scan after cutting'
        scan = elim[::-1]
        groups.append(group)    

    'one'
    groups.append(one)
        
    return groups

def measurement_group(groups):
    '''Calculate the experimental measurement group
    '''
    New = []
    for group in groups:
        new = []
        while len(group)!=0:
            new.append(group[0])
            #print('group: ', group)
            if len(group[0])==4:
                equi = equi4(group[0])
                #print('equi: ', equi)
            else:
                equi = equi2(group[0])
                
            elim = group[::-1]
            for index, i in enumerate(group):
                if i in equi:
                    elim.pop(len(group)-1-index)
            group = elim[::-1]
        New.append(new)
                    
    return New

def compute_group(base):
    '''Calculate the groups for RDM by eliminating the complex conjugates.
    '''
    computebase = []
    while len(base)!=0:
        new = base.copy()
        #print(base[0])
        computebase.append(base[0])

        b = base[0]
        bconj = base[0][::-1]

        if bconj in base:
            #print('hi')
            if b!= bconj:
                new.remove(bconj)
        new.remove(b)
            #print('hi')
        base = new
    return computebase


def permute(n, zeros, ones, current="", permutations=[]):
    
    if len(current) == n:
        permutations.append([int(x) for x in current])
        #return permutations
    
    if zeros > 0:
        permute(n, zeros - 1, ones, current + "0", permutations)
    if ones > 0:
        permute(n, zeros, ones - 1, current + "1", permutations)

    return np.array(permutations)


def look_up_parity(M, emax, e, A):
    '''
    Look up table decoders: exactly decode all encoder's bitstrings at the cost of 
    scalability.
    '''
    table = dict()

    per = permute(M, M-e, e, "", [])
    for j in per:
        s = (A@j)%2
        s = ''.join(np.array(s, dtype = str))
        j = ''.join(np.array(j, dtype = str))
        if s not in table.keys():
            table[s] = j
    
    for i in range(M+1):
        if i>emax:
            break
            
        per = permute(M, M-i, i, "", [])   

        for j in per:
            s = (A@j)%2
            s = ''.join(np.array(s, dtype = str))
            j = ''.join(np.array(j, dtype = str))
            if s not in table.keys():
                table[s] = j


    
    return table




'Part I'

def RDMs(MF, naos, freeze, function, threshold = 10**-2):
    '''Compute RDMs with non-0 contributions. While the function is kept,
    It is assumed here that approximation must be made.
    Real orbitals only.

    Input:
    ------
    mols: the set fo molecules whose measurement basis is to be computed.

    naos: the atomic orbitals

    freeze: number of orbitals to freeze. (might require better data structure).

    threshold: threshold for accepting the TDM element. 

    Output:
    -------
    BASE: the non-0 dontribution indices
    GROUP_INFO: grouping information of the 4 local x-operators only.
    '''

    ONE = []
    TWO = []
    for mf in MF:

        energies = []
        gs = []

        'energy with same parameters for different functions for different approximation.'
        core = 0
        one, two = function[0][0](mf)
            
            #active space transformation.
        if callable(function[0][1]):
            (core, one, two) = function[0][1](one, two, occupied_indices = naos[:freeze], active_indices = naos[freeze:])
            #to compute the integrals, to save space.
        one, two = spinorb_from_spatial(one,two)
        ONE.append(one)
        TWO.append(two)
            

    'simple exhaustion'
    Base = []
    GROUP_INFO = []
    for (one, two) in zip(ONE, TWO):
        one_base = []
        for j in range(len(one)):
            for i in range(j+1):
                if np.abs(one[i, j])>threshold:
                    'for debug'
                    #print(one[i, j], [i, j])
                    one_base.append([i, j])

        two_base = []
        group_info = []
        'set by hand method is the most efficient.'
        'can do grouping here.'
        for l in range(len(two)):
            for k in range(len(two)):
                for j in range(k+1):
                    if np.abs(two[l, j, k, l])>threshold:
                        if l!=j and k!=l and j!=k:
                            two_base.append([l, j, k, l])
                        if j == k:
                            if l<j:
                                two_base.append([l, j, k, l])
            
            for k in range(l):
                for j in range(k):
                    for i in range(j):
                        'three possibilities'
                        group4 = []
                        if np.abs(two[i, j, k, l])>threshold:
                            two_base.append([i, j, k, l])
                            group4.append([i, j, k, l])
                        if np.abs(two[i, k, j, l])>threshold:
                            two_base.append([i, k, j, l])
                            group4.append([i, k, j, l])
                        if np.abs(two[i, l, k, j])>threshold:
                            two_base.append([i, l, k, j])                            
                            group4.append([i, l, k, j])
                        if len(group4)!=0:
                            group_info.append(group4)
                        
        base = one_base+two_base
        Base.append(base)
        GROUP_INFO.append(group_info)
              
    return Base, GROUP_INFO

'this is a O(N^2) sort, and could possibly be more efficient.'
def RDM_grouper(indices):
    '''RDM_grouper: Input the RDM indices, and this function
    groups the RDM together. Convention is [1,1,0,0] for 
    the creation and annhiliation operators. This function still have some bugs.

    Input:
    ------
    indices: a list of indices

    Output:
    -------
    group: measurement group of rdm
    '''

    'max three for four distinctive indices'
    four = []
    two  = []
    one = []
    for i in indices:
        stuff = set(i)
        if  len(stuff) == 4:
            four.append([i,list(set(i))])
        elif len(i) == 4 and len(stuff) == 2 or len(stuff) == 1:
            one.append(i)

        #elif stuff == 2 or stuff == 3:
        else:
            if len(stuff) == 3:
                seti = [m for m in stuff if i.count(m) == 1]
            else:
                seti = list(set(i))
            two.append([i, seti])

    'four'
    
    'copy could be useless.'
    groups = []
    
    scan = four+two
    while len(scan) !=0:

        'the second element of the 1st four_s element.'
        dist = scan[0][1]
        'if eliminate from behind, no need to relabel.'
        
        elim = scan[::-1]
        group = []
        for index, item in enumerate(scan):
            if item[1] == dist:
                group.append(item[0])
                elim.pop(len(scan)-1-index)
        
        'update the scan after cutting'
        scan = elim[::-1]
        groups.append(group)    

    'one'
    groups.append(one)
        
    return groups

def encoder(orb, e, Q, hardcore = 0):
    '''Compute the compression matrix with the molecule information.

    Input:
    ------
    mol: molecule object.

    Q: the guessed qubit number.

    freeze: freezing core orbitals.

    RHF: restricted Hartree-Fock.

    Output:
    -------
    A: a (Q, M) matrix
    '''
    
    #sanity check

    #up, down = bounds(mol, RHF)
    #if Q>up or Q<=down:
    #    print(up, down)
    #    raise ValueError('Compression must fail within the Hamming-Gilbert bounds.')

    
    if hardcore:
        e = e//2
        orb = orb//2
        #for debug.
        print('information: ', orb, e)
        g = generator(orb, e, Q, 5000)
        A = g.modtest2()
    else:
        print('information: ', orb, e)
        g = generator(orb, e, Q, 5000)
        A = g.modtest2()
    
    return A


'it is better to just write the function individually.'
def xfunc(groups, G, hardcore = 0):
    '''xfunc: calculate the encoded xstrings for analysis, 
    based on the RDM which you want to measure.

    Input:
    ------
    groups: The measurement group.

    encoder: the encoding matrix

    RHF: specify the encoder type for spatial or spin orbitals
    
    Output:
    -------
    The corresponding CNOT gates. 
    
    '''    
    
    M = max(G.shape)*2**hardcore
    xstrings = []
    for i in groups:
        ustring = np.zeros(M)
            
        for ele in i[0]:
            ustring[ele]+=1
            
        if hardcore:
            ustring = np.reshape(ustring, (ustring.shape[0]//2, 2))
            ustring = ustring.T[0]

        xstring = (G@ustring)%2
        xstrings.append(xstring)

    'from xstrings, we could define Clifford transformation.'
    return xstrings

def hard_ustring(ustrings):
    '''
    Convert to hardcore bosonic format.
    Could be useless in the future once standardised
    '''
    newu=  []
    for i in ustrings:
        i = np.reshape(i, (i.shape[0]//2, 2))
        newu.append(i.T[0])

    return newu


def xfun(op):
    '''
    local evaluation of xstring

    '''
    ustring = np.zeros(M)
    for ele in i[0]:
        ustring[ele]+=1
    return ustring%2
    

def Clifford(groups):
    '''Calculate the CNOT gates for measurement basis of the unencoded bitstrings. 

    Input:
    ------
    groups: The group from grouping

    Output:
    -------
    gates: the quantum gates
    '''
    gates = []
    
    for group in groups:
        
        CNOT = []
        group = group[0]

        'eliminate excess element.'
        m = set(group)
        group = sorted([i for i in m if group.count(i) == 1])

        if len(group) == 0:
            continue
                
        for i in range(len(group)-1):
            
            pair = (group[i+1], group[i])
            CNOT.append(pair)
        H = (group[-1])
        
        gates.append((CNOT, H))

    return gates


'This part will be computationally expensive.'

def glob_sign(groups, q):
    '''
    Compute the global sign given your operator informations.

    Input:
    ------
    groups: grouping of your operators

    q: the number of qubits

    Output:
    -------
    Signs: Sign information per group element
    '''
    Signs = []
    for group in groups:
        sign = []
        for loc in group:
            
            f = FermionicEncoding(q)
            'generate the daggers'
            dag = [1]*(len(loc)//2)+[0]*(len(loc)//2)
            op_info = np.array([loc, dag]).T
            op = f.fermionic_op(op_info)
            sign.append(list(op.values())[0]/np.abs(list(op.values())[0]))
        Signs.append(sign)
    return Signs

def pzfunc(groups, G, N):
    '''pfunc: compute the pstrings wrt each qubit operator.
       zfunc: associate sign (-1)^b.c to each bitstrings.

    Input:
    ------
    groups: the grouped RDM stuff

    G: the encoder

    N: the number of electrons
    
    Output:
    -------
    
    '''

    M = max(G.shape[0], G.shape[1])
    Configs = []
    Signs = []
    'There is an acted part and an unacted part, which is not difficult to calculate.'
    'You must finish this code this week, and perhaps do some readings in the weekend.'
    for group in groups:
        'let consider the all distinct case first.'
        config = []
        signs = []
        'note: ind are the distinct transition in the fermionic algebra'
        'len(op): will tell you the order of the operator.'
        for op in group:

            ind = op.copy()
            for i in ind:
                if ind.count(i)!=1:
                    ind.remove(i)
            
            s = fix_hamming(M-len(ind), N-len(op)//2).T
    
            'A Permutation'
            'insertion will be much better and easier.'
            per = [j for j in range(M) if j not in ind] + list(ind)
    
            m = [list(np.ones(s.shape[1]))]*(len(ind)-len(op)//2)+ [list(np.zeros(s.shape[1]))]*(len(ind)-len(op)//2)
            n = [list(np.ones(s.shape[1]))]
            a = 0
            o = []
            for j in ind:
                if op.count(j) == 1:
                    s = list(s) + [m[a]]
                    a+=1
                    o.append(j)
                else:
                    s = list(s)+ n

            o = sorted(o)
            'finally permute according to the indices.'
            s = [s[per.index(i)] for i in range(M)]

            'Meanwhile, probably should include the conjugates for benchmarking.'
            xop = np.zeros(M)
            for j in op:
                xop[j] += 1

            sconj = []
            for i in np.transpose(s):
                sconj.append((xop+np.array(i))%2)

            'this explains why there is a /2 term even for the non-diagonal term.'
            s = np.array(list(np.array(s).T)+list(np.array(sconj))).T
            config.append(s)


            'generate the sign operator'
            modes = np.arange(M)
            sop = np.zeros(M)
            for index, j in enumerate(o):
                p = j*np.ones(M)
                #if index%2==0:
                #    sop+=(modes<=p)
                #else:
                sop+=(modes<p)
            #print(sop)
            sop = sop%2
            #print('zstring: ', sop)
            #sop = [(i<=o[0])^(i<o[1]) for i in range(M.shape[1])]
            m = (np.array(s).T@np.array(sop))%2
            signs.append(m)

            


        Configs.append(config)
        Signs.append(signs)
    return Configs, Signs

'Later on we need to write a decoder function.'


def state_to_histogram(state, basis, G, ustring):
    '''
    Calculate the diagonal term of the Clifford transformed ground state.
    A Clifford transform is a basis transform of the state, but we have hadamard gate.
    '''

    'CNOT transformation -- generate matrix.'

    
    if sum(ustring) !=0:
        xstring = (G@ustring)%2
        cnot = np.identity(len(xstring))
        hloc = 0
        count = 0
        hcount = 0
        for index, i in enumerate(xstring):
            if (i == 1) and count == 0:
                hloc = index
                mem = index
                #print('initial index', mem)
                count=1
            elif (i == 1) and count:
                #print(mem, index)
                cnot[index, mem] = 1
                mem = index
    
        #print(cnot)
        'first we compress'
        hbasis = []
        hcoe = []
        #Cbasis = []

        #print(len(state), len(basis))
        for index, i in enumerate(basis):
            
            cbasis = (cnot@G@i)%2
            'at the smallest non-0 entry we put the hadamard'
            cdual = cbasis.copy()
            cdual[hloc] +=1
            cdual%=2
            #print(i, cbasis, cdual, state[index])
            cstr = ''.join(np.array(np.array(cbasis, dtype = int), dtype = str))
            dstr = ''.join(np.array(np.array(cdual, dtype = int), dtype = str))
            hbasis.append(cstr)
            hbasis.append(dstr)
            coe = np.array([1/np.sqrt(2)*(-1)**cbasis[hloc], 1/np.sqrt(2)])*state[index]
            #print(coe)
            hcoe.append(coe[0])
            hcoe.append(coe[1])
            #Cbasis.append((G@i)%2)

    else:
        hbasis = []
        hcoe = []
        for index, i in enumerate(basis):
            cbasis = (G@i)%2
            cstr = ''.join(np.array(np.array(cbasis, dtype = int), dtype = str))
            hbasis.append(cstr)
            hcoe.append(state[index])

    'hist generation.'
    hist = dict()
    for (i, j) in zip(hbasis, hcoe):
        if i not in hist.keys():
            hist[i] = j
        else:
            #print(i)
            #print(hist[i], j)
            hist[i]+= j

    'Born'
    for (i, j) in hist.items():
        hist[i] = np.abs(j)**2
    
    
    return hist#,Cbasis 

def hist_format(state, basis, G, ustrings):
    '''Histogram format from experiment
    '''

    hists = []
    for i in ustrings:
        hist = state_to_histogram(state, basis, G, i)
        hists.append(hist)
    return hists

'----------------------------------------------------decoder----------------------------------------------------'
def equi4(i):
    '''
    Brute force encoder of equivalence in operators for real orbitals.
    '''
    equi = []

    equi.append(i)
    equi.append([i[1], i[0], i[2], i[3]])
    equi.append([i[0], i[1], i[3], i[2]])
    equi.append([i[1], i[0], i[3], i[2]])
    equi.append([i[2], i[3], i[0], i[1]])
    equi.append([i[3], i[2], i[0], i[1]])
    equi.append([i[2], i[3], i[1], i[0]])
    equi.append([i[3], i[2], i[1], i[0]])
    return equi

def equi2(i):
    '''
    Brute force encoder of equivalence in operators for real orbitals.
    '''
    equi = []
    equi.append(i)
    equi.append([i[1], i[0]])
    return equi


def proj(op):
    '''Projector to project out unrelated histograms.

    Input:
    ------
    op: the operator


    Output:
    -------
    projector and its location.
    
    '''

    ind = op.copy()
    for i in ind:
        if ind.count(i)!=1:
            ind.remove(i)
            
    s = []
    
    m = [1]*(len(ind)-len(op)//2)+ [0]*(len(ind)-len(op)//2)
    n = [1]
    a = 0
    #o = []
    for j in ind:
        if op.count(j) == 1:
            s = list(s) + [m[a]]
            a+=1
            #o.append(j)
        else:
            s = list(s)+ n

    return ind, s

def str_decode(string, table):
    '''
    A function to decode string: 

    Input:
    ------
    code: string to decode

    table: the dictionary generated

    Output:
    decoded: decoded word
    '''
    
    if string not in table.keys():
        return list(table.values())[0]
    else:
        decoded = table[string]
        
    return decoded

def hist_decoder(hist, G, ustring, op, table, hardcore):
    '''
    Input:
    ------
    hist: the sampled result.

    Output:
    -------
    exp: the final expectation value.
    '''
    signs = []
    for i in hist.keys():
        signs.append(decoder(i, G, ustring, op, table, hardcore))

    return np.array(signs)@np.array(list(hist.values()))

    

'This is the off-diagonal decoder -- just to include the group information.'
def decoder(string, G, ustring, op, table, hardcore):
    '''
    Decode only the ordered rdms.
    
    Input:
    ------
    N: the number of electrons
    
    string: from your histogram.

    G: the encoder.

    ustring: the unecoded Pauli-X

    op: the 2-RDM representation.

    Output:
    -------
    signs: The local sign information. 
    '''

    xstring = (G@ustring)%2
    A = np.triu(xstring)
    for index, i in enumerate(xstring):
        A[index]*=i

    'A is the inverse cnot transformation.'
    A = A+np.identity(len(xstring))
    A[A!=0] = 1
    A = A.T
    word = []
    for i in string:
        word.append(int(i))
    word = np.array(word)
    d = np.array((A@word)%2, dtype = int)
    d2 = np.array((xstring+d)%2, dtype = int)
    
    'for decoding the whole string.'
    d = ''.join(np.array(d, dtype = str))
    d2 = ''.join(np.array(d2, dtype = str))
    
    d = str_decode(d, table)
    d2 = str_decode(d2, table)

    d =  np.array(list(d), dtype = int)
    d2  = np.array(list(d2), dtype = int)

    'convert back to electronic pairs'
    if hardcore:
        a =  np.array([d, d]).T
        d = a.reshape(2*len(d))
        b =  np.array([d2, d2]).T
        d2 = b.reshape(2*len(d2))
        c =  np.array([ustring, ustring]).T
        ustring = c.reshape(2*len(ustring))

    'Check whether the pair adds up to the transition'
    ind, s = proj(op)

    'This is probably important'
    if list((d+d2)%2)!=list(ustring):
        #print(d, d2, (d+d2)%2, ustring)
        return 0


    'Check whether both of them do not pass the projector test. '
    l = []
    lconj = []
    d2 = (ustring+d)%2
    for loc in ind:
        l.append(d[loc])
        lconj.append(d2[loc])
    if l != s and lconj != s:
        return 0

    
    
    'seems like the sign operator has some error.'
    #print('This is d: ', d, d2)
    
    o = [i for i in range(len(ustring)) if ustring[i]!=0]
    sop = np.zeros(len(ustring))
    modes = np.arange(len(ustring))

    for index, j in enumerate(o):
        p = j*np.ones(len(ustring))
        sop+=(modes<p)

    
    sop = sop%2
    
    if len(o) == 2:
        sop = (ustring+sop)%2*sop
        
    'then calculate sign.'
    sign = (sop@d)%2

    
    for index, i in enumerate(xstring):
        if i!=0:
            #print(index)
            #print(sign, word)
            val = (sign+word[index])#*(sum(ustring)//2+1))

            return (-1)**val
    
    
    return (-1)**sign


'----------------------------------------------------Energy----------------------------------------------------'

def xfun(op, M):
    '''
    local evaluation of xstring

    '''
    ustring = np.zeros(M)
    for ele in op:
        ustring[ele]+=1
    return ustring%2

def hard_ustring(ustrings):
    '''
    Convert to hardcore bosonic format
    '''
    newu=  []
    for i in ustrings:
        i = np.reshape(i, (i.shape[0]//2, 2))
        newu.append(i.T[0])

    return newu

def rdm_val(experiments, tab, groups, signs, G, hardcore):
    '''
    test my decoder.

    Input:
    ------

    experiment: your histograms.

    tab: the look up table.
    
    groups: your measurement group.

    signs: your global signs.

    G: your encoder.

    The last two is for benchmarking.
    
    state: your ground state.

    basis: your basis.

    table: your decoder table.

    Output:
    -------
    ada: partial first electronic integral

    ddaa: partial second electronic integral
    '''

    'for debug'
    Vals = []
    da = np.zeros((G.shape[1], G.shape[1]))
    ddaa = np.zeros((G.shape[1], G.shape[1], G.shape[1], G.shape[1]))
    for i in range(len(groups)):
        ustring =xfun(groups[i][0], G.shape[1]*2**hardcore)
        if hardcore:
            ustring = np.reshape(ustring, (ustring.shape[0]//2, 2))
            ustring = ustring.T[0]
            
        experiment = experiments[i]
        vals = []
        for j in range(len(groups[i])):
            'initial information'
            loc = groups[i][j]
            sign = signs[i][j]
            
            val = hist_decoder(experiment, G, ustring, loc, tab, hardcore)

            if sum(ustring)!=0:
                val/=2
            val = sign*val
            vals.append(val)
            if len(loc) == 2:
                da[tuple(loc)] = val
                da[tuple(loc[::-1])] = val
            elif len(loc) == 4:
                ddaa[tuple(loc)] = val
                ddaa[tuple(loc[::-1])] = val
                ddaa[loc[1], loc[0], loc[2], loc[3]] = -val
                ddaa[loc[0], loc[1], loc[3], loc[2]] = -val
                ddaa[loc[1], loc[0], loc[3], loc[2]] = val
                ddaa[loc[2], loc[3], loc[0], loc[1]] = val
                ddaa[loc[3], loc[2], loc[0], loc[1]] = -val
                ddaa[loc[2], loc[3], loc[1], loc[0]] = -val
        Vals.append(vals)
                

    
    return da, ddaa, Vals
