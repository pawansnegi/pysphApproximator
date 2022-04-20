from compyle.api import declare
from pysph.sph.equation import Equation, Group
from pysph.sph.basic_equations import SummationDensity
from pysph.sph.wc.linalg import gj_solve, augmented_matrix, identity

class FirstOrderPreStep(Equation):
    # Liu et al 2005
    def __init__(self, dest, sources, dim=2):
        self.dim = dim

        super(FirstOrderPreStep, self).__init__(dest, sources)

    def initialize(self, d_idx, d_L):
        i, j = declare('int', 2)

        for i in range(4):
            for j in range(4):
                d_L[16*d_idx + j+4*i] = 0.0

    def loop(self, d_idx, s_idx, d_h, s_h, s_x, s_y, s_z, d_x, d_y, d_z, s_rho,
             s_m, WIJ, XIJ, DWIJ, d_L):
        Vj = s_m[s_idx] / s_rho[s_idx]
        i16 = declare('int')
        i16 = 16*d_idx

        d_L[i16+0] += WIJ * Vj

        d_L[i16+1] += -XIJ[0] * WIJ * Vj
        d_L[i16+2] += -XIJ[1] * WIJ * Vj
        d_L[i16+3] += -XIJ[2] * WIJ * Vj

        d_L[i16+4] += DWIJ[0] * Vj
        d_L[i16+8] += DWIJ[1] * Vj
        d_L[i16+12] += DWIJ[2] * Vj

        d_L[i16+5] += -XIJ[0] * DWIJ[0] * Vj
        d_L[i16+6] += -XIJ[1] * DWIJ[0] * Vj
        d_L[i16+7] += -XIJ[2] * DWIJ[0] * Vj

        d_L[i16+9] += - XIJ[0] * DWIJ[1] * Vj
        d_L[i16+10] += -XIJ[1] * DWIJ[1] * Vj
        d_L[i16+11] += -XIJ[2] * DWIJ[1] * Vj

        d_L[i16+13] += -XIJ[0] * DWIJ[2] * Vj
        d_L[i16+14] += -XIJ[1] * DWIJ[2] * Vj
        d_L[i16+15] += -XIJ[2] * DWIJ[2] * Vj


class FirstOrderCorrection(Equation):
    def _get_helpers_(self):
        return [gj_solve]

    def __init__(self, dest, sources, dim=2, tol=0.1):
        self.dim = dim
        self.tol = tol
        super(FirstOrderCorrection, self).__init__(dest, sources)

    def loop(self, d_idx, d_L, WIJ, XIJ, DWIJ, HIJ, d_wij):
        i, j, n, nt = declare('int', 4)
        n = self.dim + 1
        nt = n + 1
        # Note that we allocate enough for a 3D case but may only use a
        # part of the matrix.
        temp = declare('matrix(20)')
        res = declare('matrix(4)')
        YJI = declare('matrix(4)')
        for i in range(n):
            for j in range(n):
                temp[nt * i + j] = d_L[16 * d_idx + 4 * i + j]
            # Augmented part of matrix
            if i == 0:
                temp[nt*i + n] = WIJ
            else:
                temp[nt*i + n] = DWIJ[i-1]

        error_code = gj_solve(temp, n, 1, res)

        d_wij[d_idx] = res[0]
        for i in range(self.dim):
            DWIJ[i] = res[i+1]

class SPH(Equation):
    def initialize(self, d_idx, d_f):
        d_f[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_f, s_exact, WIJ,
             s_m, s_rho):
        d_f[d_idx] += s_exact[s_idx]*WIJ*s_m[s_idx]/s_rho[s_idx]


class SPHDerivative(Equation):
    def initialize(self, d_idx, d_fx):
        d_fx[3*d_idx] = 0.0
        d_fx[3*d_idx+1] = 0.0
        d_fx[3*d_idx+2] = 0.0

    def loop(self, d_idx, s_idx, d_fx, s_exact, DWIJ,
             s_m, s_rho):
        d_fx[3 *d_idx] += s_exact[s_idx]*DWIJ[0]*s_m[s_idx]/s_rho[s_idx]
        d_fx[3 *d_idx+1] += s_exact[s_idx]*DWIJ[1]*s_m[s_idx]/s_rho[s_idx]
        d_fx[3 *d_idx+2] += s_exact[s_idx]*DWIJ[2]*s_m[s_idx]/s_rho[s_idx]

def get_props():
    return ['f', 'exact', {'name':'fx', 'stride':3}, {'name':'L', 'stride':16}, 'wij']


def get_equations(dest, sources, derv=0, dim=0):
    eqns = []
    eqns.append(
        Group(equations=[SummationDensity(dest=dest, sources=sources)],
              update_nnps=True))
    eqns.append(
        Group(equations=[
            FirstOrderPreStep(
                dest=dest, sources=sources, dim=dim)
        ],
              real=True,
              update_nnps=True))
    g0 = [FirstOrderCorrection(dest=dest, sources=sources, dim=dim)]
    if derv == 0:
        g0.append(SPH(dest=dest, sources=sources))
    elif derv == 1:
        g0.append(SPHDerivative(dest=dest, sources=sources))
    eqns.append(Group(equations=g0))
    print(eqns)
    return eqns