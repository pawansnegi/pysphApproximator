
#customized quintic

from math import pi
from compyle.api import declare

from pysph.sph.equation import Equation
from pysph.sph.wc.linalg import augmented_matrix, gj_solve

def qs_dwdq(rij=1.0, h=1.0):
    h1 = 1. / h
    q = rij * h1

    # get the kernel normalizing factor
    fac = 7.0 * h1 * h1 / (pi * 478.)

    tmp3 = 3. - q
    tmp2 = 2. - q
    tmp1 = 1. - q

    # compute the gradient
    if (rij > 1e-12):
        if (q > 3.0):
            val = 0.0

        elif (q > 2.0):
            val = -5.0 * tmp3 * tmp3 * tmp3 * tmp3

        elif (q > 1.0):
            val = -5.0 * tmp3 * tmp3 * tmp3 * tmp3
            val += 30.0 * tmp2 * tmp2 * tmp2 * tmp2
        else:
            val = -5.0 * tmp3 * tmp3 * tmp3 * tmp3
            val += 30.0 * tmp2 * tmp2 * tmp2 * tmp2
            val -= 75.0 * tmp1 * tmp1 * tmp1 * tmp1
    else:
        val = 0.0

    return val * fac

def qs_dwdq2(xij=[0.0, 0.0, 0.0], rij=1.0, h=1.0, d2wij=[0.0, 0.0, 0.0]):
    h1 = 1. / h
    q = rij * h1

    # get the kernel normalizing factor
    fac = 7.0 * h1 * h1 / (pi * 478.)

    tmp3 = 3. - q
    tmp2 = 2. - q
    tmp1 = 1. - q

    # compute the second gradient
    if (rij > 1e-12):
        if (q > 3.0):
            val = 0.0

        elif (q > 2.0):
            val = 20.0 * tmp3 * tmp3 * tmp3

        elif (q > 1.0):
            val = 20.0 * tmp3 * tmp3 * tmp3
            val -= 120.0 * tmp2 * tmp2 * tmp2
        else:
            val = 20.0 * tmp3 * tmp3 * tmp3
            val -= 120.0 * tmp2 * tmp2 * tmp2
            val += 300.0 * tmp1 * tmp1 * tmp1
    else:
        val = 0.0

    dwdq = qs_dwdq(rij, h)
    dw2dq2 = fac * val
    fac2 = 1.0 / ( rij**2 * h**2)
    if rij > 1e-14:
        t1 = fac2 * (dw2dq2 - dwdq / q)
        d2wij[0] = t1 * xij[0] * xij[0] + dwdq/ (h**2 * q)
        d2wij[1] = t1 * xij[0] * xij[1]
        d2wij[2] = t1 * xij[1] * xij[1] + dwdq/ (h**2 * q)
    else:
        d2wij[0] = dw2dq2 / h**2
        d2wij[1] = dw2dq2 / h**2
        d2wij[2] = dw2dq2 / h**2


class SecondOrderCorrectionPreStep(Equation):
    def _get_helpers_(self):
        return [qs_dwdq2, qs_dwdq]

    def initialize(self, d_idx, d_m_mat_2):
        i = declare('int')
        for i in range(36):
            d_m_mat_2[36*d_idx + i] = 0.0

    def loop(self, d_idx, XIJ, RIJ, d_m_mat_2, DWIJ, WIJ, HIJ, s_m, s_rho, s_idx):
        d2wij = declare('matrix(3)')
        omega = s_m[s_idx] / s_rho[s_idx]

        qs_dwdq2(XIJ, RIJ, HIJ, d2wij)

        d_m_mat_2[36 * d_idx] += WIJ * omega
        d_m_mat_2[36 * d_idx + 1] += -XIJ[0] * WIJ * omega
        d_m_mat_2[36 * d_idx + 2] += -XIJ[1] * WIJ * omega
        d_m_mat_2[36 * d_idx + 3] += 0.5 * XIJ[0] * XIJ[0] * WIJ * omega
        d_m_mat_2[36 * d_idx + 4] += 0.5 * XIJ[0] * XIJ[1] * WIJ * omega
        d_m_mat_2[36 * d_idx + 5] += 0.5 * XIJ[1] * XIJ[1] * WIJ * omega

        d_m_mat_2[36 * d_idx + 6] += DWIJ[0] * omega
        d_m_mat_2[36 * d_idx + 7] += -XIJ[0] * DWIJ[0] * omega
        d_m_mat_2[36 * d_idx + 8] += -XIJ[1] * DWIJ[0] * omega
        d_m_mat_2[36 * d_idx + 9] += 0.5 * XIJ[0] * XIJ[0] * DWIJ[0] * omega
        d_m_mat_2[36 * d_idx + 10] += 0.5 * XIJ[0] * XIJ[1] * DWIJ[0] * omega
        d_m_mat_2[36 * d_idx + 11] += 0.5 * XIJ[1] * XIJ[1] * DWIJ[0] * omega

        d_m_mat_2[36 * d_idx + 12] += DWIJ[1] * omega
        d_m_mat_2[36 * d_idx + 13] += -XIJ[0] * DWIJ[1] * omega
        d_m_mat_2[36 * d_idx + 14] += -XIJ[1] * DWIJ[1] * omega
        d_m_mat_2[36 * d_idx + 15] += 0.5 * XIJ[0] * XIJ[0] * DWIJ[1] * omega
        d_m_mat_2[36 * d_idx + 16] += 0.5 * XIJ[0] * XIJ[1] * DWIJ[1] * omega
        d_m_mat_2[36 * d_idx + 17] += 0.5 * XIJ[1] * XIJ[1] * DWIJ[1] * omega

        d_m_mat_2[36 * d_idx + 18] += d2wij[0] * omega
        d_m_mat_2[36 * d_idx + 19] += -XIJ[0] * d2wij[0] * omega
        d_m_mat_2[36 * d_idx + 20] += -XIJ[1] * d2wij[0] * omega
        d_m_mat_2[36 * d_idx + 21] += 0.5 * XIJ[0] * XIJ[0] * d2wij[0] * omega
        d_m_mat_2[36 * d_idx + 22] += 0.5 * XIJ[0] * XIJ[1] * d2wij[0] * omega
        d_m_mat_2[36 * d_idx + 23] += 0.5 * XIJ[1] * XIJ[1] * d2wij[0] * omega

        d_m_mat_2[36 * d_idx + 24] += d2wij[1] * omega
        d_m_mat_2[36 * d_idx + 25] += -XIJ[0] * d2wij[1] * omega
        d_m_mat_2[36 * d_idx + 26] += -XIJ[1] * d2wij[1] * omega
        d_m_mat_2[36 * d_idx + 27] += 0.5 * XIJ[0] * XIJ[0] * d2wij[1] * omega
        d_m_mat_2[36 * d_idx + 28] += 0.5 * XIJ[0] * XIJ[1] * d2wij[1] * omega
        d_m_mat_2[36 * d_idx + 29] += 0.5 * XIJ[1] * XIJ[1] * d2wij[1] * omega

        d_m_mat_2[36 * d_idx + 30] += d2wij[2] * omega
        d_m_mat_2[36 * d_idx + 31] += -XIJ[0] * d2wij[2] * omega
        d_m_mat_2[36 * d_idx + 32] += -XIJ[1] * d2wij[2] * omega
        d_m_mat_2[36 * d_idx + 33] += 0.5 * XIJ[0] * XIJ[0] * d2wij[2] * omega
        d_m_mat_2[36 * d_idx + 34] += 0.5 * XIJ[0] * XIJ[1] * d2wij[2] * omega
        d_m_mat_2[36 * d_idx + 35] += 0.5 * XIJ[1] * XIJ[1] * d2wij[2] * omega


class SecondOrderCorrection(Equation):
    def _get_helpers_(self):
        # return [gj_solve]
        return [gj_solve, augmented_matrix, qs_dwdq2, qs_dwdq]

    def __init__(self, dest, sources, dim=2, tol=0.1):
        self.dim = dim
        self.tol = tol
        super(SecondOrderCorrection, self).__init__(dest, sources)

    def loop(self, d_idx, d_m_mat_2, WIJ, XIJ, DWIJ, HIJ, RIJ, d_wij):
        i, j, n, nt = declare('int', 4)
        d2wij = declare('matrix(3)')
        n = 6
        nt = n + 1
        # Note that we allocate enough for a 3D case but may only use a
        # part of the matrix.
        qs_dwdq2(XIJ, RIJ, HIJ, d2wij)
        temp = declare('matrix(42)')
        res = declare('matrix(6)')
        for i in range(36):
            temp[i] = 0.0
        for i in range(6):
            res[i] = 0.0

        for i in range(n):
            for j in range(n):
                temp[nt * i + j] = d_m_mat_2[36 * d_idx + 6 * i + j]
            # Augmented part of matrix
            if i == 0:
                temp[nt*i + n] = WIJ
            elif (i>0 and i<3):
                temp[nt*i + n] = DWIJ[i-1]
            elif (i==3):
                temp[nt*i + n] = d2wij[0]
            elif (i==4):
                temp[nt*i + n] = d2wij[1]
            elif (i==5):
                temp[nt*i + n] = d2wij[2]

        # print(temp, d2wij)
        error_code = gj_solve(temp, n, 1, res)

        # print(res)
        d_wij[d_idx] = res[0]
        for i in range(2):
            DWIJ[i] = res[i+1]
        # d_wij2[0] = res[3]
        # d_wij2[1] = res[4]
        # d_wij2[2] = res[5]
        # d_wij2[3] = res[6]


class SPHPressure(Equation):
    def initialize(self, d_idx, d_f):
        d_f[d_idx] = 0.0

    def loop(self, d_idx, d_f, s_exact, d_wij, s_m, s_rho, s_idx):
        d_f[d_idx] += s_exact[s_idx] * d_wij[d_idx] * s_m[s_idx] / s_rho[d_idx]


class SPHVelocity(Equation):
    def initialize(self, d_idx, d_u, d_v):
        d_u[d_idx] = 0.0
        d_v[d_idx] = 0.0

    def loop(self, d_idx, d_u, s_u, d_v, s_v, d_wij, s_m, s_rho, s_idx):
        d_u[d_idx] += s_u[s_idx] * d_wij[d_idx] * s_m[s_idx] / s_rho[d_idx]
        d_v[d_idx] += s_v[s_idx] * d_wij[d_idx] * s_m[s_idx] / s_rho[d_idx]

def get_props():
    return ['f', 'exact', "wij", {"name":"m_mat_2", "stride":36}]

def get_equations(dest, sources, derv=0, dim=2):
    from pysph.sph.basic_equations import SummationDensity
    from pysph.sph.equation import Group
    from pysph.tools.sph_evaluator import SPHEvaluator
    from pysph.base.kernels import QuinticSpline

    eqns = []

    if not(dim==2):
        raise NotImplementedError
    if not (derv==0):
        raise NotImplementedError
    g0 = []
    g0.append(SummationDensity(dest=dest, sources=sources))
    eqns.append(Group(equations=g0))

    g0 = [SecondOrderCorrectionPreStep(dest=dest, sources=sources)]
    eqns.append(Group(equations=g0))

    g0 = []
    g0.append(SecondOrderCorrection(dest=dest, sources=sources))
    g0.append(SPHPressure(dest=dest, sources=sources))
    eqns.append(Group(equations=g0))

    return eqns