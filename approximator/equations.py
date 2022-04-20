'''
Equations used in approximation study
'''
from compyle.api import declare
from pysph.sph.wc.linalg import gj_solve, augmented_matrix
from pysph.sph.equation import Equation, Group
from pysph.tools.interpolator import (SPHFirstOrderApproximationPreStep,
                                      SPHFirstOrderApproximation)
from pysph.sph.wc.kernel_correction import (
    GradientCorrectionPreStep, GradientCorrection,
    MixedGradientCorrection,
    MixedKernelCorrectionPreStep, KernelCorrection)
from pysph.sph.basic_equations import SummationDensity
from pysph.base.reduce_array import serial_reduce_array


# high order kernels
def w4(rij=0.0, hij=0.0):
    q = rij / hij

    wij = 0.0

    if q < 2:
        wij = 1 - 2.5 * q**2 + 1.5 * q**3
    elif q < 1:
        wij = 0.5 * (2 - q)**2 * (1-q)

    return wij


def w5(rij=0.0, hij=0.0):
    q = rij/hij

    wij = 0

    if q < 2.5:
        wij = 1/48 * (q-2.5)**3 * (7*q-7.5)
    if q < 1.5:
        wij = 1/48 * (165/4 + 20*q - 150*q**2 + 120*q**3 - 28*q**4)
    if q < 0.5:
        wij = 1/48 * (345/8 - 75*q**2 + 42*q**4)

    return wij


class SDHighOrderKernel(Equation):
    def _get_helpers_(self):
        return [w4, w5]

    def __init__(self, dest, sources, kernel='w4'):
        self.kernel = kernel

        super(SDHighOrderKernel, self).__init__(dest, sources)

    def initialize(self, d_idx, d_rho):
        d_rho[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, RIJ, HIJ,
             s_m):
        if self.kernel == 'w4':
            wij = w4(RIJ, HIJ)
        elif self.kernel == 'w5':
            wij = w5(RIJ, HIJ)

        d_rho[d_idx] += wij*s_m[s_idx]


class SPHHighOrderKernel(Equation):
    def _get_helpers_(self):
        return [w4, w5]

    def __init__(self, dest, sources, kernel='w4'):
        self.kernel = kernel

        super(SPHHighOrderKernel, self).__init__(dest, sources)

    def initialize(self, d_idx, d_prop):
        d_prop[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_prop, s_prop, RIJ, HIJ,
             s_m, s_rho):
        if self.kernel == 'w4':
            wij = w4(RIJ, HIJ)
        elif self.kernel == 'w5':
            wij = w5(RIJ, HIJ)

        d_prop[d_idx] += s_prop[s_idx]*wij*s_m[s_idx]/s_rho[s_idx]


class IterateSupportRadius(Equation):
    def __init__(self, dest, sources, V_ref, N, max_err=1e-4):
        self.V_ref = V_ref
        self.N = N
        self.err = 10.0
        self.max_err = max_err

        super(IterateSupportRadius, self).__init__(dest, sources)

    def initialize(self, d_idx, d_V, d_nnbr):
        d_V[d_idx] = 0.0

    def loop(self, d_idx, d_V, s_idx, s_m, WIJ, XIJ, d_nnbr):
        d_V[d_idx] += WIJ

    def post_loop(self, d_idx, d_V, d_nnbr, d_h):
        d_h[d_idx] = (d_V[d_idx]/self.V_ref)**(1.0/2) * d_h[d_idx]

    def reduce(self, dst, dt, t):
        print('reduce')
        import numpy as np
        vdiff =  serial_reduce_array(np.abs(dst.V - self.V_ref), 'sum')
        self.err = (vdiff/self.N)
        # self.err = vdiff
        print(self.err, dst.V, dst.h)

    def converged(self):
        print('converge', abs(self.err/self.V_ref))
        if abs(self.err/self.V_ref) < self.max_err:
            return 1
        else:
            return -1


class SmoothSupportRadius(Equation):
    def initialize(self, d_idx, d_prop):
        d_prop[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_prop, s_h, WIJ,
             s_m, s_rho):
        d_prop[d_idx] += s_h[s_idx]*WIJ*s_m[s_idx]/s_rho[s_idx]

    def post_loop(self, d_h, d_prop, d_idx):
        d_h[d_idx] = d_prop[d_idx]


class SummationDensityNew(Equation):
    def initialize(self, d_idx, d_prop, d_nnbr):
        d_prop[d_idx] = 0.0
        d_nnbr[d_idx] = 0.0

    def loop(self, d_idx, d_prop, s_idx, s_m, WIJ, d_nnbr):
        d_prop[d_idx] += s_m[s_idx]*WIJ
        d_nnbr[d_idx] += 1.0


class NumberDensity(Equation):
    def initialize(self, d_idx, d_prop):
        d_prop[d_idx] = 0.0

    def loop(self, d_idx, d_prop, s_idx, s_m, WIJ):
        d_prop[d_idx] += WIJ


class SummationDensityIterated(Equation):
    def __init__(self, dest, sources, err, norm='l1'):
        self.err0 = err
        self.err = 1e-14
        self.norm = norm

        super(SummationDensityIterated, self).__init__(dest, sources)

    def initialize(self, d_idx, d_prop, d_nnbr):
        d_prop[d_idx] = 0.0
        d_nnbr[d_idx] = 0.0

    def loop(self, d_idx, d_prop, s_idx, s_m, WIJ, d_nnbr):
        d_prop[d_idx] += s_m[s_idx]*WIJ
        d_nnbr[d_idx] += 1.0

    def post_loop(self, d_idx, d_h):
        # increase h
        d_h[d_idx] += 0.02 * d_h[d_idx]

    def reduce(self, dst, t, dt):
        import numpy as np
        f = declare('object')
        f = np.sin(2*np.pi*dst.x)
        if self.norm == 'l1':
            rho =  serial_reduce_array(np.abs(dst.prop - f), 'sum')
            self.err = (rho/len(dst.rho))
        elif self.norm == 'l2':
            rho =  serial_reduce_array((dst.prop - f)**2, 'sum')
            self.err = np.sqrt(rho/len(dst.rho))
        elif self.norm == 'linf':
            rho =  serial_reduce_array(np.abs(dst.prop - f), 'max')
            self.err = rho

    def converged(self):
        print(self.err, self.err0)
        if self.err - self.err0 < 1e-14:
            return 1
        else:
            return -1


class UpdateH(Equation):
    def post_loop(self, d_idx, d_h):
        d_h[d_idx] += 0.02 * d_h[d_idx]


class UpdateDestH(Equation):
    def initialize_pair(self, d_idx, d_h, s_h):
        d_h[d_idx] = s_h[d_idx]


class SPH(Equation):
    def initialize(self, d_idx, d_prop):
        d_prop[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_prop, s_prop, WIJ,
             s_m, s_rho):
        d_prop[d_idx] += s_prop[s_idx]*WIJ*s_m[s_idx]/s_rho[s_idx]


class SPHf(Equation):
    def initialize(self, d_idx, d_f):
        d_f[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_f, s_prop, WIJ,
             s_m, s_rho):
        d_f[d_idx] += s_prop[s_idx]*WIJ*s_m[s_idx]/s_rho[s_idx]


class SPHDerivative(Equation):
    def initialize(self, d_idx, d_prop):
        d_prop[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_prop, s_prop, DWIJ,
             s_m, s_rho, s_x, d_x, s_h):
        d_prop[d_idx] += s_prop[s_idx]*DWIJ[0]*s_m[s_idx]/s_rho[s_idx]


class SPHDerivativeMonaghan1992(Equation):
    def initialize(self, d_idx, d_prop):
        d_prop[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_prop, d_f, s_prop, DWIJ,
             s_m, s_rho, s_x, d_x, s_h):
        d_prop[d_idx] += (s_prop[s_idx] - d_f[d_idx]) *\
            DWIJ[0] * s_m[s_idx] / s_rho[s_idx]


class SummationDensityLibersky(Equation):
    def initialize(self, d_idx, d_wij, d_rho_n):
        d_rho_n[d_idx] = 0.0
        d_wij[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_rho, WIJ,
             s_m, d_wij):
        d_wij[d_idx] += WIJ * s_m[s_idx] / s_rho[s_idx]

    def post_loop(self, d_idx, d_rho_n, d_wij, d_rho):
        if d_wij[d_idx] > 1e-14:
            d_rho_n[d_idx] = d_rho[d_idx] / d_wij[d_idx]
        else:
            d_rho_n[d_idx] = d_rho[d_idx]


class SPHLibersky(Equation):
    def initialize(self, d_idx, d_prop):
        d_prop[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_prop, s_prop, WIJ,
             s_m, s_rho_n):
        d_prop[d_idx] += s_prop[s_idx]*WIJ*s_m[s_idx]/s_rho_n[s_idx]


class SPHLiberskyf(Equation):
    def initialize(self, d_idx, d_f):
        d_f[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_f, s_prop, WIJ,
             s_m, s_rho_n):
        d_f[d_idx] += s_prop[s_idx]*WIJ*s_m[s_idx]/s_rho_n[s_idx]


class SPHDerivativeLibersky(Equation):
    def initialize(self, d_idx, d_prop, d_m1):
        d_m1[d_idx] = 0.0
        d_prop[d_idx] = 0.0

    def loop(self, d_idx, d_prop, s_prop, d_f, s_idx, s_rho_n,
             DWIJ, s_m, XIJ, d_m1):
        d_m1[d_idx] += s_m[s_idx] * XIJ[0] * DWIJ[0] / s_rho_n[s_idx]
        d_prop[d_idx] += (s_prop[s_idx] -
                          d_f[d_idx]) * s_m[s_idx] * DWIJ[0] / s_rho_n[s_idx]

    def post_loop(self, d_prop, d_m1, d_idx):
        if d_m1[d_idx] > 1e-14:
            d_prop[d_idx] /= d_m1[d_idx]


class CorrectiveSPH(Equation):
    def initialize(self, d_wij, d_idx, d_prop):
        d_wij[d_idx] = 0.0
        d_prop[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_rho, d_prop, s_prop,
             WIJ, s_m, d_wij):
        d_wij[d_idx] += WIJ * s_m[s_idx] / s_rho[s_idx]
        d_prop[d_idx] += s_prop[s_idx] * s_m[s_idx] * WIJ / s_rho[s_idx]

    def post_loop(self, d_idx, d_prop, d_wij):
        if d_wij[d_idx] > 1e-14:
            d_prop[d_idx] /= d_wij[d_idx]


class CorrectiveSPHf(Equation):
    def initialize(self, d_idx, d_f, d_wij):
        d_wij[d_idx] = 0.0
        d_f[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_rho, d_f, s_prop,
             WIJ, s_m, d_wij):
        d_wij[d_idx] += WIJ * s_m[s_idx] / s_rho[s_idx]
        d_f[d_idx] += s_prop[s_idx] * s_m[s_idx] * WIJ / s_rho[s_idx]

    def post_loop(self, d_idx, d_f, d_wij):
        if d_wij[d_idx] > 1e-14:
            d_f[d_idx] /= d_wij[d_idx]


class CorrectiveSPHDerivative(Equation):
    def initialize(self, d_idx, d_prop, d_m1):
        d_m1[d_idx] = 0.0
        d_prop[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_rho, d_prop, s_prop,
             DWIJ, s_m, d_m1, XIJ, d_f):
        d_m1[d_idx] += DWIJ[0] * XIJ[0] * s_m[s_idx] / s_rho[s_idx]
        d_prop[d_idx] += (s_prop[s_idx] -
                          d_f[d_idx]) * s_m[s_idx] * DWIJ[0] / s_rho[s_idx]

    def post_loop(self, d_idx, d_prop, d_m1):
        if d_m1[d_idx] > 1e-14:
            d_prop[d_idx] /= d_m1[d_idx]


class GradientFreePreStep(Equation):
    def __init__(self, dest, sources, dim=1):
        self.dim = dim

        super(GradientFreePreStep, self).__init__(dest, sources)

    def initialize(self, d_idx, d_moment):
        i, j = declare('int', 2)

        for i in range(4):
            for j in range(4):
                d_moment[16*d_idx + j+4*i] = 0.0

    def loop(self, d_idx, s_idx, d_h, s_h, s_x, s_y, s_z, d_x, d_y, d_z, s_rho,
             s_m, WIJ, XIJ, d_moment):
        Vj = s_m[s_idx] / s_rho[s_idx]
        i16 = declare('int')
        i16 = 16*d_idx

        d_moment[i16+0] += WIJ * Vj

        d_moment[i16+1] += -XIJ[0] * WIJ * Vj
        d_moment[i16+2] += -XIJ[1] * WIJ * Vj
        d_moment[i16+3] += -XIJ[2] * WIJ * Vj

        d_moment[i16+4] += -XIJ[0] * WIJ * Vj
        d_moment[i16+8] += -XIJ[1] * WIJ * Vj
        d_moment[i16+12] += -XIJ[2] * WIJ * Vj

        d_moment[i16+5] += XIJ[0]**2 * WIJ * Vj
        d_moment[i16+6] += XIJ[0] * XIJ[1] * WIJ * Vj
        d_moment[i16+7] += XIJ[0] * XIJ[2] * WIJ * Vj

        d_moment[i16+9] += XIJ[1] * XIJ[0] * WIJ * Vj
        d_moment[i16+10] += XIJ[1] * XIJ[1] * WIJ * Vj
        d_moment[i16+11] += XIJ[1] * XIJ[2] * WIJ * Vj

        d_moment[i16+13] += XIJ[2] * XIJ[0] * WIJ * Vj
        d_moment[i16+14] += XIJ[2] * XIJ[1] * WIJ * Vj
        d_moment[i16+15] += XIJ[2] * XIJ[2] * WIJ * Vj


class GradientFree(Equation):
    """ First order SPH approximation
    """
    def _get_helpers_(self):
        return [gj_solve, augmented_matrix]

    def __init__(self, dest, sources, dim=1):
        self.dim = dim

        super(GradientFree, self).__init__(dest, sources)

    def initialize(self, d_idx, d_prop, d_p_sph):
        i = declare('int')

        for i in range(3):
            d_prop[4*d_idx+i] = 0.0
            d_p_sph[4*d_idx+i] = 0.0

    def loop(self, d_idx, d_h, s_h, s_x, s_y, s_z, d_x, d_y, d_z, s_rho,
             s_m, WIJ, XIJ, s_temp_prop, d_p_sph, s_idx):
        i4 = declare('int')
        Vj = s_m[s_idx] / s_rho[s_idx]
        pj = s_temp_prop[s_idx]
        i4 = 4*d_idx

        d_p_sph[i4+0] += pj * WIJ * Vj
        d_p_sph[i4+1] += -XIJ[0] * pj * WIJ * Vj
        d_p_sph[i4+2] += -XIJ[1] * pj * WIJ * Vj
        d_p_sph[i4+3] += -XIJ[2] * pj * WIJ * Vj

    def post_loop(self, d_idx, d_moment, d_prop, d_p_sph):

        a_mat = declare('matrix(16)')
        aug_mat = declare('matrix(20)')
        b = declare('matrix(4)')
        res = declare('matrix(4)')

        i, n, i16, i4 = declare('int', 4)
        i16 = 16*d_idx
        i4 = 4*d_idx
        for i in range(16):
            a_mat[i] = d_moment[i16+i]
        for i in range(20):
            aug_mat[i] = 0.0
        for i in range(4):
            b[i] = d_p_sph[4*d_idx+i]
            res[i] = 0.0

        n = self.dim + 1
        augmented_matrix(a_mat, b, n, 1, 4, aug_mat)
        gj_solve(aug_mat, n, 1, res)
        for i in range(4):
            d_prop[i4+i] = res[i]


class FIAPreStep(Equation):
    def __init__(self, dest, sources, dim=1):
        self.dim = dim

        super(FIAPreStep, self).__init__(dest, sources)

    def initialize(self, d_idx, d_moment):
        i, j = declare('int', 2)

        for i in range(3):
            for j in range(3):
                d_moment[9*d_idx + j+4*i] = 0.0

    def loop(self, d_idx, s_idx, d_h, s_h, s_x, s_y, s_z, d_x, d_y, d_z, s_rho,
             s_m, WIJ, XIJ, d_moment):
        Vj = s_m[s_idx] / s_rho[s_idx]
        i9, i, j = declare('int', 3)
        i9 = 9*d_idx

        for i in range(self.dim):
            for j in range(self.dim):
                d_moment[i9+3*i+j] += XIJ[i] * XIJ[j] * WIJ * Vj


class FIA(Equation):
    """ Rosswog
    """
    def _get_helpers_(self):
        return [gj_solve, augmented_matrix]

    def __init__(self, dest, sources, dim=1):
        self.dim = dim

        super(FIA, self).__init__(dest, sources)

    def initialize(self, d_idx, d_prop, d_p_sph):
        i = declare('int')

        for i in range(3):
            d_prop[3*d_idx+i] = 0.0
            d_p_sph[3*d_idx+i] = 0.0

    def loop(self, d_idx, d_h, s_h, s_x, s_y, s_z, d_x, d_y, d_z, s_rho,
             s_m, WIJ, XIJ, s_temp_prop, d_p_sph, s_idx, d_temp_prop):
        i4 = declare('int')
        Vj = s_m[s_idx] / s_rho[s_idx]
        pj = s_temp_prop[s_idx]
        i4 = 3*d_idx

        d_p_sph[i4+0] += -XIJ[0] * (pj - d_temp_prop[d_idx]) * WIJ * Vj
        d_p_sph[i4+1] += -XIJ[1] * (pj - d_temp_prop[d_idx]) * WIJ * Vj
        d_p_sph[i4+2] += -XIJ[2] * (pj - d_temp_prop[d_idx]) * WIJ * Vj

    def post_loop(self, d_idx, d_moment, d_prop, d_p_sph):

        a_mat = declare('matrix(9)')
        aug_mat = declare('matrix(12)')
        b = declare('matrix(3)')
        res = declare('matrix(3)')

        i, n, i16, i4 = declare('int', 4)
        i16 = 9*d_idx
        i4 = 3*d_idx
        for i in range(9):
            a_mat[i] = d_moment[i16+i]
        for i in range(12):
            aug_mat[i] = 0.0
        for i in range(3):
            b[i] = d_p_sph[i4+i]
            res[i] = 0.0

        n = self.dim
        augmented_matrix(a_mat, b, n, 1, 3, aug_mat)
        gj_solve(aug_mat, n, 1, res)
        for i in range(3):
            d_prop[i4+i] = res[i]


def get_sph(derv=0, dim=1, N=200, vref=0.0):
    eqns = []
    eqns.append(
        Group(equations=[SummationDensity(dest='src', sources=['src'])], update_nnps=True))
    if derv == 0:
        eqns.append(Group(equations=[SPH(dest="dest", sources=["src"])]))
    elif derv == 1:
        eqns.append(
            Group(equations=[SPHDerivative(dest="dest", sources=["src"])]))
    return eqns


def get_sph_var_h(derv=0, dim=1, N=200, vref=0.0, max_err=1e-4):
    eqns = []
    eqns.append(
        Group(equations=[IterateSupportRadius(dest='src', sources=['src'], V_ref=vref, N=N**2, max_err=max_err)], update_nnps=True, iterate=True, max_iterations=1000, min_iterations=1))
    eqns.append(
        Group(equations=[
            UpdateDestH(dest='dest', sources=['src'])], update_nnps=True)
            )
    eqns.append(
        Group(equations=[
            SummationDensity(dest='src', sources=['src'])], update_nnps=True)
            )
    if derv == 0:
        eqns.append(Group(equations=[SPH(dest="dest", sources=["src"])]))
    elif derv == 1:
        eqns.append(
            Group(equations=[SPHDerivative(dest="dest", sources=["src"])]))
    return eqns


def save_var_h(derv=0, dim=1, N=200, vref=0.0, max_err=1e-4):
    eqns = []
    eqns.append(
        Group(equations=[IterateSupportRadius(dest='src', sources=['src'], V_ref=vref, N=N**2, max_err=max_err)], update_nnps=True, iterate=True, max_iterations=1000, min_iterations=1))
    return eqns


def get_monaghan1992(derv=0, dim=1):
    eqns = []
    eqns.append(
        Group(equations=[SummationDensity(dest='src', sources=['src'])], update_nnps=True))
    if derv == 0:
        eqns.append(Group(equations=[SPH(dest="dest", sources=["src"])]))
    elif derv == 1:
        eqns.append(Group(equations=[SPHf(dest="dest", sources=["src"])]))
        eqns.append(
            Group(equations=[
                SPHDerivativeMonaghan1992(dest="dest", sources=["src"])
            ]))
    return eqns


def get_libersky(derv=0, dim=1):
    eqns = []
    eqns.append(
        Group(equations=[SummationDensity(dest='src', sources=['src'])],
              update_nnps=True))
    eqns.append(
        Group(
            equations=[SummationDensityLibersky(dest='src', sources=['src'])],
            update_nnps=True))
    if derv == 0:
        eqns.append(
            Group(equations=[SPHLibersky(dest="dest", sources=["src"])]))
    elif derv == 1:
        eqns.append(
            Group(equations=[SPHLiberskyf(dest="dest", sources=["src"])], update_nnps=True))
        eqns.append(
            Group(equations=[
                SPHDerivativeLibersky(dest="dest", sources=["src"])
            ]))
    return eqns


def get_csph(derv=0, dim=1):
    eqns = []
    eqns.append(
        Group(equations=[SummationDensity(dest='src', sources=['src'])], update_nnps=True))
    if derv == 0:
        eqns.append(
            Group(equations=[CorrectiveSPH(dest="dest", sources=["src"])]))
    elif derv == 1:
        eqns.append(
            Group(equations=[CorrectiveSPHf(dest="dest", sources=["src"])], update_nnps=True))
        eqns.append(
            Group(equations=[
                CorrectiveSPHDerivative(dest="dest", sources=["src"])
            ]))
    return eqns


def get_kernel_correction(derv=0, dim=1):
    eqns = []
    eqns.append(
        Group(equations=[SummationDensity(dest='src', sources=['src'])], update_nnps=True))
    if derv == 0:
        eqns.append(Group(equations=[CorrectiveSPH(dest="dest", sources=["src"])]))
    elif derv == 1:
        eqns.append(
            Group(equations=[KernelCorrection(dest="dest", sources=["src"])]))
        eqns.append(
            Group(equations=[
                GradientCorrectionPreStep(
                    dest="dest", sources=["src"], dim=dim),
                CorrectiveSPHf(dest="dest", sources=['src'])
            ], update_nnps=True))
        eqns.append(
            Group(equations=[
                GradientCorrection(dest="dest", sources=["src"], dim=dim),
                SPHDerivativeMonaghan1992(dest="dest", sources=["src"]),
            ]))
    return eqns


def get_mixed_kernel_correction(derv=0, dim=1):
    eqns = []
    eqns.append(
        Group(equations=[SummationDensity(dest='src', sources=['src'])], update_nnps=True))
    if derv == 0:
        eqns.append(Group(equations=[SPH(dest="dest", sources=["src"])]))
    elif derv == 1:
        eqns.append(
            Group(equations=[KernelCorrection(dest="dest", sources=["src"])]))
        eqns.append(
            Group(equations=[
                MixedKernelCorrectionPreStep(
                    dest="dest", sources=["src"], dim=dim)
            ], update_nnps=True))
        eqns.append(
            Group(equations=[
                MixedGradientCorrection(dest="dest", sources=["src"], dim=dim),
                SPHDerivative(dest="dest", sources=["src"]),
            ]))
    return eqns


def get_order1(derv=0, dim=1):
    eqns = []
    eqns.append(
        Group(equations=[SummationDensity(dest='src', sources=['src'])], update_nnps=True))
    eqns.append(
        Group(equations=[
            SPHFirstOrderApproximationPreStep(dest='dest',
                                              sources=['src'],
                                              dim=dim)
        ],
              real=True, update_nnps=True))
    eqns.append(
        Group(equations=[
            SPHFirstOrderApproximation(dest='dest', sources=['src'], dim=dim)
        ],
              real=True))
    return eqns


def get_kgf(derv=0, dim=1):
    eqns = []
    eqns.append(
        Group(equations=[SummationDensity(dest='src', sources=['src'])], update_nnps=True))
    eqns.append(
        Group(equations=[
            GradientFreePreStep(dest='dest', sources=['src'], dim=dim)
        ],
              real=True, update_nnps=True))
    eqns.append(
        Group(equations=[GradientFree(dest='dest', sources=['src'], dim=dim)],
              real=True))
    return eqns


def get_fia(derv=0, dim=1):
    eqns = []
    eqns.append(
        Group(equations=[SummationDensity(dest='src', sources=['src'])], update_nnps=True))
    if derv == 0:
        eqns.append(Group(equations=[SPH(dest="dest", sources=["src"])]))
    elif derv == 1:
        eqns.append(
            Group(equations=[
                FIAPreStep(dest='dest', sources=['src'], dim=dim)
            ],
                real=True, update_nnps=True))
        eqns.append(
            Group(equations=[FIA(dest='dest', sources=['src'], dim=dim)],
                real=True))
    return eqns


def only_sd():
    eqns = []
    eqns.append(
        Group(equations=[SummationDensityNew(dest='dest', sources=['src'])]))
    return eqns


def only_nd():
    eqns = []
    eqns.append(
        Group(equations=[NumberDensity(dest='dest', sources=['src'])]))
    return eqns


def hdx_analysis(err, norm):
    eqns = []
    eqns.append(
        Group(equations=[
            SummationDensityIterated(dest='dest', sources=['src'], err=err, norm=norm),
            UpdateH(dest='src', sources=None)
        ],
              iterate=True,
              max_iterations=20,
              min_iterations=1,
              update_nnps=True))
    return eqns


def get_equations(
    method='sph', derv=0, dim=1, err=1e-14, norm='l1', N=100, vref=0.0, max_err=1e-4):
    eqns = None
    if method == 'sph':
        eqns = get_sph(derv=derv, dim=dim, N=N, vref=vref)
    if method == 'sph_varh':
        eqns = get_sph_var_h(derv=derv, dim=dim, N=N, vref=vref, max_err=max_err)
    elif method == 'monaghan1992':
        eqns = get_monaghan1992(derv=derv, dim=dim)
    elif method == 'libersky':
        eqns = get_libersky(derv=derv, dim=dim)
    elif method == 'csph':
        eqns = get_csph(derv=derv, dim=dim)
    elif method == 'order1':
        eqns = get_order1(derv=derv, dim=dim)
    elif method == 'kgf':
        eqns = get_kgf(derv=derv, dim=dim)
    elif method == 'kc':
        eqns = get_kernel_correction(derv=derv, dim=dim)
    elif method == 'mkc':
        eqns = get_mixed_kernel_correction(derv=derv, dim=dim)
    elif method == 'fia':
        eqns = get_fia(derv=derv, dim=dim)
    elif method == 'sd':
        eqns = only_sd()
    elif method == 'nd':
        eqns = only_nd()
    elif method == 'hda':
        eqns = hdx_analysis(err, norm)
    elif method == 'save_h':
        eqns = save_var_h(derv=derv, dim=dim, N=N, vref=vref, max_err=max_err)
    # print(eqns)
    return eqns