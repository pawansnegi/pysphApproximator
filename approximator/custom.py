from pysph.sph.equation import Equation, Group
from pysph.sph.basic_equations import SummationDensity
from approx.approx import Approx


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


class Custom(Approx):
    def get_props(self):
        return ['f', 'exact', {'name':'fx', 'stride':3}]

    def get_equations(self, dest, sources, derv=0):
        eqns = []
        eqns.append(
            Group(equations=[SummationDensity(dest=dest, sources=sources)], update_nnps=True))
        if derv == 0:
            eqns.append(Group(equations=[SPH(dest=dest, sources=sources)]))
        elif derv == 1:
            eqns.append(
                Group(equations=[SPHDerivative(dest=dest, sources=sources)]))
        return eqns

