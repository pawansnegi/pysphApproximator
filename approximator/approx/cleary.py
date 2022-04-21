
from pysph.sph.equation import Equation, Group
from pysph.sph.basic_equations import SummationDensity

class ViscosityCleary(Equation):
    def __init__(self, dest, sources, nu, rho0):
        r"""
        Parameters
        ----------
        nu : float
            kinematic viscosity
        """

        self.nu = nu
        self.rho0 = rho0
        super(ViscosityCleary, self).__init__(dest, sources)

    def initialize(self, d_idx, d_fxx):
        d_fxx[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, s_rho,
             d_fxx, s_m, R2IJ, DWIJ, VIJ, XIJ, RIJ, s_exact, d_exact):

        # # averaged shear viscosity Eq. (6)
        etai = self.nu * d_rho[d_idx]
        etaj = self.nu * s_rho[s_idx]

        etaij = 2 * (etai * etaj)/(etai + etaj)

        if RIJ > 1e-14:
            xijdotdwij = XIJ[0]*DWIJ[0] + XIJ[1]*DWIJ[1] + XIJ[2]*DWIJ[2]
            tmp = s_m[s_idx] / (s_rho[s_idx] * d_rho[d_idx])
            tmp = tmp * (2 * etaij) * (xijdotdwij/R2IJ)

            d_fxx[d_idx] += tmp * (d_exact[d_idx] - s_exact[s_idx])


def get_props():
    return ['exact', 'fxx']

def get_equations(dest, sources, derv=0, dim=2):
    eqns = []
    eqns.append(
        Group(equations=[SummationDensity(dest=dest, sources=sources)], update_nnps=True))
    if derv == 2:
        eqns.append(Group(equations=[ViscosityCleary(dest=dest, sources=sources, nu=1.0, rho0=1.0)]))
    else:
        raise NotImplementedError
    return eqns