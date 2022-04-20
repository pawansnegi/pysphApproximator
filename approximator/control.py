import numpy as np
import os
import sys
from pysph.base.utils import get_particle_array
from pysph.solver.application import Application
from pysph.solver.solver import Solver
from pysph.sph.integrator_step import EulerStep
from pysph.sph.integrator import EulerIntegrator
from pysph.base.nnps import DomainManager
import matplotlib
from equations import get_equations
# matplotlib.use('tkagg')
import matplotlib.pyplot as plt

pi = np.pi
c0 = 10


class MyStep(EulerStep):
    # dummy stepper
    def stage1(self):
        pass

class Approximator(Application):
    def initialize(self):
        self.dim = 2
        self.dx = 0.01
        self.use_sph = None
        self.rand = 0
        self.derv = 0
        self.approx = None

        self.L = 1.0

    def add_user_options(self, group):
        group.add_argument(
            '--dim', action='store', dest='dim', type=int, default=2,
            help='dimesion of the problem')

        group.add_argument(
            '--hdx', action='store', dest='hdx', type=float, default=1.0,
            help='h/dx')

        group.add_argument(
            '--N', action='store', dest='N', type=int, default=10,
            help='Number of particle in unit length')

        group.add_argument(
            "--use-sph", action="store", dest="use_sph",
            type=str, default='stan',
            help="sph, order1, monaghan1992, libersky"
            )

        group.add_argument(
            "--perturb", action="store", dest="perturb",
            type=int, default=0,
            help="if value greater the 0 the particles are perturbed 2 is packed"
            )

        group.add_argument(
            "--derv", action="store", dest="derv",
            type=int, default=0,
            help="if zero than find derivative"
            )

        group.add_argument(
            "--periodic", action="store", dest="periodic",
            type=int, default=0,
            help="if zero then not periodic"
            )

        group.add_argument(
            "--norm", action="store", dest="norm",
            type=str, default="l1",
            help='options are "l1, l2 and linf"'
            )

    def consume_user_options(self):
        self.dest = 'fluid'
        self.sources = ['fluid']
        self.hdx = self.options.hdx
        self.dim = self.options.dim
        self.rand = self.options.perturb
        self.derv = self.options.derv
        self.use_sph = self.options.use_sph
        self.periodic = self.options.periodic
        self.norm = self.options.norm
        self.N = self.options.N
        self.dx = self.L / self.N
        self._get_approx()

    def _get_approx(self):
        if self.use_sph == 'stan':
            import approx.standard as stn
            self.approx = stn
        if self.use_sph == 'order1':
            import approx.order1 as order1
            self.approx = order1
        elif self.use_sph == 'custom':
            import custom as cst
            self.approx = cst


    def get_function(self, x, y, z):
        if self.dim == 1:
            f = np.sin(2*pi*(x)/self.L)
        elif self.dim == 2:
            f = np.sin(2*pi*(x+y)/self.L)
        elif self.dim == 3:
            f = np.sin(2*pi*(x+y+z)/self.L)
        return f

    def get_function_grad(self, x, y, z):
        if self.dim == 1:
            fx =  2*pi/self.L * np.cos(2*pi*(x)/self.L)
        elif self.dim == 2:
            fx =  2*pi/self.L * np.cos(2*pi*(x+y)/self.L)
        elif self.dim == 3:
            fx =  2*pi/self.L * np.cos(2*pi*(x+y+z)/self.L)
        return fx

    def get_function_laplace(self, x, y, z):
        if self.dim == 1:
            fxx = -2* (2*pi/self.L)**2 * np.sin(2*pi*(x)/self.L)
        elif self.dim == 2:
            fxx =  -2* (2*pi/self.L)**2 * np.sin(2*pi*(x+y)/self.L)
        elif self.dim == 3:
            fxx = -2* (2*pi/self.L)**2 * np.sin(2*pi*(x+y+z)/self.L)
        return fxx

    def get_function_div(self, x, y):
        if self.dim == 1:
            fx =  2*pi/self.L * np.cos(2*pi*(x)/self.L)
        elif self.dim == 2:
            fx =  4*pi/self.L * np.cos(2*pi*(x+y)/self.L)
        elif self.dim == 3:
            fx =  6*pi/self.L * np.cos(2*pi*(x+y+z)/self.L)
        return fx

    def _get_fluid(self):
        L = self.L
        dx = self.dx
        nl = 6 * dx

        x, y, z = None, None, None
        _x = np.arange(dx / 2, L, dx)
        _y = np.zeros_like(_x)

        if self.dim == 1:
            x = _x.copy()
            y = np.zeros_like(x)
            z = np.zeros_like(x)
        if self.dim == 2:
            x, y = np.meshgrid(_x, _x)
            z = np.zeros_like(x)
        if self.dim == 3:
            x, y, z = np.meshgrid(_x, _x, _x)

        x, y, z = [t.ravel() for t in (x, y, z)]

        if self.rand == 1:
            if self.dim  > 0:
                x += 0.1 * (np.random.random(len(x)) + 0.5)/0.5 * dx
            if self.dim > 1:
                y += 0.1 * (np.random.random(len(x)) + 0.5)/0.5 * dx
            if self.dim > 2:
                z += 0.1 * (np.random.random(len(x)) + 0.5)/0.5 * dx
        elif self.rand == 2:
            if self.dim == 2:
                filename = 'nx%d.npz'%self.options.N
                dirname = os.path.dirname(os.path.abspath(__file__))
                print(dirname)
                datafile = os.path.join(dirname, 'data', filename)
                print(datafile)
                if os.path.exists(datafile):
                    print('here')
                    data = np.load(datafile)
                    x = data['x']
                    y = data['y']
                    x += 0.5
                    y += 0.5
                    print(x, y)
            else:
                print("Packing for %d dimension is absent"%self.dim)
                sys.exit(0)


        # Source setup
        h = np.ones_like(x)*dx*self.hdx
        rho = np.ones_like(x)
        m = rho*dx**(self.dim)

        fluid = get_particle_array(name="fluid", x=x, y=y, z=z, h=h, m=m, rho=rho, exact=0)
        return fluid

    def _add_properties(self, particles):
        props = None
        props = self.approx.get_props()

        for pa in particles:
            x = pa.x
            y = pa.y
            z = pa.z
            output_props = []
            for prop in props:
                if isinstance(prop, dict):
                    output_props.append(prop["name"])
                    pa.add_property(**prop)
                else:
                    output_props.append(prop)
                    pa.add_property(prop)
            pa.exact[:] = self.get_function(x, y, z)
            pa.add_output_arrays(output_props)

    def create_particles(self):
        hdx = self.hdx
        particles = [self._get_fluid()]

        self._add_properties(particles)

        return particles

    def create_domain(self):
        L = self.L
        if self.periodic:
            if self.dim == 1:
                return DomainManager(
                    xmin=0, xmax=L, periodic_in_x=True)
            elif self.dim == 2:
                return DomainManager(
                    xmin=0, xmax=L, ymin=0, ymax=L, periodic_in_x=True,
                    periodic_in_y=True)
            elif self.dim == 3:
                return DomainManager(
                    xmin=0, xmax=L, ymin=0, ymax=L, zmin=0, zmax=L, periodic_in_x=True,
                    periodic_in_y=True, periodic_in_z=True)

    def create_equations(self):
        eqns = None
        eqns = self.approx.get_equations(self.dest, self.sources, self.derv, self.dim)

        print(eqns)
        return eqns

    def create_solver(self):
        integrator = EulerIntegrator(fluid=MyStep())
        return Solver(dim=self.dim,
                      pfreq=1,
                      integrator=integrator,
                      tf=1.0,
                      dt=1.0)

    def post_process(self, info):
        from pysph.solver.utils import load
        self.read_info(info)
        if len(self.output_files) == 0:
            return

        data = load(self.output_files[-1])
        fluid = data['arrays']['fluid']
        x = fluid.x
        y = fluid.y
        z = fluid.z
        fc = None
        fe = None

        method = self.use_sph
        if self.derv == 0:
            fe = self.get_function(x, y, z)
            fc = fluid.f
        if self.derv == 1:
            fe = self.get_function_grad(x, y, z)
            fc = fluid.fx[0::3]
        if self.derv == 2:
            fe = self.get_function_laplace(x, y, z)
            fc = fluid.fxx[0::9]
        if self.derv == 3:
            fe = self.get_function_div(x, y, z)
            fc = fluid.div

        filename = os.path.join(self.output_dir, 'results.npz')
        print(filename)
        np.savez(filename, x=x, y=y, z=z, fc=fc, fe=fe)



if __name__ == "__main__":
    app = Approximator()
    app.run()
    app.post_process(app.info_filename)