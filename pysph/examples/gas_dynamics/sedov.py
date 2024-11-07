"""Sedov point explosion problem. (7 minutes)

Particles are distributed on concentric circles about the origin with
increasing number of particles with increasing radius. A unit charge
is distributed about the center which gives the initial pressure
disturbance.

"""
# NumPy and standard library imports
import os.path
import numpy

# PySPH base and carray imports
from pysph.base.utils import get_particle_array as gpa
from pysph.solver.application import Application
from pysph.sph.scheme import GasDScheme, SchemeChooser
from pysph.sph.gas_dynamics.psph import PSPHScheme
from pysph.sph.gas_dynamics.tsph import TSPHScheme
from pysph.sph.gas_dynamics.magma2 import MAGMA2Scheme

# Numerical constants
dim = 2
gamma = 5.0/3.0
gamma1 = gamma - 1.0

# solution parameters
dt = 1e-4
tf = 0.1

# scheme constants
alpha1 = 10.0
alpha2 = 1.0
beta = 2.0
kernel_factor = 1.2


class SedovPointExplosion(Application):
    def create_particles(self):
        fpath = os.path.join(
            os.path.dirname(__file__), 'ndspmhd-sedov-initial-conditions.npz'
        )
        data = numpy.load(fpath)
        x = data['x']
        y = data['y']
        rho = data['rho']
        p = data['p']
        e = data['e'] + 1e-9
        h = data['h']
        m = data['m']

        fluid = gpa(name='fluid', x=x, y=y, rho=rho, p=p, e=e, h=h, m=m)
        self.scheme.setup_properties([fluid])

        # set the initial smoothing length proportional to the particle
        # volume
        fluid.h[:] = kernel_factor * (fluid.m/fluid.rho)**(1./dim)

        print("Sedov's point explosion with %d particles"
              % (fluid.get_number_of_particles()))

        return [fluid,]

    def create_scheme(self):
        mpm = GasDScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            kernel_factor=kernel_factor, alpha1=alpha1, alpha2=alpha2,
            beta=beta, adaptive_h_scheme="mpm",
            update_alpha1=True, update_alpha2=True
        )
        psph = PSPHScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            hfact=kernel_factor
        )

        tsph = TSPHScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            hfact=kernel_factor
        )

        # Reconstruction does not work with this initial condition and
        # initial distribution combination.
        magma2 = MAGMA2Scheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            ndes=40, reconstruction_order=0
        )

        s = SchemeChooser(
            default='mpm', mpm=mpm, psph=psph, tsph=tsph, magma2=magma2
        )
        return s

    def configure_scheme(self):
        s = self.scheme
        s.configure_solver(
            dt=dt, tf=tf, adaptive_timestep=False, pfreq=25
        )

    def post_process(self, info_filename):
        # super().post_process(info_filename)
        self.read_info(info_filename)
        from pathlib import Path
        from pysph.solver.utils import load
        outdir = Path(self.output_dir)


        try:
            import matplotlib
            matplotlib.use("Agg")
            from matplotlib import pyplot as plt
            from exactpack.solvers.sedov.sedov import Sedov
        except ImportError:
            print("Post processing requires matplotlib and exactpack")
            return

        if self.rank > 0 or len(self.output_files) == 0:
            return

        npts = 2001
        rvec = numpy.linspace(0.0, 0.5, npts)
        solver = Sedov(geometry=2, eblast=1.0, gamma=gamma)
        solution = solver(r=rvec, t=tf)

        pa = load(self.output_files[-1])["arrays"]["fluid"]


        solution.plot('density', color='k', label='Exact')
        r = numpy.sqrt(pa.x**2 + pa.y**2)
        plt.scatter(r, pa.rho, label='Computed')

        # plt.xlim(0.0, 1.2)
        # plt.ylim(0.0, 6.5)
        plt.xlabel('Position (cm)')
        plt.ylabel('Density (g/cc)')
        plt.grid(True)
        plt.savefig(outdir.joinpath('sedov_density.png'))


if __name__ == '__main__':
    app = SedovPointExplosion()
    app.run()
