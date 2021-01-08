import h5py
import pdb
import numpy

def read_in_raw(N, g, L, m, base_dir="/rds/project/dirac_vol4/rds-dirac-dp099/cosmhol-hbor", OR=10):

    in_dir = f"{base_dir}/g{g}/su{N}/L{L}/m2-{m:.5f}/mag"
    file_prefix = f"cosmhol-hbor-su2_L{L}_g{g}._m2-{m:.5f}_or{OR}"

    M2 = numpy.fromfile(f"{in_dir}/{file_prefix}_m2.0.dat")
    M4 = numpy.fromfile(f"{in_dir}/{file_prefix}_m4.0.dat")

    pdb.set_trace()


read_in_raw(2, 1, 16, 0.24075)