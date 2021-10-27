import numpy as np
import math


def read_eqdsk(gname, b_ccw):
    """See "Description of the GEQ files"
    and http://nstx.pppl.gov/nstx/Software/
    Applications/a-g-file-variables.txt
    https://w3.pppl.gov/ntcc/TORAY/G_EQDSK.pdf
    gname - g-file name
    b_ccw - toroidal field orientation, Depends on EFIT COCOS.  1 if you want to keep g-file orientation, -1 otherwise.
    """
    print("Reading g-file")
    out = {}
    # get individual values
    with open(gname, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                out['nr'] = int(line.split()[7])
                out['nz'] = int(line.split()[8])
            if i == 1:
                out['rdim'] = float(line.split()[0])
                out['zdim'] = float(line.split()[1])
                out['rcentr'] = float(line.split()[2])
                out['rleft'] = float(line.split()[3])
                out['zmid'] = float(line.split()[4])
            if i == 2:
                out['raxis'] = float(line.split()[0])
                out['zaxis'] = float(line.split()[1])
                out['simag'] = float(line.split()[2])
                out['sibry'] = float(line.split()[3])
                out['bcentr'] = float(line.split()[4]) * b_ccw
            if line.find('NBDRY') >= 0:
                NBDRY = int(line.split()[2])
            if line.find('LIMITR') >= 0:
                LIMITR = int(line.split()[2])
    # count nonempty lines
    nonemplin = 0
    with open(gname) as f:
        for line in f:
            if line.strip():
                nonemplin += 1
    # get psi array
    out['psi'] = np.zeros((out['nz'], out['nr']))
    temp = np.ravel(np.genfromtxt(
        gname, skip_header=57,
        skip_footer=(nonemplin - (57 + math.ceil(out['nr'] * out['nz'] / 5)))))
    # convert to psirz grid
    j = 0
    for i in range(out['nz']):
        for k in range(out['nr']):
            out['psi'][i, k] = temp[j]
            j += 1
    # generate RZ grid
    rmin = out['rleft']
    out['dr'] = out['rdim'] / (out['nr'] - 1)
    zmin = out['zmid'] - out['zdim'] / 2
    out['dz'] = out['zdim'] / (out['nz'] - 1)
    out['r'] = np.linspace(rmin, rmin+out['rdim'], out['nr'])
    out['z'] = np.linspace(zmin, zmin+out['zdim'], out['nz'])
    # read boundary
    out['rbdry'] = np.zeros(2)
    out['zbdry'] = np.zeros(2)

    with open(gname, 'r') as f:
        for i, line in enumerate(f):
            if line.find('RBDRY') >= 0:
                out['rbdry'][0] = float(line.split()[2])
                out['rbdry'][1] = float(line.split()[3])
                rbline = i
            if line.find('ZBDRY') >= 0:
                out['zbdry'][0] = float(line.split()[2])
                out['zbdry'][1] = float(line.split()[3])
                zbline = i
        temp = np.ravel(np.genfromtxt(
            gname, skip_header=(rbline + 1),
            skip_footer=(nonemplin - ((rbline + 1)
                                      + math.ceil((NBDRY - 2) / 3)))))
        temp = temp[0:(NBDRY - 2)]
        out['rbdry'] = np.insert(out['rbdry'], 2, temp)
        temp = np.ravel(np.genfromtxt(
            gname, skip_header=(zbline + 1),
            skip_footer=(nonemplin - ((zbline + 1)
                                      + math.ceil((NBDRY - 2) / 3)))))
        temp = temp[0:(NBDRY - 2)]
        out['zbdry'] = np.insert(out['zbdry'], 2, temp)
    # read wall
    out['rwall'] = np.zeros(2)
    out['zwall'] = np.zeros(2)
    with open(gname, 'r') as f:
        for i, line in enumerate(f):
            if line.find('XLIM') >= 0:
                out['rwall'][0] = float(line.split()[2])
                out['rwall'][1] = float(line.split()[3])
                rbline = i
            if line.find('YLIM') >= 0:
                out['zwall'][0] = float(line.split()[2])
                out['zwall'][1] = float(line.split()[3])
                zbline = i
        temp = np.ravel(np.genfromtxt(
            gname, skip_header=(rbline + 1),
            skip_footer=(nonemplin - ((rbline + 1)
                                      + math.ceil((LIMITR - 2) / 3)))))
        temp = temp[0:(LIMITR - 2)]
        out['rwall'] = np.insert(out['rwall'], 2, temp)
        temp = np.ravel(np.genfromtxt(
            gname, skip_header=(zbline + 1),
            skip_footer=(nonemplin - ((zbline + 1)
                                      + math.ceil((LIMITR - 2) / 3)))))
        temp = temp[0:(LIMITR - 2)]
        out['zwall'] = np.insert(out['zwall'], 2, temp)
    return out


def calc_rho(eqdsk_out, psi):
    """ Calculates rho for given psi.
    Args:
        eqdsk_out (dictionary): dictionary with eqdsk file parameters (e.g. obtained with read_eqdsk())
        psi (ndarray): psi to calculate rho at. For example eqdsk_out['psi'].

    Returns:
    out (ndarray): array of rho, corresponding to given psi.
    """
    psi_ax = eqdsk_out['simag']
    psi_bry = eqdsk_out['sibry']
    out = np.sqrt((psi-psi_ax) / (psi_bry - psi_ax))
    return out


def rho_to_psi(eqdsk_out, rho):
    """ Calculates psi from given rho.
    Args:
        eqdsk_out (dictionary): dictionary with eqdsk file parameters (e.g. obtained with read_eqdsk())
        rho (ndarray): rho to calculate psi at.

    Returns:
    out (2d ndarray): array of psi, corresponding to given rho.
    """
    psi_ax = eqdsk_out['simag']
    psi_bry = eqdsk_out['sibry']
    out = rho ** 2 * (psi_bry - psi_ax) + psi_ax
    return out


def norm_psi(eqdsk_out):
    """ Normalize psi, so psi_axis = 0.
    Args:
        eqdsk_out (dictionary): dictionary with eqdsk file parameters (e.g. obtained with read_eqdsk())

    Returns:
    eqdsk_out (dictionary): updated dictionary with normalized psi item: 2d array of rho, corresponding to g-file psi.
    """
    psi_axis = eqdsk_out['simag']
    eqdsk_out['psi'] = eqdsk_out['psi'] - psi_axis
    eqdsk_out['simag'] = eqdsk_out['simag'] - psi_axis
    eqdsk_out['sibry'] = eqdsk_out['sibry'] - psi_axis
    return eqdsk_out

