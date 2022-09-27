from pysph.sph.wc.linalg import mat_mult, mat_vec_mult
from math import sqrt
from compyle.api import declare
def characteristic_decomposition(Uprimitive=[0.0, 0.0], c=0.0,
                                 invK=[0.0, 0.0],
                                 uclambda=[0.0, 0.0], K=[0.0, 0.0]):
    # Uprimitive = [rho, u, v, w, p]
    # A = K * diag(uclambda) * invK



    K[0] = 1 / (c * c)
    K[1] = 1 / (c * c)
    K[2] = 0.0
    K[3] = 0.0
    K[4] = 1.0

    K[5] = 1 / (Uprimitive[0] * c)
    K[6] = -1 / (Uprimitive[0] * c)
    K[7] = 0.0
    K[8] = 0.0
    K[9] = 0.0

    K[10] = 0.0
    K[11] = 0.0
    K[12] = 0.0
    K[13] = 1.0
    K[14] = 0.0

    K[15] = 0.0
    K[16] = 0.0
    K[17] = 1.0
    K[18] = 0.0
    K[19] = 0.0

    K[20] = 1.0
    K[21] = 1.0
    K[22] = 0.0
    K[23] = 0.0
    K[24] = 0.0

    invK[0] = 0.0
    invK[1] = 0.5 * c * Uprimitive[0]
    invK[2] = 0.0
    invK[3] = 0.0
    invK[4] = 0.5

    invK[5] = 0.0
    invK[6] = -0.5 * c * Uprimitive[0]
    invK[7] = 0.0
    invK[8] = 0.0
    invK[9] = 0.5

    invK[10] = 0
    invK[11] = 0
    invK[12] = 0
    invK[13] = 1
    invK[14] = 0

    invK[15] = 0
    invK[16] = 0
    invK[17] = 1
    invK[18] = 0
    invK[19] = 0

    invK[20] = 1
    invK[21] = 0
    invK[22] = 0
    invK[23] = 0
    invK[24] = -1 / (c * c)

    uclambda[0] = Uprimitive[1] + c
    uclambda[1] = Uprimitive[1] - c
    uclambda[2] = Uprimitive[1]
    uclambda[3] = Uprimitive[1]
    uclambda[4] = Uprimitive[1]


def rotated_characteristic_decomposition(Uprimitive=[0.0, 0.0], c=0.0,
                                         invKrot=[0.0, 0.0],
                                         uclambda=[0.0, 0.0],
                                         Krot=[0.0, 0.0], eij=[0.0, 0.0]):
    # S.J Billet and E F Toro
    # theta = elevation
    # psi = azimuth
    # T(theta, psi) = Ry(psi)*Rz(theta)

    TUprimitive = declare('matrix(5)')
    K_TU, invK_TU, T = declare('matrix(25)', 3)
    i, j, k, n = declare('int', 4)

    sin_phi = eij[2]
    cos_psi = sqrt(1 - sin_phi * sin_phi)
    sin_theta = eij[1] / cos_psi
    cos_theta = eij[0] / cos_psi

    T[0] = 1.0
    T[1] = 0.0
    T[2] = 0.0
    T[3] = 0.0
    T[4] = 0.0

    T[5] = 0.0
    T[6] = cos_psi * cos_theta
    T[7] = cos_psi * sin_theta
    T[8] = sin_phi
    T[9] = 0.0

    T[10] = 0.0
    T[11] = -sin_theta
    T[12] = cos_theta
    T[13] = 0.0
    T[14] = 0.0

    T[15] = 0.0
    T[16] = -sin_phi * cos_theta
    T[17] = -sin_phi * sin_theta
    T[18] = cos_psi
    T[19] = 0.0

    T[20] = 0.0
    T[21] = 0.0
    T[22] = 0.0
    T[23] = 0.0
    T[24] = 1.0

    mat_vec_mult(T, Uprimitive, 5, TUprimitive)
    characteristic_decomposition(TUprimitive, c, invK_TU, uclambda, K_TU)

    n = 5
    for i in range(n):
        for k in range(n):
            s = 0.0
            for j in range(n):
                s += T[n * j + i] * K_TU[n * j + k]
            Krot[n*i + k] = s
    mat_mult(invK_TU, T, 5, invKrot)

if __name__ == '__main__':
    import numpy as np
    Q = np.arange(25) + 1
    Q2 = Q[10:15]
    gamma = 1.4
    eij = np.random.random(3)
    eij = eij / np.linalg.norm(eij)
    invKrot, Krot = declare('matrix(25)', 2)
    uclambda = declare('matrix(5)')
    ci = gamma * Q2[4] / Q2[0]
    rotated_characteristic_decomposition(Q2, ci, invKrot, uclambda, Krot, eij)