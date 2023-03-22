from math import sqrt, cos, sin
from cmath import exp as cexp
import numpy as np


PI = 3.141592653589793238462643383279502884
C_SI = 299792458.
L_SI = 2.5e9
AU_SI = 149597870700.0
Omega0 = 1.9909865927683788e-07
MAX_MODES = 6
eorbit = 0.004824185218078991
SQRT2 = 1.414213562373095048801688724209698079
SQRT3 = 1.732050807568877293527446341505872367
SQRT6 = 2.449489742783178098197284074705891392
INVSQRT2 = 0.707106781186547524400844362104849039
INVSQRT3 = 0.577350269189625764509148780501957455
INVSQRT6 = 0.408248290463863016366214012450981898


class Namespace:
    pass


def d_sinc(x:float):
    if x == 0.:
        return 1
    else:
        return sin(x) / x

def dot_product_2d(
    out, # double np array
    arr1, # double np array
    m1: int,
    n1: int,
    arr2, # double np array
    m2: int,
    n2: int
):
    for i in range(m1):
        for j in range(n2):
            out[(i*3 + j)] = 0.
            for k in range(n1):
                out[i * 3  + j] += arr1[i * 3 + k]*arr2[k * 3 + j]


def d_vec_H_vec_product(arr1, H, arr2):
    return arr1.dot(H.dot(arr2))


def SpinWeightedSphericalHarmonic(
    s: int,
    l: int,
    m: int,
    theta: float,
    phi: float
):
    if l == 2 and m == -2:
        fac = sqrt( 5.0 / ( 64.0 * PI ) ) * ( 1.0 - cos( theta ))*( 1.0 - cos( theta ))
    elif (l == 2) and m == -1:
        fac = sqrt( 5.0 / ( 16.0 * PI ) ) * sin( theta )*( 1.0 - cos( theta ))
    elif (l == 2) and (m == 1):
        fac =  sqrt( 5.0 / ( 16.0 * PI ) ) * sin( theta )*( 1.0 + cos( theta ))
    elif (l == 2) and (m == 2):
        fac = sqrt( 5.0 / ( 64.0 * PI ) ) * ( 1.0 + cos( theta ))*( 1.0 + cos( theta ))
    else:
        raise ValueError(f"Not yet added l={l} and m={m}")

    return fac * cexp(1j * m * phi)


def d_TDICombinationFD(
    Gslr: Namespace,
    f: float,
    TDItag: int,
    rescaled: int
):
    transferL = Namespace()
    x = PI*f*L_SI/C_SI
    z = cexp(1j*2.*x)
    if TDItag==1:
        raise ValueError("Ian didn't bother to include X,Y,Z TDI here!")
    else:
        # Should give AET from 1st gen X,Y,Z
        factor_convention = complex(2, 0)
        if rescaled == 1:
            factorAE = complex(1., 0)
            factorT = complex(1., 0)
        else:
            factorAE = 1j*SQRT2*sin(2.*x)*z
            factorT = 2.*SQRT2*sin(2.*x)*sin(x)*cexp(1j*3*x)
        Araw = 0.5 * ( (1.+z)*(Gslr.G31 + Gslr.G13) - Gslr.G23 - z*Gslr.G32 - Gslr.G21 - z*Gslr.G12 )
        Eraw = 0.5*INVSQRT3 * ( (1.-z)*(Gslr.G13 - Gslr.G31) + (2.+z)*(Gslr.G12 - Gslr.G32) + (1.+2.*z)*(Gslr.G21 - Gslr.G23) )
        Traw = INVSQRT6 * ( Gslr.G21 - Gslr.G12 + Gslr.G32 - Gslr.G23 + Gslr.G13 - Gslr.G31)
        transferL.transferL1 = factor_convention * factorAE * Araw
        transferL.transferL2 = factor_convention * factorAE * Eraw
        transferL.transferL3 = factor_convention * factorT * Traw
        return transferL
        


def d_EvaluateGslr(
    t: float,
    f: float,
    H, # complex np array
    k, # float np array
    response: int,
    p0, # float "array" (actually a list right now)
):
    alpha = Omega0*t
    c = cos(alpha)
    s = sin(alpha)
    a = AU_SI
    e = eorbit
    
    p1L = [- a*e*(1 + s*s), a*e*c*s, -a*e*SQRT3*c]
    p2L = [
        a*e/2*(SQRT3*c*s + (1 + s*s)),
        a*e/2*(-c*s - SQRT3*(1 + c*c)),
        -a*e*SQRT3/2*(SQRT3*s - c)
    ]
    p3L = [
        a*e/2*(-SQRT3*c*s + (1 + s*s)),
        a*e/2*(-c*s + SQRT3*(1 + c*c)),
        -a*e*SQRT3/2*(-SQRT3*s - c)
    ]
    n = np.array([-1./2*c*s, 1./2*(1 + c*c), SQRT3/2*s], dtype=float)

    kn1 = k[0]*n[0] + k[1]*n[1] + k[2]*n[2]
    n1Hn1 = d_vec_H_vec_product(n, H, n)
    
    n = np.array([
        (c*s - SQRT3*(1 + s*s))*0.25,
        (SQRT3*c*s - (1 + c*c))*0.25,
        ( -SQRT3*s - 3*c)*0.25
    ])

    kn2 = k[0]*n[0] + k[1]*n[1] + k[2]*n[2]
    n2Hn2 = d_vec_H_vec_product(n, H, n)

    n = np.array([
        (c*s + SQRT3*(1 + s*s))*0.25,
        (-SQRT3*c*s - (1 + c*c))*0.25,
        (-SQRT3*s + 3*c)*0.25
    ])
    kn3 = k[0]*n[0] + k[1]*n[1] + k[2]*n[2]
    n3Hn3 = d_vec_H_vec_product(n, H, n)

    temp1 = p1L[0]+p2L[0]
    temp2 = p1L[1]+p2L[1]
    temp3 = p1L[2]+p2L[2]
    temp4 = p2L[0]+p3L[0]
    temp5 = p2L[1]+p3L[1]
    temp6 = p2L[2]+p3L[2]
    temp7 = p3L[0]+p1L[0]
    temp8 = p3L[1]+p1L[1]
    temp9 = p3L[2]+p1L[2]

    p1L = [temp1, temp2, temp3]
    p2L = [temp4, temp5, temp6]
    p3L = [temp7, temp8, temp9]

    kp1Lp2L = k[0]*p1L[0] + k[1]*p1L[1] + k[2]*p1L[2]
    kp2Lp3L = k[0]*p2L[0] + k[1]*p2L[1] + k[2]*p2L[2]
    kp3Lp1L = k[0]*p3L[0] + k[1]*p3L[1] + k[2]*p3L[2]

    kp0 = k[0]*p0[0] + k[1]*p0[1] + k[2]*p0[2]
    
    if response == 1:
        factorcexp0 = cexp(1j*2.*PI*f/C_SI * kp0)
    else:
        factorcexp0=complex(1.,0.)

    prefactor = PI*f*L_SI/C_SI

    factorcexp12 = cexp(1j*prefactor * (1.+kp1Lp2L/L_SI))
    factorcexp23 = cexp(1j*prefactor * (1.+kp2Lp3L/L_SI))
    factorcexp31 = cexp(1j*prefactor * (1.+kp3Lp1L/L_SI))

    factorsinc12 = d_sinc(prefactor * (1.-kn3))
    factorsinc21 = d_sinc(prefactor * (1.+kn3))
    factorsinc23 = d_sinc(prefactor * (1.-kn1))
    factorsinc32 = d_sinc(prefactor * (1.+kn1))
    factorsinc31 = d_sinc(prefactor * (1.-kn2))
    factorsinc13 = d_sinc(prefactor * (1.+kn2))

    Gslr_out = Namespace()

    commonfac = 1j*prefactor*factorcexp0
    Gslr_out.G12 = commonfac * n3Hn3 * factorsinc12 * factorcexp12
    Gslr_out.G21 = commonfac * n3Hn3 * factorsinc21 * factorcexp12
    Gslr_out.G23 = commonfac * n1Hn1 * factorsinc23 * factorcexp23
    Gslr_out.G32 = commonfac * n1Hn1 * factorsinc32 * factorcexp23
    Gslr_out.G31 = commonfac * n2Hn2 * factorsinc31 * factorcexp31
    Gslr_out.G13 = commonfac * n2Hn2 * factorsinc13 * factorcexp31

    return Gslr_out


def d_JustLISAFDresponseTDI(
    H, # complex np array
    f: float,
    t: float,
    lam: float,
    beta: float,
    TDItag: int,
    order_fresnel_stencil: int
):
    kvec = [-cos(beta)*cos(lam), -cos(beta)*sin(lam), -sin(beta)]
    alpha = Omega0*t
    c = cos(alpha)
    s = sin(alpha)
    a = AU_SI
    p0 = [a*c, a*s, 0.]

    kR = kvec[0] * p0[0] + kvec[1] * p0[1] + kvec[2] * p0[2]
    phaseRdelay = 2.*PI/C_SI *f*kR

    Gslr = d_EvaluateGslr(t, f, H, kvec, 1, p0)

    Tslr = Namespace()
    Tslr.G12 = Gslr.G12 * cexp(-1j * phaseRdelay)
    Tslr.G21 = Gslr.G21 * cexp(-1j * phaseRdelay)
    Tslr.G23 = Gslr.G23 * cexp(-1j * phaseRdelay)
    Tslr.G32 = Gslr.G32 * cexp(-1j * phaseRdelay)
    Tslr.G31 = Gslr.G31 * cexp(-1j * phaseRdelay)
    Tslr.G13 = Gslr.G13 * cexp(-1j * phaseRdelay)

    # FIXME: Add this one too!
    transferL = d_TDICombinationFD(Tslr, f, TDItag, 0)

    transferL.phaseRdelay = phaseRdelay
    return transferL



def response_modes(
    phases, # double np array
    response_out, # double np array
    binNum: int,
    mode_i: int,
    tf, # double np array
    freqs, # double np array
    phi_ref: float,
    ell : int,
    mm : int,
    length: int,
    numBinAll: int,
    numModes: int,
    H, # complex np array
    lam: float,
    beta: float,
    TDItag: int,
    order_fresnel_stencil: int
):
    eps = 1E-9

    for i in range(length):
        mode_index = (binNum * numModes + mode_i) * length + i
        freq_index = binNum * length + i
        freq = freqs[freq_index]
        t_wave_frame = tf[mode_index]
        transferL = d_JustLISAFDresponseTDI(H, freq, t_wave_frame, lam, beta, TDItag, order_fresnel_stencil)
        response_out[mode_index] = transferL.transferL1.real
        idx = numBinAll * numModes * length + mode_index
        response_out[idx] = transferL.transferL1.imag
        idx = 2 * numBinAll * numModes * length + mode_index
        response_out[idx] = transferL.transferL2.real
        idx = 3 * numBinAll * numModes * length + mode_index
        response_out[idx] = transferL.transferL2.imag
        idx = 4 * numBinAll * numModes * length + mode_index
        response_out[idx] = transferL.transferL3.real
        idx = 5 * numBinAll * numModes * length + mode_index
        response_out[idx] = transferL.transferL3.imag
        phase_change = transferL.phaseRdelay
        phases[mode_index] += phase_change


def responseCore(
    phases, # double np array
    response_out, # double np array
    ells, # int np array
    mms, # int np array
    tf, # double np array
    freqs, # double np array
    phi_ref: float,
    inc: float,
    lam: float,
    beta: float,
    psi: float,
    length: int,
    numModes: int,
    binNum: int,
    numBinAll: int,
    TDItag: int,
    order_fresnel_stencil: int,
):
    HSplus = np.array([1,0,0,-1,0,0,0,0,0], dtype=float)
    HScross = np.array([0,1,0,1,0,0,0,0,0], dtype=float)

    kvec = np.array([-cos(beta)*cos(lam), -cos(beta)*sin(lam), -sin(beta)])
    clambd = cos(lam)
    slambd = sin(lam)
    cbeta = cos(beta)
    sbeta = sin(beta)
    cpsi = cos(psi)
    spsi = sin(psi)

    O1 = np.array([
        cpsi*slambd-clambd*sbeta*spsi,
        -clambd*cpsi*sbeta-slambd*spsi,
        -cbeta*clambd,
        -clambd*cpsi-sbeta*slambd*spsi,
        -cpsi*sbeta*slambd+clambd*spsi,
        -cbeta*slambd,
        cbeta*spsi,
        cbeta*cpsi,
        -sbeta
    ], dtype=float)

    invO1 = np.array([
        cpsi*slambd-clambd*sbeta*spsi,
        -clambd*cpsi-sbeta*slambd*spsi,
        cbeta*spsi,
        -clambd*cpsi*sbeta-slambd*spsi,
        -cpsi*sbeta*slambd+clambd*spsi,
        cbeta*cpsi,
        -cbeta*clambd,
        -cbeta*slambd,
        -sbeta
    ], dtype=float)

    out1 = np.array([0,0,0,0,0,0,0,0,0], dtype=float)
    H_mat = np.array([[0,0,0],[0,0,0],[0,0,0]], dtype=complex)
    Hplus = np.array([0,0,0,0,0,0,0,0,0], dtype=float)
    Hcross = np.array([0,0,0,0,0,0,0,0,0], dtype=float)

    dot_product_2d(out1, HSplus, 3, 3, invO1, 3, 3)
    dot_product_2d(Hplus, O1, 3, 3, out1, 3, 3)
    dot_product_2d(out1, HScross, 3, 3, invO1, 3, 3)
    dot_product_2d(Hcross, O1, 3, 3, out1, 3, 3)

    for mode_i in range(numModes):
        ell = ells[mode_i]
        mm = mms[mode_i]

        Ylm = SpinWeightedSphericalHarmonic(-2, ell, mm, inc, phi_ref)
        Yl_m = (-1)**ell * np.conj(SpinWeightedSphericalHarmonic(-2, ell, -1*mm, inc, phi_ref))
        Yfactorplus = 1./2 * (Ylm + Yl_m)
        Yfactorcross = 1./2. * 1j * (Ylm - Yl_m)

        for i in range(3):
            for j in range(3):
                trans1 = Hplus[i*3 + j]
                trans2 = Hcross[i*3 + j]
                H_mat[i,j] = (Yfactorplus*trans1+ Yfactorcross*trans2)

        response_modes(
            phases,
            response_out,
            binNum,
            mode_i,
            tf,
            freqs,
            phi_ref,
            ell,
            mm,
            length,
            numBinAll,
            numModes,
            H_mat,
            lam,
            beta,
            TDItag,
            order_fresnel_stencil
        )
    

def response(
    phases, # double np array
    response_out, # double np array
    tf, # double np array
    ells_in, # int np array
    mms_in, # int np array
    freqs, # double np array
    phi_ref, # double np array
    inc, # double np array
    lam, # double np array
    beta, # double np array
    psi, # double np array
    TDItag: int,
    order_fresnel_stencil: int,
    numModes: int,
    length: int,
    numBinAll: int
):
    ells = np.array([MAX_MODES], dtype=int)
    mms = np.array([MAX_MODES], dtype=int)
    start = 0
    increment = 1

    for i in range(start, numModes):
        ells[i] = ells_in[i]
        mms[i] = mms_in[i]
    
    for binNum in range(start, numBinAll):
        responseCore(
            phases,
            response_out,
            ells,
            mms,
            tf,
            freqs,
            phi_ref[binNum],
            inc[binNum],
            lam[binNum],
            beta[binNum],
            psi[binNum],
            length,
            numModes,
            binNum,
            numBinAll,
            TDItag,
            order_fresnel_stencil
        )


CACHED_FREQS = [0]
CACHED_OBJECTS = [None]
def LISA_response(
    response_out, # double np array
    ells_in, # int np array
    mms_in, # int np array
    freqs, # double np array
    phi_ref, # double np array
    inc, # double np array
    lam, # double np array
    beta, # double np array
    psi, # double np array
    TDItag: int,
    order_fresnel_stencil : int,
    numModes: int,
    length: int,
    numBinAll: int,
    includesAmps: int
):
    start_param = includesAmps
    #if numBinAll*numModes*length == CACHED_FREQS[0]:
    #    response_out[(start_param+2)*numBinAll*numModes*length:] = CACHED_OBJECTS[0][:]
    #    return

    #CACHED_FREQS[0] = numBinAll*numModes*length

    phases = response_out[start_param*numBinAll*numModes*length:
                          (start_param+1)*numBinAll*numModes*length]
    tf = response_out[(start_param+1)*numBinAll*numModes*length:
                      (start_param+2)*numBinAll*numModes*length]
    #tf = tf.astype(int) // 100
    response_vals = response_out[(start_param+2)*numBinAll*numModes*length:]

    response(
        phases,
        response_vals,
        tf,
        ells_in,
        mms_in,
        freqs, 
        phi_ref,
        inc,
        lam,
        beta,
        psi,
        TDItag,
        order_fresnel_stencil,
        numModes,
        length,
        numBinAll
    )
    #CACHED_OBJECTS[0] = response_out[(start_param+2)*numBinAll*numModes*length:]
