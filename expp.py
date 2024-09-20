import numpy as np
import torch

def decompose(vect):
    if (vect.dtype == torch.bfloat16):
        vect = vect.float()
    
    if (vect.dtype == torch.float32):
        vect = vect.numpy()

    vect_int = np.frombuffer(vect.tobytes(), np.uint32)

    sign = vect_int >> 31
    mant = np.bitwise_and(vect_int >> 16, 0b000000001111111)
    exp = np.bitwise_and(vect_int >> 23, 0b011111111)
    
    return sign, mant, exp

def P(vect, input_fraction = 7, coefficient_fraction = 5, constant_fraction = 6, alpha1 = 0.21875, beta1 = 0.4375, gamma1 =3.296875, gamma2 = 2.171875, mul_surplus_bits = 1, not_surplus_bits = 0):
    alpha = np.round(alpha1 * 2 ** coefficient_fraction).astype(np.uint32)
    beta = np.round(beta1 * 2 ** coefficient_fraction).astype(np.uint32)

    sum_fraction = max(input_fraction, constant_fraction)

    gamma_1 = np.round(gamma1 * 2 ** constant_fraction).astype(np.int32) * 2 ** (sum_fraction - constant_fraction)
    gamma_2 = np.round(gamma2 * 2 ** constant_fraction).astype(np.int32) * 2 ** (sum_fraction - constant_fraction)

    mant_add = np.bitwise_and(np.frombuffer(vect.tobytes(), dtype = np.uint32) >> 16, 0x007F).astype(np.int32) * 2 ** (sum_fraction - 7)
    res_add_1 = np.where(mant_add < 2 ** (sum_fraction - 1), mant_add + gamma_1 , mant_add + gamma_2)
    
    mant_mul = np.bitwise_and(np.frombuffer(vect.tobytes(), dtype = np.uint32) >> 16, 0x007F).astype(np.int32) * 2 ** (mul_surplus_bits)
    res_mul_1 = np.where(mant_mul < 2 ** (mul_surplus_bits + input_fraction - 1), mant_mul * alpha, (beta * (2 ** (mul_surplus_bits + input_fraction) - mant_mul - 1)))

    res_mul_2 = (res_mul_1 * res_add_1) >> (sum_fraction + coefficient_fraction + mul_surplus_bits - not_surplus_bits)

    res = np.where(mant_add < 2 ** (sum_fraction - 1), res_mul_2, 2 ** (7 + not_surplus_bits) - res_mul_2 - 1) >> not_surplus_bits

    return np.frombuffer((np.bitwise_and(np.frombuffer(vect.tobytes(), dtype = np.uint32), 0xFF800000) + (res << 16)).astype(np.int32), dtype = np.float32)

def exps(vect, input_fraction = 7, a_fraction = 14, coefficient_fraction = 7, constant_fraction = 8):
    a = np.round(1 / np.log(2) * 2 ** a_fraction).astype(np.int32)

    sign, mant, exp = decompose(vect)

    #Add the implicit leading one
    mant = 2 ** 7 + mant

    mant_comp = np.where(sign == 1, (mant.astype(np.int32)), mant.astype(np.int32))

    shm = np.where(exp >= 127, (mant_comp * a) << (exp - 127), (mant_comp * a) >> (127 - exp))
    shm = (shm >> a_fraction) + np.bitwise_and(shm >> (a_fraction - 1), 0b1)
    shm = np.where(sign == 1, -shm, shm)
    
    nm = np.bitwise_and(shm, 0x007F)
    ne = (shm >> 7) + 127

    quant_score = (((ne << 7) + nm) << 16)
    
    exp_a = np.frombuffer(quant_score.astype(np.uint32).tobytes(), np.float32())
    exp_a = np.where(ne.astype(np.uint32) >= 255, 0, exp_c)
    
    return exp_a

def expp(vect):
    exp_schraudolph = exps(vect)
    return P(exp_schraudolph)
