import numpy as np; 

def kfilter(y, num, x0, v0, A, C, Q, R):
    x_f = np.zeros(num); v_f = np.zeros(num);
    x_p = np.zeros(num); v_p = np.zeros(num);
    x_f[-1] = x0; v_f[-1] = v0;
    for i in range(num):
        x_p[i] = A*x_f[i-1]
        v_p[i] = A*v_f[i-1]*A + Q
        K = v_p[i]*C/(C*v_p[i]*C+R)
        x_f[i] = x_p[i] + K*(y[i]-C*x_p[i])
        v_f[i] = v_p[i] - K*C*v_p[i]
    return {'x_f':x_f,'v_f':v_f,'x_p':x_p,'v_p':v_p, 'K_T':K}

def ksmooth(y, num, x_f, v_f, x_p, v_p, A, vvT):
    x_s = np.zeros(num); v_s = np.zeros(num); J = np.zeros(num-1); P = np.zeros(num)
    x_s[num-1] = x_f[num-1]; v_s[num-1] = v_f[num-1]
    P[num-1] =  v_s[num-1] + x_s[num-1]*x_s[num-1]
    for i in range(1,num)[::-1]:
        J[i-1] = v_f[i-1]*A/v_p[i]
        x_s[i-1] = x_f[i-1] + J[i-1]*(x_s[i]-x_p[i])
        v_s[i-1] = v_f[i-1] + J[i-1]*(v_s[i]-v_p[i])*J[i-1]
        P[i-1] =  v_s[i-1] + x_s[i-1]*x_s[i-1]
    PP = np.zeros(num-1); vv = np.zeros(num-1)
    vv[num-2] = vvT; PP[num-2] = vv[num-2] + x_s[num-1]*x_s[num-2]
    for i in range(2,num)[::-1]:
        vv[i-2] = v_f[i-1]*J[i-2] + J[i-1]*(vv[i-1]-A*v_f[i-1])*J[i-2]
        PP[i-2] = vv[i-2] + x_s[i-1]*x_s[i-2]
    return {'x_s':x_s,'v_s':v_s,'J':J,'P':P,'PP':PP}

def klearn(y, num, x0, v0, A, C, Q, R):
    while True:
        flt = kfilter(y, num, x0, v0, A, C, Q, R)
        x_f = flt['x_f']; v_f = flt['v_f']
        x_p = flt['x_p']; v_p = flt['v_p']; K_T = flt['K_T']
        sm = ksmooth(y, num, x_f, v_f, x_p, v_p, A, (1-K_T*C)*A*v_f[num-2])
        x_s = sm['x_s']; P = sm['P']; PP = sm['PP']
        C = np.sum(y*x_s) / sum(P)
        R = np.sum(y*y-C*x_s*y) / num
        A = np.sum(PP) / np.sum(P[:-1])
        x0 = x_s[0]
        v0 = P[0] - x_s[0]*x_s[0]
        yield x0, v0, A, C, Q, R
