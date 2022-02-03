import matplotlib.pyplot as plt




t = [0]
s = [10]
p = [0]
se = [0]
e = [1]
h = 0.0000001
N = 100000




def Fs(s, p, se, e):
    return 600*se-100*e*s


def Fp(s, p, se, e):
    return 150*se


def Fse(s, p, se, e):
    return 100*e*s-600*se-150*se


def Fe(s, p, se, e):
    return -100*e*s+600*se+150*se


def main():
    for i in range(N):
        K1 = Fs(s[-1], p[-1], se[-1], e[-1])
        L1 = Fp(s[-1], p[-1], se[-1], e[-1])
        M1 = Fse(s[-1], p[-1], se[-1], e[-1])
        N1 = Fe(s[-1], p[-1], se[-1], e[-1])
        K2 = Fs(s[-1] + h * K1 / 2, p[-1] + h * L1 / 2, se[-1] + h * M1 / 2, e[-1] + h * N1 / 2)
        L2 = Fp(s[-1] + h * K1 / 2, p[-1] + h * L1 / 2, se[-1] + h * M1 / 2, e[-1] + h * N1 / 2)
        M2 = Fse(s[-1] + h * K1 / 2, p[-1] + h * L1 / 2, se[-1] + h * M1 / 2, e[-1] + h * N1 / 2)
        N2 = Fe(s[-1] + h * K1 / 2, p[-1] + h * L1 / 2, se[-1] + h * M1 / 2, e[-1] + h * N1 / 2)
        K3 = Fs(s[-1] + h * K2 / 2, p[-1] + h * L2 / 2, se[-1] + h * M2 / 2, e[-1] + h * N2 / 2)
        L3 = Fp(s[-1] + h * K2 / 2, p[-1] + h * L2 / 2, se[-1] + h * M2 / 2, e[-1] + h * N2 / 2)
        M3 = Fse(s[-1] + h * K2 / 2, p[-1] + h * L2 / 2, se[-1] + h * M2 / 2, e[-1] + h * N2 / 2)
        N3 = Fe(s[-1] + h * K2 / 2, p[-1] + h * L2 / 2, se[-1] + h * M2 / 2, e[-1] + h * N2 / 2)
        K4 = Fs(s[-1] + h * K3 / 2, p[-1] + h * L3 / 2, se[-1] + h * M3 / 2, e[-1] + h * N3 / 2)
        L4 = Fp(s[-1] + h * K3 / 2, p[-1] + h * L3 / 2, se[-1] + h * M3 / 2, e[-1] + h * N3 / 2)
        M4 = Fse(s[-1] + h * K3 / 2, p[-1] + h * L3 / 2, se[-1] + h * M3 / 2, e[-1] + h * N3 / 2)
        N4 = Fe(s[-1] + h * K3 / 2, p[-1] + h * L3 / 2, se[-1] + h * M3 / 2, e[-1] + h * N3 / 2)
        s.append(s[-1] + h / 6 * (K1 + 2 * K2 + 2 * K3 + K4))
        p.append(p[-1] + h / 6 * (L1 + 2 * L2 + 2 * L3 + L4))
        se.append(se[-1] + h / 6 * (M1 + 2 * M2 + 2 * M3 + M4))
        e.append(e[-1] + h / 6 * (N1 + 2 * N2 + 2 * N3 + N4))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plt.plot([150* i for i in se], s,)
    #ax.plot(p, s, e,)

    
    plt.show()




main()
