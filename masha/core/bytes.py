def nearest_bytes(n):
    p = int(float(n).hex().split('p+')[1]) + 1
    return 2 ** p


def round8(a):
    return int(a) + 4 & ~7


def round2(a):
    return int(a) + int(int(a) % 2)
