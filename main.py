from calibration import calibration
from live import live

if __name__ == '__main__':
    mean, std = calibration()
    print(mean, std)


    data, result = live(mean, std)

    for D, r in zip(data, result):
        print(D)
        print(r)
        print('\n')