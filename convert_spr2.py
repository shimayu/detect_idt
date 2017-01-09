import sys
import numpy as np
from PIL import Image

Byte_size = 8

# the number of handlers
Addr_size = 256
Separate_size = Addr_size / 2
Wide_size = 16
Raw_size = 2

def hex2int(byte_data):
    byte_data_ord = []
    byte_addr = []
        
    for i in xrange(Byte_size * Addr_size):
        byte_data_ord.append(ord(byte_data[i]))
        byte_data_ord[i] = byte_data_ord[i] << (i % 8) * 8

    for i in xrange(Byte_size * Addr_size):
        if i % 8 == 0:
            tmp = byte_data_ord[i]
            for bit in xrange(Byte_size):
                tmp = tmp | byte_data_ord[i+bit]
            byte_addr.append(tmp)
        else:
            continue

    return byte_addr


def create_bw(data_arr):
    bitmap = []
    for i in xrange(Raw_size):
        for bit in xrange((Byte_size / 2) * Wide_size):
            if (data_arr[i] >> bit) & 1 == 1:
                # black(0)
                color = 0
            else:
                # white(255)
                color = 255
            bitmap.append([color, color, color])
    return bitmap


if __name__ == '__main__':
    argvs = sys.argv

    f = open(argvs[1], "rb")
    data = f.read()
    addr = []
    addr_up = []
    addr_down = []
    addr = hex2int(data)

    for i in xrange(Separate_size):
        addr_up.append(addr[i])

    for i in range(Separate_size, Addr_size):
        addr_down.append(addr[i])
       
    bw_arr = []
    byte_arr = []

    for x in addr_down:
        if len(byte_arr) < Raw_size:
            byte_arr.append(x)
        else:
            bw_arr.append(create_bw(byte_arr))
            byte_arr = [x]

    # for i in xrange(Addr_size):
    #     print(int(addr[i]))
        # print(bw_arr[i])
            
    ip = Image.fromarray(np.uint8(np.array(bw_arr)))
    ip.save(argvs[2])

