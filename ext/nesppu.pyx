# cython: language_level=3

cimport numpy as np

cdef unsigned int[64] COLORMAP = [
    0x545454,
    0x001E74,
    0x081090,
    0x300088,
    0x440064,
    0x5C0030,
    0x540400,
    0x3C1800,
    0x202A00,
    0x083A00,
    0x004000,
    0x003C00,
    0x00323C,
    0x000000,
    0x000000,
    0x000000,
    0x989698,
    0x084CC4,
    0x3032EC,
    0x5C1EE4,
    0x8814B0,
    0xA01464,
    0x982220,
    0x783C00,
    0x545A00,
    0x287200,
    0x087C00,
    0x007628,
    0x006678,
    0x000000,
    0x000000,
    0x000000,
    0xECEEEC,
    0x4C9AEC,
    0x787CEC,
    0xB062EC,
    0xE454EC,
    0xEC58B4,
    0xEC6A64,
    0xD48820,
    0xA0AA00,
    0x74C400,
    0x4CD020,
    0x38CC6C,
    0x38B4CC,
    0x3C3C3C,
    0x000000,
    0x000000,
    0xECEEEC,
    0xA8CCEC,
    0xBCBCEC,
    0xD4B2EC,
    0xECAEEC,
    0xECAED4,
    0xECB4B0,
    0xE4C490,
    0xCCD278,
    0xB4DE78,
    0xA8E290,
    0x98E2B4,
    0xA0D6E4,
    0xA0A2A0,
    0x000000,
    0x000000,
]


def get_color(index):
    return (0xFF << 24) | COLORMAP[index]


def blit(
    np.ndarray[np.uint32_t, ndim=2] source,
    np.ndarray[np.uint32_t, ndim=2] destination,
    coordinate,
):
    cdef unsigned int value
    cdef unsigned int source_height = source.shape[0]
    cdef unsigned int source_width = source.shape[1]
    cdef unsigned int destination_height = destination.shape[0]
    cdef unsigned int destination_width = destination.shape[1]
    cdef int y = coordinate[0]
    cdef int x = coordinate[1]

    cdef unsigned int i_source_start = max(0, -y)
    cdef unsigned int j_source_start = max(0, -x)
    cdef unsigned int i_source_stop = max(0, min(<int>source_height, <int>destination_height - y))
    cdef unsigned int j_source_stop = max(0, min(<int>source_width, <int>destination_width - x))
    cdef unsigned int i_source, j_source, i_destination, j_destination

    for i_source in range(i_source_start, i_source_stop):
        i_destination = i_source + y
        for j_source in range(j_source_start, j_source_stop):
            value = source[i_source, j_source]
            if value:
                j_destination = j_source + x
                destination[i_destination, j_destination] = value


def render_tile(char* rom, unsigned short address, char[3] colors, np.ndarray[np.uint32_t, ndim=2] destination):
    cdef unsigned char x, y, lower_value, higher_value, lower_bit, higher_bit, color_index, color
    cdef unsigned int rgb

    for x in range(8):
        lower_value = rom[address + x]
        higher_value = rom[address + x + 8]
        for y in range(8):
            lower_bit = (lower_value >> (7 - y)) & 0x01
            higher_bit = (higher_value >> (7 - y)) & 0x01
            color_index = lower_bit | (higher_bit << 1)
            if color_index == 0:
                destination[x, y] = 0
            else:
                color = colors[color_index - 1]
                destination[x, y] = COLORMAP[color] | <unsigned int>0xff000000