# cython: language_level=3

def set_cpu_attributes(cpu, pc, a, x, y, sp, n, z, c, v, ic):
    cpu.pc = pc
    cpu.a = a
    cpu.x = x
    cpu.y = y
    cpu.sp = sp
    cpu.n = n
    cpu.z = z
    cpu.c = c
    cpu.v = v
    cpu.instruction_count = ic


def run(cpu):
    cdef unsigned char* rom = cpu.rom
    cdef unsigned char* ram = cpu.ram
    cdef unsigned short pc = cpu.pc
    cdef unsigned char a = cpu.a
    cdef unsigned char x = cpu.x
    cdef unsigned char y = cpu.y
    cdef unsigned char sp = cpu.sp
    cdef unsigned char n = cpu.n
    cdef unsigned char z = cpu.z
    cdef unsigned char c = cpu.c
    cdef unsigned char v = cpu.v
    cdef unsigned int ic = cpu.instruction_count

    cdef unsigned char opc
    cdef unsigned char addressing
    cdef unsigned char operand
    cdef unsigned short address
    cdef unsigned char value

    # Instruction loop
    while 1:
        # Read opcode
        ic += 1
        opc = rom[pc - 0x8000]
        pc += 1

        # No operand

        # Increment/decrement

        # INX
        if opc == 0xe8:
            x += 1
            n = x >> 7
            z = x == 0
            continue
        # DEX
        elif opc == 0xca:
            x -= 1
            n = x >> 7
            z = x == 0
            continue
        # INY
        elif opc == 0xc8:
            y += 1
            n = y >> 7
            z = y == 0
            continue
        # DEY
        elif opc == 0x88:
            y -= 1
            n = y >> 7
            z = y == 0
            continue

        # Shifting

        # ASL ACC
        elif opc == 0x0a:
            c = a >> 7
            a <<= 1
            n = a >> 7
            z = a == 0
            continue
        # LSR ACC
        elif opc == 0x4a:
            c = a & 0x01
            a >>= 1
            n = a >> 7
            z = a == 0
            continue
        # ROL ACC
        elif opc == 0x2a:
            value = c
            c = a >> 7
            a <<= 1
            a |= value
            n = a >> 7
            z = a == 0
            continue
        # ROR ACC
        elif opc == 0x6a:
            value = c
            c = a & 0x01
            a >>= 1
            a |= value << 7
            n = a >> 7
            z = a == 0
            continue

        # Flow control

        # NOP
        elif opc == 0xea:
            continue
        # RTI/BRK
        elif opc in (0x00, 0x40):
            break
        # RTS
        elif opc == 0x60:
            sp += 1
            pc = ram[0x0100 | sp]
            sp += 1
            pc |= ram[0x0100 | sp] << 8
            pc += 1
            continue
        # PHA
        elif opc == 0x48:
            ram[0x0100 | sp] = a
            sp -= 1
            continue
        # PLA
        elif opc == 0x68:
            sp += 1
            a = ram[0x0100 | sp]
            n = a >> 7
            z = a == 0  # Comment here for the nicest bug
            continue
        # PHP
        elif opc == 0x08:
            value = (n << 7) | (v << 6) | (z << 1) | (c << 0)
            ram[0x0100 | sp] = value
            sp -= 1
            continue
        # PLP
        elif opc == 0x28:
            sp += 1
            value = ram[0x0100 | sp]
            n = (value & 0x80) != 0
            v = (value & 0x40) != 0
            z = (value & 0x02) != 0
            c = (value & 0x01) != 0
            continue

        # Flags

        # CLC
        elif opc == 0x18:
            c = 0
            continue
        # SEC
        elif opc == 0x38:
            c = 1
            continue
        # CLI/SEI/CLV/CLD/SED
        elif opc in (0x58, 0x78, 0xB8, 0xD8, 0xF8):
            continue
        # TAX
        elif opc == 0xaa:
            x = a
            n = a >> 7
            z = a == 0
            continue

        # Transfer

        # TXA
        elif opc == 0x8a:
            a = x
            n = a >> 7
            z = a == 0
            continue
        # TAY
        elif opc == 0xa8:
            y = a
            n = a >> 7
            z = a == 0
            continue
        # TYA
        elif opc == 0x98:
            a = y
            n = a >> 7
            z = a == 0
            continue
        # TXS
        elif opc == 0x9a:
            sp = x
            continue
        # TSX
        elif opc == 0xba:
            x = sp
            n = x >> 7
            z = x == 0
            continue

        # Byte operand
        operand = rom[pc - 0x8000]
        pc += 1

        addressing = (opc & 0b00011100) >> 2
        # ZPY
        if opc in (0x96, 0xb6):
            operand += y
            address = operand
            value = ram[address]
        # ZPX
        elif addressing == 0x05:
            operand += x
            address = operand
            value = ram[address]
        # ZPG
        elif addressing == 0x01:
            address = operand
            value = ram[address]
        # INX
        elif (opc & 0x0f) == 0x01 and addressing == 0x00:
            operand += x
            address = ram[operand]
            operand += 1
            address |= ram[operand] << 8
            if opc == 0x81:
                value = 0
            elif address < 0x800:
                value = ram[address]
            elif address > 0x8000:
                value = rom[address - 0x8000]
            else:
                set_cpu_attributes(cpu, pc, a, x, y, sp, n, z, c, v, ic)
                value = cpu.cpu_read(address)
        # INY
        elif (opc & 0x0f) == 0x01 and addressing == 0x04:
            address = ram[operand]
            operand += 1
            address |= ram[operand] << 8
            address += y
            if opc == 0x91:
                value = 0
            elif address < 0x800:
                value = ram[address]
            elif address > 0x8000:
                value = rom[address - 0x8000]
            else:
                set_cpu_attributes(cpu, pc, a, x, y, sp, n, z, c, v, ic)
                value = cpu.cpu_read(address)
        # IMM / REL
        else:
            address = 0
            value = operand

        # Branching

        # BPL
        if opc == 0x10:
            if not n:
                pc += <char>value
            continue
        # BMI
        elif opc == 0x30:
            if n:
                pc += <char>value
            continue
        # BNE
        elif opc == 0xd0:
            if not z:
                pc += <char>value
            continue
        # BEQ
        elif opc == 0xf0:
            if z:
                pc += <char>value
            continue
        # BCC
        elif opc == 0x90:
            if not c:
                pc += <char>value
            continue
        # BCS
        elif opc == 0xB0:
            if c:
                pc += <char>value
            continue
        # BVC
        elif opc == 0x50:
            if not v:
                pc += <char>value
            continue
        # BVS
        elif opc == 0x70:
            if v:
                pc += <char>value
            continue

        # Store

        # STA ZPG/ZPX
        elif opc in (0x85, 0x95):
            ram[address] = a
            continue
        # STX ZPG
        elif opc in (0x86, 0x96):
            ram[address] = x
            continue
        # STY ZPG
        elif opc in (0x84, 0x94):
            ram[address] = y
            continue
        # STA INX/INY
        elif opc in (0x81, 0x91):
            if address < 0x800:
                ram[address] = a
            else:
                set_cpu_attributes(cpu, pc, a, x, y, sp, n, z, c, v, ic)
                cpu.cpu_write(address, a)
            continue

        # Load

        # LDA IMM/ZPG/ZPX/INX/INY
        elif opc in (0xa9, 0xa5, 0xb5, 0xa1, 0xb1):
            a = value
            n = (a & 0x80) != 0
            z = a == 0
            continue
        # LDX IMM/ZPG/ZPY
        elif opc in (0xa2, 0xa6, 0xb6):
            x = value
            n = (x & 0x80) != 0
            z = x == 0
            continue
        # LDY IMM/ZPG/ZPX
        elif opc in (0xa0, 0xa4, 0xb4):
            y = value
            n = (y & 0x80) != 0
            z = y == 0
            continue

        # Bitwise operation

        # ORA IMM/ZPG/ZPX/INX/INY
        elif opc in (0x09, 0x05, 0x15, 0x01, 0x11):
            a |= value
            n = (a & 0x80) != 0
            z = a == 0
            continue
        # AND IMM/ZPG/ZPX/INX/INY
        elif opc in (0x29, 0x25, 0x35, 0x21, 0x31):
            a &= value
            n = (a & 0x80) != 0
            z = a == 0
            continue
        # EOR IMM/ZPG/ZPX/INX/INY
        elif opc in (0x49, 0x45, 0x55, 0x41, 0x51):
            a ^= value
            n = (a & 0x80) != 0
            z = a == 0
            continue

        # Arithmetic operation

        # ADC/SBC IMM/ZPG/ZPX/INX/INY
        elif opc in (0x69, 0x65, 0x75, 0x61, 0x71, 0xe9, 0xe5, 0xf5, 0xe1, 0xf1):
            # Invert for SBC
            if opc in (0xe9, 0xe5, 0xf5, 0xe1, 0xf1):
                value = ~value
            # Save signs of operand
            n = a >> 8
            z = value >> 8
            # Add carry
            a += c
            c = a < c
            # Add value
            a += value
            c |= a < value
            # Compute overflow
            v = a >> 8
            v = (v ^ n) & (v ^ z)
            # Set N and Z
            n = (a & 0x80) != 0
            z = a == 0
            continue


        # Increment / decrement

        # INC ZPG/ZPX
        elif opc in (0xe6, 0xf6):
            value += 1
            n = (value & 0x80) != 0
            z = value == 0
            ram[address] = value
            continue
        # DEC ZPG/ZPX
        elif opc in (0xc6, 0xd6):
            value -= 1
            n = (value & 0x80) != 0
            z = value == 0
            ram[address] = value
            continue

        # Shifting

        # ASL ZPG
        elif opc in (0x06, 0x16):
            c = value >> 7
            value <<= 1
            n = value >> 7
            z = value == 0
            ram[address] = value
            continue
        # LSR ZPG
        elif opc in (0x46, 0x56):
            c = value & 0x01
            value >>= 1
            n = value >> 7
            z = value == 0
            ram[address] = value
            continue
        # ROL ZPG
        elif opc in (0x26, 0x36):
            n = c
            c = value >> 7
            value <<= 1
            value |= n
            n = value >> 7
            z = value == 0
            ram[address] = value
            continue
        # ROR ZPG
        elif opc in (0x66, 0x76):
            n = c
            c = value & 0x01
            value >>= 1
            value |= n << 7
            n = value >> 7
            z = value == 0
            ram[address] = value
            continue

        # Comparison

        # CMP IMM/ZPG/ZPX
        elif opc in (0xc9, 0xc5, 0xd5):
            c = a >= value
            value = a - value
            n = (value & 0x80) != 0
            z = value == 0
            continue
        # CPX IMM/ZPG
        elif opc in (0xe0, 0xe4):
            c = x >= value
            value = x - value
            n = (value & 0x80) != 0
            z = value == 0
            continue
        # CPY IMM/ZPG
        elif opc in (0xc0, 0xc4):
            c = y >= value
            value = y - value
            n = (value & 0x80) != 0
            z = value == 0
            continue
        # BIT ZPG
        elif opc == 0x24:
            z = (a & value) == 0
            n = (value & 0x80) != 0
            v = (value & 0x40) != 0
            continue

        # Word operand
        address = (rom[pc - 0x8000] << 8) | operand
        pc += 1

        # ABX and ABY
        if addressing == 0x06 or opc == 0xbe:
            address += y
        elif addressing == 0x07:
            address += x

        # Store

        # STA ABS/ABX/ABY
        if opc in (0x8d, 0x9d, 0x99):
            if address < 0x800:
                ram[address] = a
            else:
                set_cpu_attributes(cpu, pc, a, x, y, sp, n, z, c, v, ic)
                cpu.cpu_write(address, a)
            continue
        # STX ABS
        if opc == 0x8e:
            if address < 0x800:
                ram[address] = x
            else:
                set_cpu_attributes(cpu, pc, a, x, y, sp, n, z, c, v, ic)
                cpu.cpu_write(address, x)
            continue
        # STY ABS
        if opc == 0x8c:
            if address < 0x800:
                ram[address] = y
            else:
                set_cpu_attributes(cpu, pc, a, x, y, sp, n, z, c, v, ic)
                cpu.cpu_write(address, y)
            continue

        # Flow control

        # JSR
        elif opc == 0x20:
            pc -= 1
            ram[0x0100 | sp] = pc >> 8
            sp -= 1
            ram[0x0100 | sp] = pc & 0xff
            sp -= 1
            pc = address
            continue
        # JMP
        elif opc == 0x4c:
            if pc == address + 3:
                pc = address
                break
            pc = address
            continue
        # JMP IND
        elif opc == 0x6c:
            if address < 0x800:
                value = ram[address]
            elif address >= 0x8000:
                value = rom[address - 0x8000]
            else:
                set_cpu_attributes(cpu, pc, a, x, y, sp, n, z, c, v, ic)
                value = cpu.cpu_read(address)
            address += 1
            if address < 0x800:
                pc = (ram[address] << 8) | value
            elif address >= 0x8000:
                pc = (rom[address - 0x8000] << 8) | value
            else:
                set_cpu_attributes(cpu, pc, a, x, y, sp, n, z, c, v, ic)
                pc = (rom[cpu.cpu_read(address)] << 8) | value
            continue

        # Get value at absolute address
        if address < 0x800:
            value = ram[address]
        elif address >= 0x8000:
            value = rom[address - 0x8000]
        else:
            set_cpu_attributes(cpu, pc, a, x, y, sp, n, z, c, v, ic)
            value = cpu.cpu_read(address)

        # Load

        # LDA ABS/ABX/ABY
        if opc in (0xad, 0xbd, 0xb9):
            a = value
            n = value >> 7
            z = value == 0
            continue
        # LDX ABS/ABY
        elif opc in (0xae, 0xbe):
            x = value
            n = value >> 7
            z = value == 0
            continue
        # LDY ABS/ABX
        elif opc in (0xac, 0xbc):
            y = value
            n = value >> 7
            z = value == 0
            continue
        # CMP ABS/ABX/ABY
        elif opc in (0xcd, 0xdd, 0xd9):
            c = a >= value
            value = a - value
            n = (value & 0x80) != 0
            z = value == 0
            continue
        # CPX ABS
        elif opc == 0xec:
            c = x >= value
            value = x - value
            n = (value & 0x80) != 0
            z = value == 0
            continue
        # CPY ABS
        elif opc == 0xcc:
            c = y >= value
            value = y - value
            n = (value & 0x80) != 0
            z = value == 0
            continue
        # INC ABS/ABX
        elif opc in (0xee, 0xfe):
            value += 1
            n = (value & 0x80) != 0
            z = value == 0
            if address < 0x800:
                ram[address] = value
            else:
                set_cpu_attributes(cpu, pc, a, x, y, sp, n, z, c, v, ic)
                cpu.cpu_write(address, value)
            continue
        # DEC ABS/ABX
        elif opc in (0xce, 0xde):
            value -= 1
            n = (value & 0x80) != 0
            z = value == 0
            if address < 0x800:
                ram[address] = value
            else:
                set_cpu_attributes(cpu, pc, a, x, y, sp, n, z, c, v, ic)
                cpu.cpu_write(address, value)
            continue
        # ADC/SBC ABS/ABX/ABY
        elif opc in (0x6d, 0x7d, 0x79, 0xed, 0xfd, 0xf9):
            # Invert for SBC
            if opc in ((0xed, 0xfd, 0xf9)):
                value = ~value
            # Save signs of operand
            n = a >> 8
            z = value >> 8
            # Add carry
            a += c
            c = a < c
            # Add value
            a += value
            c |= a < value
            # Compute overflow
            v = a >> 8
            v = (v ^ n) & (v ^ z)
            # Set N and Z
            n = (a & 0x80) != 0
            z = a == 0
            continue
        # ORA ABS/ABX/ABY
        elif opc in (0x0d, 0x1d, 0x19):
            a |= value
            n = (a & 0x80) != 0
            z = a == 0
            continue
        # AND ABS/ABX/ABY
        elif opc in (0x2d, 0x3d, 0x39):
            a &= value
            n = (a & 0x80) != 0
            z = a == 0
            continue
        # EOR ABS/ABX/ABY
        elif opc in (0x4d, 0x5d, 0x59):
            a ^= value
            n = (a & 0x80) != 0
            z = a == 0
            continue
        # ASL ABS
        elif opc in (0x0e, 0x1e):
            c = value >> 7
            value <<= 1
            n = value >> 7
            z = value == 0
            if address < 0x800:
                ram[address] = value
            else:
                set_cpu_attributes(cpu, pc, a, x, y, sp, n, z, c, v, ic)
                cpu.cpu_write(address, value)
            continue
        # LSR ABS
        elif opc in (0x4e, 0x5e):
            c = value & 0x01
            value >>= 1
            n = value >> 7
            z = value == 0
            if address < 0x800:
                ram[address] = value
            else:
                set_cpu_attributes(cpu, pc, a, x, y, sp, n, z, c, v, ic)
                cpu.cpu_write(address, value)
            continue
        # ROL ABS
        elif opc in (0x2e, 0x3e):
            n = c
            c = value >> 7
            value <<= 1
            value |= n
            n = value >> 7
            z = value == 0
            if address < 0x800:
                ram[address] = value
            else:
                set_cpu_attributes(cpu, pc, a, x, y, sp, n, z, c, v, ic)
                cpu.cpu_write(address, value)
            continue
        # ROR ABS
        elif opc in (0x6e, 0x7e):
            n = c
            c = value & 0x01
            value >>= 1
            value |= n << 7
            n = value >> 7
            z = value == 0
            if address < 0x800:
                ram[address] = value
            else:
                set_cpu_attributes(cpu, pc, a, x, y, sp, n, z, c, v, ic)
                cpu.cpu_write(address, value)
            continue
        # BIT ABS
        elif opc == 0x2c:
            z = (a & value) == 0
            n = (value & 0x80) != 0
            v = (value & 0x40) != 0
            continue

        # Opcode not supported
        pc -= 3
        ic -= 1
        break

    # Set the value back to the CPU instance
    set_cpu_attributes(cpu, pc, a, x, y, sp, n, z, c, v, ic)

    # Except RTI or JMP
    if opc not in (0x40, 0x4c):
        raise ValueError(f"Invalid opcode: 0x{opc:02x}")
    return opc
