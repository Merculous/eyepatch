
from struct import unpack
from sys import version_info
from typing import Generator, Tuple

from capstone import (
    CS_ARCH_ARM,
    CS_MODE_ARM,
    CS_MODE_LITTLE_ENDIAN,
    CS_MODE_THUMB,
    Cs,
    CsError,
)
from capstone.arm_const import ARM_GRP_JUMP, ARM_OP_IMM, ARM_OP_MEM, ARM_REG_PC
from keystone import KS_ARCH_ARM, KS_MODE_ARM, KS_MODE_THUMB, Ks, KsError

import eyepatch
import eyepatch.base

if version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

ARM_SIZE = 4
THUMB_SIZE = 2

ARM_BITWISE = 3
THUMB_BITWISE = 1


class ByteString(eyepatch.base._ByteString):
    pass


class Insn(eyepatch.base._Insn):
    def follow_call(self) -> Self:
        if not self.info.group(ARM_GRP_JUMP):
            raise eyepatch.InsnError('Instruction is not a call')

        op = self.info.operands[-1]

        if op.type == ARM_OP_IMM:
            return next(self.patcher.disasm(op.imm + self.offset))

    def patch(self, insn: str, force_thumb: bool = True) -> None:
        insn_opcode = b''

        if not isinstance(insn, str):
            raise Exception('insn must be a string!')

        if insn == '':
            raise Exception('insn is string but is empty!')

        # Check how many instructions we are using.
        # Capstone allows ";" to indicate to use more
        # than one instruction.

        insn_count = 1 if ';' not in insn else len(insn.split(';'))

        if insn_count > 2:
            raise Exception(f'Expected 1 or 2 instructions got: {insn_count}!')

        # Get the expected length of opcodes to match the original
        # Ignore "self.data" for now as it saves the buffer size
        # of 4 bytes, even if in "force_thumb" mode's "ARM" (1st)
        # value's Insn.bytes length is 2.

        orig_len = len(self.info.bytes)

        insn_1 = None
        insn_2 = None
        tmp = None

        mode = ''

        if force_thumb:
            if insn_count == 2:
                tmp = insn.split(';')

                # First insn
                insn_1 = self.patcher.asm_thumb(tmp[0])
                insn_opcode += insn_1

                # Second insn
                insn_2 = self.patcher.asm_thumb(tmp[1])
                insn_opcode += insn_2

            else:
                insn_1 = self.patcher.asm_thumb(insn)
                insn_opcode += insn_1

            mode += 'thumb'

        else:
            insn_1 = self.patcher.asm(insn)
            insn_opcode += insn_1
            mode += 'arm'

        if len(insn_opcode) > orig_len:
            raise Exception(
                'Assembled instruction opcode is longer the orignal instruction!')

        self._data = bytearray(insn_opcode)
        self.patcher._data[self.offset:self.offset +
                           len(insn_opcode)] = insn_opcode

        # "new_arm" can also be THUMB code, when "force_thumb is True".
        # This means that first disasm value can be valid a THUMB instruction,
        # but like ARM, is 4 bytes. The next value, will default to 2 byte THUMB.

        new_insn = None
        new_arm, new_thumb = next(self.patcher.disasm(
            self.offset, force_thumb=force_thumb))

        if mode == 'thumb':
            if force_thumb:
                new_insn = new_arm  # Can either be 2 or 4 byte THUMB
            else:
                new_insn = new_thumb  # Force 2 byte THUMB
        else:
            new_insn = new_arm  # 4 byte ARM only

        # Update our previous instruction with patched
        self._info = new_insn.info

    def getValuePointedToByLDR(self, buffer: bytes, bitwise: int) -> int:
        op = self.info.operands[-1]
        insn_value = (self.offset & ~bitwise) + op.mem.disp + 4
        value_data = buffer[insn_value:insn_value+4]
        value = unpack('<I', value_data)[0]
        return value


class Patcher(eyepatch.base._Patcher):
    _insn = Insn
    _string = ByteString

    def __init__(self, data: bytes):
        super().__init__(
            data=data,
            asm=Ks(KS_ARCH_ARM, KS_MODE_ARM),
            disasm=Cs(CS_ARCH_ARM, CS_MODE_ARM + CS_MODE_LITTLE_ENDIAN),
        )

        self._thumb_asm = Ks(KS_ARCH_ARM, KS_MODE_THUMB)
        self._thumb_disasm = Cs(CS_ARCH_ARM, CS_MODE_THUMB + CS_MODE_LITTLE_ENDIAN)
        self._thumb_disasm.detail = True

    def asm_thumb(self, insn: str) -> bytes:
        try:
            asm, _ = self._thumb_asm.asm(insn, as_bytes=True)
        except KsError:
            raise eyepatch.AssemblyError(
                f'Failed to assemble ARM thumb instruction: {insn}'
            )

        return asm

    def disasm(
        self, offset: int = 0, reverse: bool = False, force_thumb: bool = True
    ) -> Generator[Tuple[_insn, _insn], None, None]:

        i = offset

        while i >= 0 and i < len(self.data):
            thumb_insn = None
            arm_insn = None

            arm_buffer = self.data[i:i+ARM_SIZE]
            thumb_buffer = arm_buffer[:THUMB_SIZE]

            try:
                if force_thumb:
                    arm_insn = next(self._thumb_disasm.disasm(arm_buffer, i))
                else:
                    arm_insn = next(self._disasm.disasm(arm_buffer, i))
            except StopIteration:
                pass

            try:
                thumb_insn = next(self._thumb_disasm.disasm(thumb_buffer, i))
            except StopIteration:
                pass

            if arm_insn:
                arm_insn = self._insn(i, arm_buffer, arm_insn, self)

            if thumb_insn:
                thumb_insn = self._insn(i, thumb_buffer, thumb_insn, self)

            yield arm_insn, thumb_insn

            if reverse:
                i -= THUMB_SIZE
            else:
                i += THUMB_SIZE

    def search_imm(
        self, imm: int, offset: int = 0, reverse: bool = False, skip: int = 0
    ) -> _insn:

        match = None

        for arm_insn, thumb_insn in self.disasm(offset, reverse):
            loop_count = 1
            current_insn = None

            if arm_insn and not thumb_insn:
                current_insn = arm_insn

            elif thumb_insn and not arm_insn:
                current_insn = thumb_insn

            elif arm_insn and thumb_insn:
                loop_count += 1
                current_insn = arm_insn
            else:
                continue

            for x in range(loop_count):
                bitwise = 0

                if x == 1:
                    current_insn = thumb_insn

                if len(current_insn.info.operands) == 0:
                    continue

                op = current_insn.info.operands[-1]

                if op.type == ARM_OP_MEM:
                    if op.mem.base != ARM_REG_PC:
                        continue

                    insn_size = current_insn.info.size

                    if insn_size == ARM_SIZE or x == 0:
                        # If ARM or THUMB (4 byte mode)
                        bitwise += ARM_BITWISE

                    elif insn_size == THUMB_SIZE:
                        bitwise += THUMB_BITWISE

                    else:
                        raise Exception(f'Unknown insn size: {insn_size}')

                    insn_imm = current_insn.getValuePointedToByLDR(
                        self.data, bitwise)

                    if insn_imm != imm:
                        continue

                    match = current_insn

                elif op.type == ARM_OP_IMM:
                    if op.imm != imm:
                        continue

                    match = current_insn

                if match:
                    if skip != 0:
                        # FIXME: This might be wrong when
                        # loop_count == 2.
                        match = None
                        skip -= 1

                    break

            if match:
                break

        if match is None:
            raise eyepatch.SearchError(
                f'Failed to find instruction with immediate value: {hex(imm)}'
            )

        return match

    def search_insn(
        self, insn_name: str, offset: int = 0, skip: int = 0, reverse: bool = False
    ) -> _insn:

        match = None

        for arm_insn, thumb_insn in self.disasm(offset, reverse):
            loop_count = 1
            current_insn = None

            if arm_insn and not thumb_insn:
                current_insn = arm_insn

            elif thumb_insn and not arm_insn:
                current_insn = thumb_insn

            elif arm_insn and thumb_insn:
                loop_count += 1
                current_insn = arm_insn

            else:
                continue

            for x in range(loop_count):
                if x == 1:
                    current_insn = thumb_insn

                if current_insn.info.mnemonic == insn_name:
                    if skip == 0:
                        match = current_insn

                    skip -= 1

                if match:
                    break

            if match:
                break

        if match is None:
            raise eyepatch.SearchError(
                f'Failed to find instruction: {insn_name}')

        return match

    def search_thumb_insns(self, *insns: str, offset: int = 0) -> _insn:
        instructions = '\n'.join(insns)
        data = self.asm_thumb(instructions)
        offset = self.data.find(data, offset)
        if offset == -1:
            raise eyepatch.SearchError(
                f'Failed to find instructions: {instructions} at offset: {hex(offset)}'
            )

        return next(self.disasm(offset))

    def search_xref(
        self, offset: int, base_addr: int, skip: int = 0, is_kernel: bool = False
    ) -> _insn:
        KERNEL_DIFFERENCE = 0x36000

        # Always get THUMB instructions, even if the instruction is 4 bytes.
        # See xref comment below on why.

        match = None

        for arm_insn, thumb_insn in self.disasm(offset, True):
            # For xref's, we MUST be using THUMB mode, regardless if such
            # instructions are 4 bytes long, like the "LDR.W" insn.

            current_insn = None
            two_insns = False

            if thumb_insn is None and arm_insn:
                # THUMB failed, ARM worked
                current_insn = arm_insn

            elif arm_insn is None and thumb_insn:
                # ARM failed, THUMB worked
                current_insn = thumb_insn

            elif thumb_insn is None and arm_insn is None:
                # ARM and THUMB failed
                continue

            else:
                # Successfully got both ARM and THUMB instructions
                # Check ARM first, then do THUMB

                two_insns = True
                current_insn = arm_insn

            loop_count = 2 if two_insns else 1

            for x in range(loop_count):
                bitwise = 0

                if x == 1:
                    current_insn = thumb_insn

                if len(current_insn.info.operands) == 0:
                    continue

                op = current_insn.info.operands[-1]

                if op.type != ARM_OP_MEM:
                    continue

                if op.mem.base != ARM_REG_PC:
                    continue

                insn_size = current_insn.info.size

                if insn_size == ARM_SIZE or x == 0:
                    # If ARM or THUMB (4 byte mode)
                    bitwise += ARM_BITWISE

                elif insn_size == THUMB_SIZE:
                    bitwise += THUMB_BITWISE

                else:
                    raise Exception(f'Unknown insn size: {insn_size}')

                offset2 = current_insn.getValuePointedToByLDR(
                    self.data, bitwise)
                result = offset2 - base_addr

                if not is_kernel:
                    if result != offset:
                        continue

                    match = current_insn

                else:
                    if result - offset != KERNEL_DIFFERENCE:
                        continue

                    match = current_insn

                if skip == 0:
                    break

                skip -= 1

            if match:
                break

        if match is None:
            raise eyepatch.SearchError(
                f'Failed to find xrefs to offset: 0x{offset:x}')

        return match

    def search_string(
        self,
        string: bytes,
        offset: int = 0,
        skip: int = 0,
        exact: bool = True
    ) -> _string:

        # TODO
        # SKIP VAR

        expected = b'\x00' + string + b'\x00'

        match = None

        if exact:
            match = self.data.find(expected, offset)
            match += 1

        else:
            match = self.data.find(string, offset)

        if match is None:
            raise ValueError('Either string or offset must be provided.')

        str_buff = self._data[match:match+len(string)]

        return self._string(match, str_buff, self)
