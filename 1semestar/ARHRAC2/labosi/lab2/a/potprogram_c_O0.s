	.file	"main.cpp"
	.intel_syntax noprefix
	.text
	.local	_ZStL8__ioinit
	.comm	_ZStL8__ioinit,1,1
	.globl	_Z12potprogram_ciii
	.type	_Z12potprogram_ciii, @function
_Z12potprogram_ciii:
.LFB1723:
	.cfi_startproc
	push	ebp
	.cfi_def_cfa_offset 8
	.cfi_offset 5, -8
	mov	ebp, esp
	.cfi_def_cfa_register 5
	call	__x86.get_pc_thunk.ax
	add	eax, OFFSET FLAT:_GLOBAL_OFFSET_TABLE_
	mov	edx, DWORD PTR 8[ebp]
	mov	eax, DWORD PTR 12[ebp]
	add	eax, edx
	imul	eax, DWORD PTR 16[ebp]
	pop	ebp
	.cfi_restore 5
	.cfi_def_cfa 4, 4
	ret
	.cfi_endproc
.LFE1723:
	.size	_Z12potprogram_ciii, .-_Z12potprogram_ciii
	.section	.rodata
.LC0:
	.string	"ASM: "
.LC1:
	.string	"C++: "
	.text
	.globl	main
	.type	main, @function
main:
.LFB1724:
	.cfi_startproc
	lea	ecx, 4[esp]
	.cfi_def_cfa 1, 0
	and	esp, -16
	push	DWORD PTR -4[ecx]
	push	ebp
	mov	ebp, esp
	.cfi_escape 0x10,0x5,0x2,0x75,0
	push	esi
	push	ebx
	push	ecx
	.cfi_escape 0xf,0x3,0x75,0x74,0x6
	.cfi_escape 0x10,0x6,0x2,0x75,0x7c
	.cfi_escape 0x10,0x3,0x2,0x75,0x78
	sub	esp, 12
	call	__x86.get_pc_thunk.bx
	add	ebx, OFFSET FLAT:_GLOBAL_OFFSET_TABLE_
	sub	esp, 8
	lea	eax, .LC0@GOTOFF[ebx]
	push	eax
	mov	eax, DWORD PTR _ZSt4cout@GOT[ebx]
	push	eax
	call	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@PLT
	add	esp, 16
	mov	esi, eax
	sub	esp, 4
	push	6
	push	5
	push	3
	call	potprogram_asm@PLT
	add	esp, 16
	sub	esp, 8
	push	eax
	push	esi
	call	_ZNSolsEi@PLT
	add	esp, 16
	sub	esp, 8
	mov	edx, DWORD PTR _ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_@GOT[ebx]
	push	edx
	push	eax
	call	_ZNSolsEPFRSoS_E@PLT
	add	esp, 16
	sub	esp, 8
	lea	eax, .LC1@GOTOFF[ebx]
	push	eax
	mov	eax, DWORD PTR _ZSt4cout@GOT[ebx]
	push	eax
	call	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@PLT
	add	esp, 16
	mov	esi, eax
	sub	esp, 4
	push	6
	push	5
	push	3
	call	_Z12potprogram_ciii
	add	esp, 16
	sub	esp, 8
	push	eax
	push	esi
	call	_ZNSolsEi@PLT
	add	esp, 16
	sub	esp, 8
	mov	edx, DWORD PTR _ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_@GOT[ebx]
	push	edx
	push	eax
	call	_ZNSolsEPFRSoS_E@PLT
	add	esp, 16
	mov	eax, 0
	lea	esp, -12[ebp]
	pop	ecx
	.cfi_restore 1
	.cfi_def_cfa 1, 0
	pop	ebx
	.cfi_restore 3
	pop	esi
	.cfi_restore 6
	pop	ebp
	.cfi_restore 5
	lea	esp, -4[ecx]
	.cfi_def_cfa 4, 4
	ret
	.cfi_endproc
.LFE1724:
	.size	main, .-main
	.type	_Z41__static_initialization_and_destruction_0ii, @function
_Z41__static_initialization_and_destruction_0ii:
.LFB2230:
	.cfi_startproc
	push	ebp
	.cfi_def_cfa_offset 8
	.cfi_offset 5, -8
	mov	ebp, esp
	.cfi_def_cfa_register 5
	push	ebx
	sub	esp, 4
	.cfi_offset 3, -12
	call	__x86.get_pc_thunk.bx
	add	ebx, OFFSET FLAT:_GLOBAL_OFFSET_TABLE_
	cmp	DWORD PTR 8[ebp], 1
	jne	.L7
	cmp	DWORD PTR 12[ebp], 65535
	jne	.L7
	sub	esp, 12
	lea	eax, _ZStL8__ioinit@GOTOFF[ebx]
	push	eax
	call	_ZNSt8ios_base4InitC1Ev@PLT
	add	esp, 16
	sub	esp, 4
	lea	eax, __dso_handle@GOTOFF[ebx]
	push	eax
	lea	eax, _ZStL8__ioinit@GOTOFF[ebx]
	push	eax
	mov	eax, DWORD PTR _ZNSt8ios_base4InitD1Ev@GOT[ebx]
	push	eax
	call	__cxa_atexit@PLT
	add	esp, 16
.L7:
	nop
	mov	ebx, DWORD PTR -4[ebp]
	leave
	.cfi_restore 5
	.cfi_restore 3
	.cfi_def_cfa 4, 4
	ret
	.cfi_endproc
.LFE2230:
	.size	_Z41__static_initialization_and_destruction_0ii, .-_Z41__static_initialization_and_destruction_0ii
	.type	_GLOBAL__sub_I__Z12potprogram_ciii, @function
_GLOBAL__sub_I__Z12potprogram_ciii:
.LFB2231:
	.cfi_startproc
	push	ebp
	.cfi_def_cfa_offset 8
	.cfi_offset 5, -8
	mov	ebp, esp
	.cfi_def_cfa_register 5
	sub	esp, 8
	call	__x86.get_pc_thunk.ax
	add	eax, OFFSET FLAT:_GLOBAL_OFFSET_TABLE_
	sub	esp, 8
	push	65535
	push	1
	call	_Z41__static_initialization_and_destruction_0ii
	add	esp, 16
	leave
	.cfi_restore 5
	.cfi_def_cfa 4, 4
	ret
	.cfi_endproc
.LFE2231:
	.size	_GLOBAL__sub_I__Z12potprogram_ciii, .-_GLOBAL__sub_I__Z12potprogram_ciii
	.section	.init_array,"aw"
	.align 4
	.long	_GLOBAL__sub_I__Z12potprogram_ciii
	.section	.text.__x86.get_pc_thunk.ax,"axG",@progbits,__x86.get_pc_thunk.ax,comdat
	.globl	__x86.get_pc_thunk.ax
	.hidden	__x86.get_pc_thunk.ax
	.type	__x86.get_pc_thunk.ax, @function
__x86.get_pc_thunk.ax:
.LFB2232:
	.cfi_startproc
	mov	eax, DWORD PTR [esp]
	ret
	.cfi_endproc
.LFE2232:
	.section	.text.__x86.get_pc_thunk.bx,"axG",@progbits,__x86.get_pc_thunk.bx,comdat
	.globl	__x86.get_pc_thunk.bx
	.hidden	__x86.get_pc_thunk.bx
	.type	__x86.get_pc_thunk.bx, @function
__x86.get_pc_thunk.bx:
.LFB2233:
	.cfi_startproc
	mov	ebx, DWORD PTR [esp]
	ret
	.cfi_endproc
.LFE2233:
	.hidden	__dso_handle
	.ident	"GCC: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0"
	.section	.note.GNU-stack,"",@progbits
