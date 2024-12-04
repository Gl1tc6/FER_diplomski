	.file	"main.cpp"
	.intel_syntax noprefix
	.text
	.globl	_Z12potprogram_ciii
	.type	_Z12potprogram_ciii, @function
_Z12potprogram_ciii:
.LFB1807:
	.cfi_startproc
	mov	eax, DWORD PTR 8[esp]
	add	eax, DWORD PTR 4[esp]
	imul	eax, DWORD PTR 12[esp]
	ret
	.cfi_endproc
.LFE1807:
	.size	_Z12potprogram_ciii, .-_Z12potprogram_ciii
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC0:
	.string	"ASM: "
.LC1:
	.string	"C++: "
	.text
	.globl	main
	.type	main, @function
main:
.LFB1808:
	.cfi_startproc
	lea	ecx, 4[esp]
	.cfi_def_cfa 1, 0
	and	esp, -16
	push	DWORD PTR -4[ecx]
	push	ebp
	mov	ebp, esp
	.cfi_escape 0x10,0x5,0x2,0x75,0
	push	edi
	push	esi
	push	ebx
	push	ecx
	.cfi_escape 0xf,0x3,0x75,0x70,0x6
	.cfi_escape 0x10,0x7,0x2,0x75,0x7c
	.cfi_escape 0x10,0x6,0x2,0x75,0x78
	.cfi_escape 0x10,0x3,0x2,0x75,0x74
	sub	esp, 16
	call	__x86.get_pc_thunk.bx
	add	ebx, OFFSET FLAT:_GLOBAL_OFFSET_TABLE_
	lea	eax, .LC0@GOTOFF[ebx]
	push	eax
	mov	edi, DWORD PTR _ZSt4cout@GOT[ebx]
	push	edi
	call	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@PLT
	mov	esi, eax
	add	esp, 12
	push	6
	push	5
	push	3
	call	potprogram_asm@PLT
	add	esp, 8
	push	eax
	push	esi
	call	_ZNSolsEi@PLT
	mov	DWORD PTR [esp], eax
	call	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_@PLT
	add	esp, 8
	lea	eax, .LC1@GOTOFF[ebx]
	push	eax
	push	edi
	call	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@PLT
	add	esp, 8
	push	48
	push	eax
	call	_ZNSolsEi@PLT
	mov	DWORD PTR [esp], eax
	call	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_@PLT
	add	esp, 16
	mov	eax, 0
	lea	esp, -16[ebp]
	pop	ecx
	.cfi_restore 1
	.cfi_def_cfa 1, 0
	pop	ebx
	.cfi_restore 3
	pop	esi
	.cfi_restore 6
	pop	edi
	.cfi_restore 7
	pop	ebp
	.cfi_restore 5
	lea	esp, -4[ecx]
	.cfi_def_cfa 4, 4
	ret
	.cfi_endproc
.LFE1808:
	.size	main, .-main
	.type	_GLOBAL__sub_I__Z12potprogram_ciii, @function
_GLOBAL__sub_I__Z12potprogram_ciii:
.LFB2298:
	.cfi_startproc
	push	esi
	.cfi_def_cfa_offset 8
	.cfi_offset 6, -8
	push	ebx
	.cfi_def_cfa_offset 12
	.cfi_offset 3, -12
	sub	esp, 16
	.cfi_def_cfa_offset 28
	call	__x86.get_pc_thunk.bx
	add	ebx, OFFSET FLAT:_GLOBAL_OFFSET_TABLE_
	lea	esi, _ZStL8__ioinit@GOTOFF[ebx]
	push	esi
	.cfi_def_cfa_offset 32
	call	_ZNSt8ios_base4InitC1Ev@PLT
	add	esp, 12
	.cfi_def_cfa_offset 20
	lea	eax, __dso_handle@GOTOFF[ebx]
	push	eax
	.cfi_def_cfa_offset 24
	push	esi
	.cfi_def_cfa_offset 28
	push	DWORD PTR _ZNSt8ios_base4InitD1Ev@GOT[ebx]
	.cfi_def_cfa_offset 32
	call	__cxa_atexit@PLT
	add	esp, 20
	.cfi_def_cfa_offset 12
	pop	ebx
	.cfi_restore 3
	.cfi_def_cfa_offset 8
	pop	esi
	.cfi_restore 6
	.cfi_def_cfa_offset 4
	ret
	.cfi_endproc
.LFE2298:
	.size	_GLOBAL__sub_I__Z12potprogram_ciii, .-_GLOBAL__sub_I__Z12potprogram_ciii
	.section	.init_array,"aw"
	.align 4
	.long	_GLOBAL__sub_I__Z12potprogram_ciii
	.local	_ZStL8__ioinit
	.comm	_ZStL8__ioinit,1,1
	.section	.text.__x86.get_pc_thunk.bx,"axG",@progbits,__x86.get_pc_thunk.bx,comdat
	.globl	__x86.get_pc_thunk.bx
	.hidden	__x86.get_pc_thunk.bx
	.type	__x86.get_pc_thunk.bx, @function
__x86.get_pc_thunk.bx:
.LFB2300:
	.cfi_startproc
	mov	ebx, DWORD PTR [esp]
	ret
	.cfi_endproc
.LFE2300:
	.hidden	__dso_handle
	.ident	"GCC: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0"
	.section	.note.GNU-stack,"",@progbits
