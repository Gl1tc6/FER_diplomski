	.file	"main.cpp"
	.intel_syntax noprefix
	.text
	.p2align 4
	.type	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0, @function
_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0:
.LFB2301:
	.cfi_startproc
	push	edi
	.cfi_def_cfa_offset 8
	.cfi_offset 7, -8
	push	esi
	.cfi_def_cfa_offset 12
	.cfi_offset 6, -12
	mov	esi, eax
	push	ebx
	.cfi_def_cfa_offset 16
	.cfi_offset 3, -16
	mov	eax, DWORD PTR [eax]
	call	__x86.get_pc_thunk.bx
	add	ebx, OFFSET FLAT:_GLOBAL_OFFSET_TABLE_
	mov	eax, DWORD PTR -12[eax]
	mov	edi, DWORD PTR 124[esi+eax]
	test	edi, edi
	je	.L7
	cmp	BYTE PTR 28[edi], 0
	je	.L3
	movzx	eax, BYTE PTR 39[edi]
.L4:
	sub	esp, 8
	.cfi_def_cfa_offset 24
	movsx	eax, al
	push	eax
	.cfi_def_cfa_offset 28
	push	esi
	.cfi_def_cfa_offset 32
	call	_ZNSo3putEc@PLT
	mov	DWORD PTR [esp], eax
	call	_ZNSo5flushEv@PLT
	add	esp, 16
	.cfi_def_cfa_offset 16
	pop	ebx
	.cfi_remember_state
	.cfi_restore 3
	.cfi_def_cfa_offset 12
	pop	esi
	.cfi_restore 6
	.cfi_def_cfa_offset 8
	pop	edi
	.cfi_restore 7
	.cfi_def_cfa_offset 4
	ret
.L3:
	.cfi_restore_state
	sub	esp, 12
	.cfi_def_cfa_offset 28
	push	edi
	.cfi_def_cfa_offset 32
	call	_ZNKSt5ctypeIcE13_M_widen_initEv@PLT
	pop	eax
	.cfi_def_cfa_offset 28
	mov	eax, DWORD PTR [edi]
	pop	edx
	.cfi_def_cfa_offset 24
	push	10
	.cfi_def_cfa_offset 28
	push	edi
	.cfi_def_cfa_offset 32
	call	[DWORD PTR 24[eax]]
	add	esp, 16
	.cfi_def_cfa_offset 16
	jmp	.L4
.L7:
	call	_ZSt16__throw_bad_castv@PLT
	.cfi_endproc
.LFE2301:
	.size	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0, .-_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
	.p2align 4
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
	.section	.text.startup,"ax",@progbits
	.p2align 4
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
	push	esi
	push	ebx
	.cfi_escape 0x10,0x6,0x2,0x75,0x7c
	.cfi_escape 0x10,0x3,0x2,0x75,0x78
	call	__x86.get_pc_thunk.bx
	add	ebx, OFFSET FLAT:_GLOBAL_OFFSET_TABLE_
	push	ecx
	.cfi_escape 0xf,0x3,0x75,0x74,0x6
	sub	esp, 16
	push	5
	mov	esi, DWORD PTR _ZSt4cout@GOT[ebx]
	lea	eax, .LC0@GOTOFF[ebx]
	push	eax
	push	esi
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_i@PLT
	add	esp, 12
	push	6
	push	5
	push	3
	call	potprogram_asm@PLT
	pop	edx
	pop	ecx
	push	eax
	push	esi
	call	_ZNSolsEi@PLT
	add	esp, 16
	call	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
	sub	esp, 4
	lea	eax, .LC1@GOTOFF[ebx]
	push	5
	push	eax
	push	esi
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_i@PLT
	pop	eax
	pop	edx
	push	48
	push	esi
	call	_ZNSolsEi@PLT
	add	esp, 16
	call	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
	lea	esp, -12[ebp]
	xor	eax, eax
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
.LFE1808:
	.size	main, .-main
	.p2align 4
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
	call	__x86.get_pc_thunk.bx
	add	ebx, OFFSET FLAT:_GLOBAL_OFFSET_TABLE_
	sub	esp, 16
	.cfi_def_cfa_offset 28
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
.LFB2303:
	.cfi_startproc
	mov	ebx, DWORD PTR [esp]
	ret
	.cfi_endproc
.LFE2303:
	.hidden	__dso_handle
	.ident	"GCC: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0"
	.section	.note.GNU-stack,"",@progbits
