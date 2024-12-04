.intel_syntax noprefix
.global strojni_potprogram

strojni_potprogram:
# cdecl prolog
push ebp          /* spremi ebp */
mov ebp, esp     /* ubaci esp u ebp */

# glavni dio
mov eax, 42
mov ebx, 0x42

mov dx, 0xFFFF
push dx
pop edx
push dx
pop edx
mov dx, 0

mov dl, 0xDD


# cdecl epilog
leave
ret
