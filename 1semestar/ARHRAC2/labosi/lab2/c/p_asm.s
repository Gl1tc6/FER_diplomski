.intel_syntax noprefix
.global zbroj_asm

zbroj_asm:
    push ebp
    mov ebp, esp

    # edx <- n
    mov edx, [ebp+8]    # prvi parametar - n
    
    mov ecx, 0        # brojač
    mov eax, 0        # zbroj

    cmp edx, 0
    jle kraj            # ako je n <= 0, preskoči petlju

petlja:
    add eax, ecx        # dodaj trenutni broj (ecx) u zbroj (eax)
    inc ecx             # povećaj brojač
    cmp ecx, edx        # usporedi brojač s granicom
    jl petlja           # ako je brojač < n, nastavi petlju

kraj:
    leave
    ret
