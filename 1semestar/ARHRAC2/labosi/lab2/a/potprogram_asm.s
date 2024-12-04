  // ovo je komentar
  //
  // oznaka sintakse:
  .intel_syntax noprefix

  // neka simbol potprogram_asm
  // bude vidljiv izvana:
  .global potprogram_asm

  // odredišna oznaka potprograma:
  potprogram_asm:
    push  ebp          /* spremi ebp */
    mov   ebp, esp     /* ubaci esp u ebp */
                        /* zauzmi 4 bajta za lokalne varijable: */
    sub   esp, 4       /* lokalne varijable su "ispod" ebp */
    
    // glavna funkcionalnost potprograma
    mov   eax, [ebp+12] /* b */
    add   eax, [ebp+8]  /* a */
    imul  eax, [ebp+16] /* c */
    // povratna vrijednost je zadržana u eax*/


                        /* oslobodi lokalne varijable:*/

                        /* cdecl epilog: */
    leave            /* umjesto 'add esp,4, pop ebp' može biti 'leave'*/
    ret                /* povratak iz potprograma */  
