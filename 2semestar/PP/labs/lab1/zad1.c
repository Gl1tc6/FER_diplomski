#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <stdbool.h>
#include <sys/time.h>

// Program params
#define MAX_CYCLES 3
#define MAX_PROCESSES 50
#define MAX_THINK 3
#define MAX_EAT 3
#define LEFT_FORK_REQUEST 1
#define LEFT_FORK_REPLY 2
#define RIGHT_FORK_REQUEST 3
#define RIGHT_FORK_REPLY 4

#define RED     "\x1B[31m"
#define GREEN   "\x1B[32m"
#define YELLOW  "\x1B[33m"
#define GRAY    "\x1B[90m"
#define BOLD_CYAN "\x1B[1;36m"
#define RESET   "\x1B[0m"

// dobivanje vremena koje je prošlo
double get_time(struct timeval start) {
    struct timeval now;
    gettimeofday(&now, NULL);
    return (now.tv_sec - start.tv_sec) + (now.tv_usec - start.tv_usec) / 1000000.0;
}

// printanje sa trenutno prošlim vremenom i točnom indentacijom
void n_proc_print(char* prstr, int rank, struct timeval st){
    double time_pass = get_time(st);
    printf("[%.4f ms]", time_pass);
    for(int i =0; i<rank; i++){
        printf("\t");
    }
    printf(" %s\n", prstr);
}

int main(int argc, char** argv){
    // Initialize MPI
    int rank, size, dummy = 0;
    struct timeval s_time;

    MPI_Init(&argc, &argv);

    // Get total number of processes and current rank
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Validate number of processes
    if (size < 2 || size > MAX_PROCESSES) {
        if (rank == 0) {
            printf("Error: At least 2 processes required but no more than %d\n", MAX_PROCESSES);
        }
        MPI_Finalize();
        return 1;
    }

    srand(time(NULL)+rank);
    gettimeofday(&s_time, NULL);

    int lf, rf;
    int req_r = 0, req_l = 0;
    int pend_req_r = -1, pend_req_l = -1;

    int lsusjed = (rank + size-1) % size;
    int rsusjed = (rank+1) % size;

    /* 
        raspoređivanje bešteka (inicijalno)
        0 - nema vilicu
        1 - imam vilicu
        2 - imam prljavu vilicu
    */
    lf = (rank < lsusjed) ? 2 : 0;
    rf = (rank < rsusjed) ? 2 : 0;

    char buffer[200];
    sprintf(buffer, "Filozof %d započinje s: %s | %s", 
        rank, 
        lf > 0 ? "[lijeva vilica]" : "[ništa]", 
        rf > 0 ? "[desna vilica]" : "[ništa]"); 

    n_proc_print(buffer, rank, s_time);
    
    MPI_Barrier(MPI_COMM_WORLD);

    for(int cycle=0; cycle < MAX_CYCLES; cycle++) {
        
        // Razmisljanje
        int think = rand() % MAX_THINK + 1;
        
        sprintf(buffer, "Filozof %d razmišlja %d sekundi", 
        rank, 
        think); 
        n_proc_print(buffer, rank, s_time);
        double think_start = get_time(s_time);
        while(get_time(s_time) - think_start < think) {
            MPI_Status stat;
            int flag = 0;
            MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &stat);

            if (flag) {
                MPI_Recv(&dummy, 1, MPI_INT, MPI_ANY_SOURCE, stat.MPI_TAG, MPI_COMM_WORLD, &stat);
                
                if (stat.MPI_TAG == LEFT_FORK_REQUEST && lf == 2) {
                    printf("%sFilozof %d čisti lijevu vilicu%s\n",GRAY, rank, RESET);
                    lf = 0;
                    MPI_Send(&dummy, 1, MPI_INT, stat.MPI_SOURCE, LEFT_FORK_REPLY, MPI_COMM_WORLD);
                    sprintf(buffer, "%d šalje lijevu vilicu %d", rank, stat.MPI_SOURCE);
                    n_proc_print(buffer, rank, s_time);
                    //printf("[stanje%d] lf=%d rf=%d reqL=%d reqR=%d\n", rank, lf, rf, req_l, req_r);
                } 
                else if (stat.MPI_TAG == RIGHT_FORK_REQUEST && rf == 2) {
                    printf("%sFilozof %d čisti desnu vilicu%s\n",GRAY, rank, RESET);
                    lf = 0;
                    MPI_Send(&dummy, 1, MPI_INT, stat.MPI_SOURCE, RIGHT_FORK_REPLY, MPI_COMM_WORLD);
                    sprintf(buffer, "%d šalje desnu vilicu %d", rank, stat.MPI_SOURCE);
                    n_proc_print(buffer, rank, s_time);
                    //printf("[stanje%d] lf=%d rf=%d reqL=%d reqR=%d\n", rank, lf, rf, req_l, req_r);
                }
                else if (stat.MPI_TAG == LEFT_FORK_REPLY) {
                    lf = 1;
                    req_l = 0;
                    sprintf(buffer, "%d dobio lijevu vilicu od %d", rank , stat.MPI_SOURCE);
                    n_proc_print(buffer, rank, s_time);
                    //printf("[stanje%d] lf=%d rf=%d reqL=%d reqR=%d\n", rank, lf, rf, req_l, req_r);
                }
                else if (stat.MPI_TAG == RIGHT_FORK_REPLY) {
                    rf = 1;
                    req_r = 0;
                    sprintf(buffer, "%d dobio desnu vilicu od %d", rank, stat.MPI_SOURCE);
                    n_proc_print(buffer, rank, s_time);
                    //printf("[stanje%d] lf=%d rf=%d reqL=%d reqR=%d\n", rank, lf, rf, req_l, req_r);
                }
            }
            //usleep(100000);  // 10ms
        }

        // gladni filozof
        sprintf(buffer,RED "Filozof %d je gladan" RESET, 
        rank);
        n_proc_print(buffer, rank, s_time);

        // // traži lijevu vilicu
        if(lf == 0 && !req_l) {
            MPI_Send(&dummy, 1, MPI_INT, lsusjed, LEFT_FORK_REQUEST, MPI_COMM_WORLD);
            req_l = 1;
            sprintf(buffer,YELLOW "Traži lijevu vilicu od %d" RESET, lsusjed);
            n_proc_print(buffer, rank, s_time);
            //printf("[stanje%d] lf=%d rf=%d reqL=%d reqR=%d\n", rank, lf, rf, req_l, req_r);
        }

        // traži desnu vilicu
        if(rf == 0 && !req_r) {
            MPI_Send(&dummy, 1, MPI_INT, rsusjed, RIGHT_FORK_REQUEST, MPI_COMM_WORLD);
            req_r = 1;
            sprintf(buffer,YELLOW "Traži desnu vilicu od %d" RESET, rsusjed);
            n_proc_print(buffer, rank, s_time);
            //printf("[stanje%d] lf=%d rf=%d reqL=%d reqR=%d\n", rank, lf, rf, req_l, req_r);
        }

        // čekanje na obje vilice
        while (lf == 0 || rf == 0) {
            MPI_Status stat;
            int flag = 0;
            MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &stat);

            if (flag) {
                MPI_Recv(&dummy, 1, MPI_INT, MPI_ANY_SOURCE, stat.MPI_TAG, MPI_COMM_WORLD, &stat);
                
                if (stat.MPI_TAG == LEFT_FORK_REPLY) {
                    lf = 1;
                    req_l = 0;
                    sprintf(buffer, "%d dobio lijevu vilicu od %d", rank , stat.MPI_SOURCE);
                    n_proc_print(buffer, rank, s_time);
                    fflush(NULL);
                    //printf("[stanje%d] lf=%d rf=%d reqL=%d reqR=%d\n", rank, lf, rf, req_l, req_r);
                }
                else if (stat.MPI_TAG == RIGHT_FORK_REPLY) {
                    rf = 1;
                    req_r = 0;
                    sprintf(buffer, "%d dobio desnu vilicu od %d", rank, stat.MPI_SOURCE);
                    n_proc_print(buffer, rank, s_time);
                    fflush(NULL);
                    //printf("[stanje%d] lf=%d rf=%d reqL=%d reqR=%d\n", rank, lf, rf, req_l, req_r);
                }
                else if (stat.MPI_TAG == LEFT_FORK_REQUEST) {
                    pend_req_l = stat.MPI_SOURCE;
                    sprintf(buffer, YELLOW "Zabilježen zahtjev za lijevom vilicom od %d (ja sam gladan, ti čekaj)" RESET, 
                        stat.MPI_SOURCE);
                    n_proc_print(buffer, rank, s_time);
                }
                else if (stat.MPI_TAG == RIGHT_FORK_REQUEST) {
                    pend_req_r = stat.MPI_SOURCE;
                    sprintf(buffer,YELLOW "Zabilježen zahtjev za desnom vilicom od %d (ja sam gladan, ti čekaj)" RESET, 
                           stat.MPI_SOURCE);
                    n_proc_print(buffer, rank, s_time);
                }
            }
            //usleep(100000);  // 10ms
        }

        // jelo
        int jelo = rand() % MAX_EAT + 1;
        sprintf(buffer,"Filozof %d jede %d sekundi", rank, jelo); 
        n_proc_print(buffer, rank, s_time);
        //printf("[stanje%d] lf=%d rf=%d reqL=%d reqR=%d\n", rank, lf, rf, req_l, req_r);
        
        double eat_start = get_time(s_time);
        while (get_time(s_time) - eat_start < jelo) {
            // Check for requests while eating (but don't give forks yet)
            MPI_Status stat;
            int flag = 0;
            MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &stat);
            
            if (flag) {
                MPI_Recv(&dummy, 1, MPI_INT, MPI_ANY_SOURCE, stat.MPI_TAG, MPI_COMM_WORLD, &stat);
                
                if (stat.MPI_TAG == LEFT_FORK_REQUEST) {
                    pend_req_l = stat.MPI_SOURCE;
                    sprintf(buffer, YELLOW "Zabilježen zahtjev za lijevom vilicom od %d (jedem)" RESET, 
                        stat.MPI_SOURCE);
                    n_proc_print(buffer, rank, s_time);
                }
                else if (stat.MPI_TAG == RIGHT_FORK_REQUEST) {
                    pend_req_r = stat.MPI_SOURCE;
                    sprintf(buffer,YELLOW "Zabilježen zahtjev za desnom vilicom od %d (jedem)" RESET, 
                           stat.MPI_SOURCE);
                    n_proc_print(buffer, rank, s_time);
                }
            }
            usleep(100000);  // 100ms
        }

        lf = 2;
        rf = 2;
        //printf("[stanje%d] lf=%d rf=%d reqL=%d reqR=%d\n", rank, lf, rf, req_l, req_r);

        sprintf(buffer,GREEN "Filozof %d je pojeo!" RESET, rank); 
        n_proc_print(buffer, rank, s_time);

        if (pend_req_l != -1) {
            MPI_Send(&dummy, 1, MPI_INT, pend_req_l, LEFT_FORK_REPLY, MPI_COMM_WORLD);
            sprintf(buffer, "Pošalji lijevu vilicu %d (prijašnji zahtjev)", pend_req_l);
            n_proc_print(buffer, rank, s_time);
            lf = 0;
            pend_req_l = -1;
        }
        
        if (pend_req_r != -1) {
            MPI_Send(&dummy, 1, MPI_INT, pend_req_r, RIGHT_FORK_REPLY, MPI_COMM_WORLD);
            sprintf(buffer, "Pošalji desnu vilicu %d (prijašnji zahtjev)",pend_req_r);
            n_proc_print(buffer, rank, s_time);
            rf = 0;
            pend_req_r = -1;
        }
    }
    printf(BOLD_CYAN "########## Gotov sam!!! Javio se %d ##########\n" RESET, rank);
    if (lf == 2) {
        MPI_Send(&dummy, 1, MPI_INT, lsusjed, LEFT_FORK_REPLY, MPI_COMM_WORLD);
        sprintf(buffer, "Pošalji lijevu vilicu %d (nakon svega)", lsusjed);
        n_proc_print(buffer, rank, s_time);
        lf = 0;
    }
    
    if (rf == 2) {
        MPI_Send(&dummy, 1, MPI_INT, rsusjed, RIGHT_FORK_REPLY, MPI_COMM_WORLD);
        sprintf(buffer, "Pošalji desnu vilicu %d (nakon svega)", rsusjed);
        n_proc_print(buffer, rank, s_time);
        rf = 0;
    }
    MPI_Finalize();
    return 0;
}