// 4 u nizu - glavni program
#include<iostream>
#include<ctime>
using namespace std;
#include"board.h"

// Za paralelizaciju nam treba još
#include <mpi.h>
#include<vector>

const int DEPTH = 7;	// default dubina stabla
const int TASK_GENERATION_DEPTH = 3; // dubina za generiranje zadataka - može se povećati za više procesora

double Evaluate(Board Current, CellData LastMover, int iLastCol, int iDepth);

struct SimpleTask {
	vector<int> moves;  // sekvenca poteza
	int depth;
	double value;
	int originalColumn; // koji je početni stupac
};

// generiranje sekvenci poteza (bez čuvanja Board objekata)
void GenerateTaskSequences(Board& board, int currentDepth, int maxDepth, 
                          vector<int>& currentMoves, vector<SimpleTask>& tasks,
                          CellData currentPlayer, int originalCol = -1) {
	
	if(currentDepth >= maxDepth) {
		SimpleTask task;
		task.moves = currentMoves;
		task.depth = currentDepth;
		task.value = 0.0;
		task.originalColumn = (originalCol == -1) ? 
			(currentMoves.empty() ? -1 : currentMoves[0]) : originalCol;
		tasks.push_back(task);
		return;
	}
	
	CellData nextPlayer = (currentPlayer == CPU) ? HUMAN : CPU;
	
	for(int col = 0; col < board.Columns(); col++) {
		if(board.MoveLegal(col)) {
			// Dodaj potez
			currentMoves.push_back(col);
			board.Move(col, nextPlayer);
			
			// Provjeri je li igra završena
			if(board.GameEnd(col)) {
				SimpleTask task;
				task.moves = currentMoves;
				task.depth = currentDepth + 1;
				task.originalColumn = (originalCol == -1) ? currentMoves[0] : originalCol;
				if(nextPlayer == CPU) {
					task.value = 1.0; // pobjeda
				} else {
					task.value = -1.0; // poraz
				}
				tasks.push_back(task);
			} else {
				// Rekurzivno generiraj
				int origCol = (originalCol == -1) ? col : originalCol;
				GenerateTaskSequences(board, currentDepth + 1, maxDepth, 
									currentMoves, tasks, nextPlayer, origCol);
			}
			
			// Ukloni potez
			board.UndoMove(col);
			currentMoves.pop_back();
		}
	}
}

int main(int argc, char **argv)
{
    bool debug = false;
	MPI_Init(&argc, &argv);
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (size == 1){		// jedan proces - običan sekvencijalan kod
		MPI_Finalize();

		Board B;
		double dResult, dBest;
		int iBestCol, iDepth = DEPTH;
		if(argc<2)
		{	cout << "Uporaba: <program> <fajl s trenutnim stanjem> [<dubina>]" << endl;
			return 0;
		}
		B.Load(argv[1]);
		if(argc>2)
			iDepth = atoi(argv[2]);
		srand( (unsigned)time( NULL ) );
		// provjerimo jel igra vec gotova (npr. ako je igrac pobijedio)
		for(int iCol=0; iCol<B.Columns(); iCol++)
			if(B.GameEnd(iCol))
			{	cout << "Igra zavrsena!" << endl;
				return 0;
			}
		// pretpostavka: na potezu je CPU
		do
		{	cout << "Dubina: " << iDepth << endl;
			dBest = -1; iBestCol = -1;
			for(int iCol=0; iCol<B.Columns(); iCol++)
			{	if(B.MoveLegal(iCol))
				{	if(iBestCol == -1)
						iBestCol = iCol;
					B.Move(iCol, CPU);
					dResult = Evaluate(B, CPU, iCol, iDepth-1);
					B.UndoMove(iCol);
					if(dResult > dBest || (dResult == dBest && rand()%2 == 0))
					{	dBest = dResult;
						iBestCol = iCol;
					}
					cout << "Stupac " << iCol << ", vrijednost: " << dResult << endl;
				}
			}
			iDepth /= 2;	
		}while(dBest == -1 && iDepth > 0);
		cout << "Najbolji: " << iBestCol << ", vrijednost: " << dBest << endl;
		B.Move(iBestCol, CPU);
		B.Save(argv[1]);
		// jesmo li pobijedili
		for(int iCol=0; iCol<B.Columns(); iCol++)
			if(B.GameEnd(iCol))
			{	cout << "Igra zavrsena! (pobjeda racunala)" << endl;
				return 0;
			}
		return 0;
	}
	
	// više od jednog procesa - paralelno
	Board B;
	double dBest;
	int iBestCol, iDepth = DEPTH;

	if (rank == 0){
		if(argc<2)
		{	cout << "Uporaba: <program> <fajl s trenutnim stanjem> [<dubina>]" << endl;
			MPI_Finalize();
			return 0;
		}
	}

	B.Load(argv[1]);
	if(argc>2)
		iDepth = atoi(argv[2]);

	if(rank == 0){
		srand( (unsigned)time( NULL ) );
		// provjerimo jel igra vec gotova (npr. ako je igrac pobijedio)
		for(int iCol=0; iCol<B.Columns(); iCol++)
			if(B.GameEnd(iCol))
			{	cout << "Igra zavrsena!" << endl;
				MPI_Finalize();
				return 0;
			}
	}

	MPI_Bcast(&iDepth, 1, MPI_INT, 0, MPI_COMM_WORLD);
	
	do
	{
		if(rank == 0) cout << "Dubina: " << iDepth << endl;

		vector<SimpleTask> allTasks;
		
		// Generiraj zadatke (samo na procesu 0)
		if(rank == 0) {
			// Odredimo dubinu za generiranje zadataka - prilagodi broju procesora
			int taskGenDepth = TASK_GENERATION_DEPTH;
			
			// Povećaj dubinu generiranja ako imamo puno procesora
			if(size >= 8) taskGenDepth = min(taskGenDepth + 1, iDepth - 3);
			if(size >= 16) taskGenDepth = min(taskGenDepth + 1, iDepth - 3);
			
			// PAZI: generiramo previše ili premalo zadataka
			// dovoljno dubine za Evaluate funkciju
			taskGenDepth = max(1, min(taskGenDepth, iDepth - 4));
			if(taskGenDepth < 1) taskGenDepth = 1;
			
			cout << "Generiranje zadataka do dubine " << taskGenDepth 
				 << " za " << size << " procesora..." << endl;
			
			vector<int> currentMoves;
			GenerateTaskSequences(B, 0, taskGenDepth, currentMoves, allTasks, EMPTY);
			
			cout << "Generirano " << allTasks.size() << " zadataka" << endl;
		}
		
		// Broadcast
		int numTasks = 0;
		if(rank == 0) numTasks = allTasks.size();
		MPI_Bcast(&numTasks, 1, MPI_INT, 0, MPI_COMM_WORLD);
		
		if(numTasks == 0) {
			// Fallback
			vector<double> columnValues(7, -2.0);
			
			for(int col = rank; col < B.Columns(); col += size) {
				if(B.MoveLegal(col)) {
					Board tempBoard = B;
					tempBoard.Move(col, CPU);
					columnValues[col] = Evaluate(tempBoard, CPU, col, iDepth-1);
					if(rank == 0) {
						cout << "Stupac " << col << ", vrijednost: " << columnValues[col] << endl;
					}
				}
			}
			
			// Gather rezultata
			vector<double> allColumnValues(7 * size, -2.0);
			MPI_Gather(columnValues.data(), 7, MPI_DOUBLE, 
					  allColumnValues.data(), 7, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			
			dBest = -2.0;
			iBestCol = -1;
			
			if(rank == 0) {
				for(int p = 0; p < size; p++) {
					for(int col = 0; col < 7; col++) {
						double value = allColumnValues[p * 7 + col];
						if(value > -2.0 && B.MoveLegal(col)) {
							if(iBestCol == -1 || value > dBest || 
							   (value == dBest && rand() % 2 == 0)) {
								dBest = value;
								iBestCol = col;
							}
						}
					}
				}
			}
		} else {
			// Rad s generiranim zadacima
			if(rank != 0) allTasks.resize(numTasks);
			
			// Pošaljemo zadatke - sekvence
			vector<int> taskSizes(numTasks);
			vector<int> allMoves;
			vector<int> originalCols(numTasks);
			
			if(rank == 0) {
				for(int i = 0; i < numTasks; i++) {
					taskSizes[i] = allTasks[i].moves.size();
					originalCols[i] = allTasks[i].originalColumn;
					for(int move : allTasks[i].moves) {
						allMoves.push_back(move);
					}
				}
			}
			
			MPI_Bcast(taskSizes.data(), numTasks, MPI_INT, 0, MPI_COMM_WORLD);
			MPI_Bcast(originalCols.data(), numTasks, MPI_INT, 0, MPI_COMM_WORLD);
			
			int totalMoves = allMoves.size();
			MPI_Bcast(&totalMoves, 1, MPI_INT, 0, MPI_COMM_WORLD);
			if(rank != 0) allMoves.resize(totalMoves);
			MPI_Bcast(allMoves.data(), totalMoves, MPI_INT, 0, MPI_COMM_WORLD);
			
			// Rekonstruiraj
			if(rank != 0) {
				int moveIndex = 0;
				for(int i = 0; i < numTasks; i++) {
					allTasks[i].moves.clear();
					allTasks[i].originalColumn = originalCols[i];
					for(int j = 0; j < taskSizes[i]; j++) {
						allTasks[i].moves.push_back(allMoves[moveIndex++]);
					}
				}
			}
			
			// Obradi
			vector<double> taskResults(numTasks, -2.0);
			
			for(int i = rank; i < numTasks; i += size) {
				Board taskBoard = B;  // kopiraj početno stanje
				
				// repliciraj sekvencu
				CellData currentPlayer = CPU;
				bool validSequence = true;
				
				for(int move : allTasks[i].moves) {
					if(!taskBoard.MoveLegal(move)) {
						validSequence = false;
						break;
					}
					taskBoard.Move(move, currentPlayer);
					if(taskBoard.GameEnd(move)) {
						// igra završena u ovom zadatku
						taskResults[i] = (currentPlayer == CPU) ? 1.0 : -1.0;
						validSequence = false;
						break;
					}
					currentPlayer = (currentPlayer == CPU) ? HUMAN : CPU;
				}
				
				if(validSequence) {
					//  vrijednost za ostale dubine
					int remainingDepth = iDepth - allTasks[i].moves.size() - 1;
					if(debug){
					cout << "Proces " << rank << ": Task " << i 
						 << ", moves=" << allTasks[i].moves.size()
						 << ", remaining_depth=" << remainingDepth << endl;
                    }
					if(remainingDepth > 0) {
						// probaj sve CPU poteze iz ovog stanja
						double bestValue = -2.0;
						bool foundMove = false;
						
						for(int col = 0; col < taskBoard.Columns(); col++) {
							if(taskBoard.MoveLegal(col)) {
								foundMove = true;
								Board tempBoard = taskBoard;
								tempBoard.Move(col, CPU);
								double value = Evaluate(tempBoard, CPU, col, remainingDepth);
								if(value > bestValue) {
									bestValue = value;
								}
							}
						}
						
						taskResults[i] = foundMove ? bestValue : 0.0;
						if(debug){
						cout << "Proces " << rank << ": Task " << i 
							 << " result=" << taskResults[i] << endl;
                        }
					} else {
						taskResults[i] = 0.0;
                        if(debug){
						cout << "Proces " << rank << ": Task " << i 
							 << " depth exhausted, result=0.0" << endl;
                            }
					}
				}
			}
			
			// Saberi rezultate
			vector<double> allResults(numTasks, -2.0);
			MPI_Allreduce(taskResults.data(), allResults.data(), numTasks, 
						 MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
			
			dBest = -2.0;
			iBestCol = -1;
			
			if(rank == 0) {
				vector<double> columnSums(B.Columns(), 0.0);
				vector<int> columnCounts(B.Columns(), 0);
				
				for(int i = 0; i < numTasks; i++) {
					if(allResults[i] > -2.0) {
						int col = allTasks[i].originalColumn;
						if(col >= 0 && col < B.Columns()) {
							columnSums[col] += allResults[i];
							columnCounts[col]++;
						}
					}
				}
				
				for(int col = 0; col < B.Columns(); col++) {
					if(B.MoveLegal(col) && columnCounts[col] > 0) {
						double avgValue = columnSums[col] / columnCounts[col];
						if(iBestCol == -1 || avgValue > dBest || 
						   (avgValue == dBest && rand() % 2 == 0)) {
							dBest = avgValue;
							iBestCol = col;
						}
						cout << "Stupac " << col << ", vrijednost: " << avgValue 
							 << " (iz " << columnCounts[col] << " zadataka)" << endl;
					}
				}
			}
		}
		
		// broadcast rez
		MPI_Bcast(&iBestCol, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&dBest, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		
		iDepth /= 2;
	} while (dBest == -1 && iDepth > 0);

	if(rank == 0) {
		cout << "Najbolji: " << iBestCol << ", vrijednost: " << dBest << endl;
		if(iBestCol >= 0) {
			B.Move(iBestCol, CPU);
			B.Save(argv[1]);
			
			// jesmo li pobijedili
			for(int iCol=0; iCol<B.Columns(); iCol++)
				if(B.GameEnd(iCol)) {
					cout << "Igra zavrsena! (pobjeda racunala)" << endl;
					break;
				}
		}
	}
	
	MPI_Finalize();
	return 0;
}

// rekurzivna funkcija: ispituje sve moguce poteze i vraca ocjenu dobivenog stanja ploce
double Evaluate(Board Current, CellData LastMover, int iLastCol, int iDepth)
{
	double dResult, dTotal;
	CellData NewMover;
	bool bAllLose = true, bAllWin = true;
	int iMoves;
	
	if(Current.GameEnd(iLastCol))	// igra gotova?
		if(LastMover == CPU)
			return 1;	// pobjeda
		else //if(LastMover == HUMAN)
			return -1;	// poraz
	// nije gotovo, idemo u sljedecu razinu
	if(iDepth == 0)
		return 0;	// a mozda i ne... :)
	iDepth--;
	if(LastMover == CPU)	// tko je na potezu
		NewMover = HUMAN;
	else
		NewMover = CPU;
	dTotal = 0;
	iMoves = 0;	// broj mogucih poteza u ovoj razini
	for(int iCol=0; iCol<Current.Columns(); iCol++)
	{	if(Current.MoveLegal(iCol))	// jel moze u stupac iCol
		{	iMoves++;
			Current.Move(iCol, NewMover);
			dResult = Evaluate(Current, NewMover, iCol, iDepth);
			Current.UndoMove(iCol);
			if(dResult > -1)
				bAllLose = false;
			if(dResult != 1)
				bAllWin = false;
			if(dResult == 1 && NewMover == CPU)
				return 1;	// ako svojim potezom mogu doci do pobjede (pravilo 1)
			if(dResult == -1 && NewMover == HUMAN)
				return -1;	// ako protivnik moze potezom doci do pobjede (pravilo 2)
			dTotal += dResult;
		}
	}
	if(bAllWin == true)	// ispitivanje za pravilo 3.
		return 1;
	if(bAllLose == true)
		return -1;
	dTotal /= iMoves;	// dijelimo ocjenu s brojem mogucih poteza iz zadanog stanja
	return dTotal;
}