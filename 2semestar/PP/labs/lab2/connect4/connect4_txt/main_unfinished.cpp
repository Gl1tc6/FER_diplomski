// 4 u nizu - glavni program
#include<iostream>
#include<ctime>
using namespace std;
#include"board.h"	// razred za igracu plocu

// Za paralelizaciju nam treba još
#include <mpi.h>
#include<vector>

const int DEPTH = 6;	// default dubina stabla

double Evaluate(Board Current, CellData LastMover, int iLastCol, int iDepth);

struct col_res
{
	int col;
	double val;
	bool legal;
};


int main(int argc, char **argv)
{
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
		// zasto petlja? ako svi potezi vode u poraz, racunamo jos jednom za duplo manju dubinu
		// jer igrac mozda nije svjestan mogucnosti pobjede
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
	double dResult, dBest;
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

		vector<col_res> results;
		for(int iCol = rank; iCol < B.Columns(); iCol += size) {
			col_res res;
			res.col = iCol;
			res.legal = B.MoveLegal(iCol);

			if(res.legal) {
				// kopija ploče
				Board tempBoard = B;
				tempBoard.Move(iCol, CPU);
				res.val = Evaluate(tempBoard, CPU, iCol, iDepth-1);
				
				cout << "Proces " << rank << ": Stupac " << iCol 
					<< ", vrijednost: " << res.val << endl;
			} else {
				res.val = -2; // ilegalni move
			}
			results.push_back(res);
		}

		if(results.empty()){
			col_res dummy;
			dummy.col = -1;
			dummy.val = -2;
			dummy.legal = false;
			results.push_back(dummy);
		}

		int cnt = results.size();
		vector<int> allCnts(size);
		MPI_Gather(&cnt, 1, MPI_INT, allCnts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

		vector<int> myCols(cnt);
		vector<double> myVals(cnt);
		vector<int> myLegals(cnt);
		
		for(int i = 0; i < cnt; i++) {
			myCols[i] = results[i].col;
			myVals[i] = results[i].val;
			myLegals[i] = results[i].legal ? 1 : 0;
		}

		// Izračunajte displacement za MPI_Gatherv
		vector<int> displs;
		int tot_res = 0;
		if(rank == 0) {
			displs.resize(size);
			displs[0] = 0;
			for(int i = 1; i < size; i++) {
				displs[i] = displs[i-1] + allCnts[i-1];
			}
			for(int i = 0; i < size; i++) {
				tot_res += allCnts[i];
			}
		}

		// Prikupite podatke ODVOJENO
		vector<int> allCols, allLegals;
		vector<double> allVals;
		if(rank == 0) {
			allCols.resize(tot_res);
			allVals.resize(tot_res);
			allLegals.resize(tot_res);
		}

		MPI_Gatherv(myCols.data(), cnt, MPI_INT, 
					allCols.data(), allCnts.data(), displs.data(), 
					MPI_INT, 0, MPI_COMM_WORLD);
					
		MPI_Gatherv(myVals.data(), cnt, MPI_DOUBLE, 
					allVals.data(), allCnts.data(), displs.data(), 
					MPI_DOUBLE, 0, MPI_COMM_WORLD);
					
		MPI_Gatherv(myLegals.data(), cnt, MPI_INT, 
					allLegals.data(), allCnts.data(), displs.data(), 
					MPI_INT, 0, MPI_COMM_WORLD);

		// najbolji potez
		dBest = -2; 
		iBestCol = -1;

		if(rank == 0) {
			for(int i = 0; i < tot_res; i++) {
				int col = allCols[i];
				double value = allVals[i];
				bool legal = (allLegals[i] == 1);
				
				if(col >= 0 && legal) {
					if(iBestCol == -1 || value > dBest || (value == dBest && rand() % 2 == 0)) {
						dBest = value;
						iBestCol = col;
					}
				}
			}
		}

		// broadcast
		MPI_Bcast(&iBestCol, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&dBest, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		
		iDepth /= 2;
	} while (dBest == -1 && iDepth > 0);

	if(rank == 0) {
		cout << "Najbolji: " << iBestCol << ", vrijednost: " << dBest << endl;
		B.Move(iBestCol, CPU);
		B.Save(argv[1]);
		
		// jesmo li pobijedili
		for(int iCol=0; iCol<B.Columns(); iCol++)
			if(B.GameEnd(iCol)) {
				cout << "Igra zavrsena! (pobjeda racunala)" << endl;
				break;
			}
	}
	
	MPI_Finalize();
	return 0;
}

// rekurzivna funkcija: ispituje sve moguce poteze i vraca ocjenu dobivenog stanja ploce
// Current: trenutno stanje ploce
// LastMover: HUMAN ili CPU
// iLastCol: stupac prethodnog poteza
// iDepth: dubina se smanjuje do 0
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
