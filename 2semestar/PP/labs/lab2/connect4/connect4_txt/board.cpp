// Implementacija razreda Board

#include<iostream>
#include "board.h"
using namespace std;

#ifndef CHECK
#define CHECKMSG(condition, text) \
if(!(condition)) {fprintf(stderr,"file: " __FILE__ "\nline: %d\nmsg:  " text "\n",__LINE__); exit(1);}
#define CHECK(condition) \
if(!(condition)) {fprintf(stderr,"Assertion failed!\nfile: " __FILE__ "\nline: %d\n" ,__LINE__); exit(1);}
#endif

void Board::Free(void)
{
	static int i;
	if(field == NULL)
		return;
	for(i=0;i<rows;i++)
		delete[] field[i];
	delete[] field;
	delete[] height;
}

void Board::Take(void)
{
	static int i,j;
	field = new CellData*[rows];
	for(i = 0; i<rows; i++)
	{	field[i] = new CellData[cols];
		for(j=0; j<cols; j++)
			field[i][j] = EMPTY;
	}
	height = new int[cols];
	for(i = 0; i<cols; i++)
		height[i] = 0;
}

Board::Board(const Board &src)
{	rows = src.rows;
	cols = src.cols;
	LastMover = src.LastMover;
	lastcol = src.lastcol;
	Take();
	for(int i=0; i<rows; i++)
		for(int j=0; j<cols; j++)
			field[i][j] = src.field[i][j];
	for(int i=0; i<cols; i++)
		height[i] = src.height[i];
}

bool Board::MoveLegal(const int col)	// moze li potez u stupcu col
{	assert(col <= cols);
	if(field[rows-1][col] != EMPTY)
		return false;
	return true;
}

bool Board::Move(const int col, const CellData player)	// napravi potez
{
	if(!MoveLegal(col))
		return false;
	field[height[col]][col] = player;
	height[col]++;
	LastMover = player;
	lastcol = col;
	return true;
}

bool Board::UndoMove(const int col)
{
	assert(col <= cols);
	if(height[col] == 0)
		return false;
	field[height[col]-1][col] = EMPTY;
	height[col]--;
	return true;
}

bool Board::GameEnd(const int lastcol)	// je li zavrsno stanje
{
	int seq, player, row, col, r, c;
	assert(lastcol <= cols);
	col = lastcol;
	row = height[lastcol] - 1;
	if(row < 0)
		return false;
	player = field[row][col];
	// uspravno
	seq = 1; r = row - 1;
	while(r>=0 && field[r][col] == player)
	{	seq++; r--;	}
	if(seq > 3)
		return true;
	// vodoravno
	seq = 0; c = col;
	while((c-1)>=0 && field[row][c-1] == player)
		c--;
	while(c<cols && field[row][c] == player)
	{	seq++; c++;	}
	if(seq > 3)
		return true;
	// koso s lijeva na desno
	seq = 0; r = row; c = col;
	while((c-1)>=0 && (r-1)>=0 && field[r-1][c-1] == player)
	{	c--; r--;	}
	while(c<cols && r<rows && field[r][c] == player)
	{	c++; r++; seq++;	}
	if(seq > 3)
		return true;
	// koso s desna na lijevo
	seq = 0; r = row; c = col;
	while((c-1)>=0 && (r+1)<rows && field[r+1][c-1] == player)
	{	c--; r++;	}
	while(c<cols && r>=0 && field[r][c] == player)
	{	c++; r--; seq++;	}
	if(seq > 3)
		return true;
	return false;
}

bool Board::Load(const char* fname)
{
	CellData value;
	FILE *fp;
	if((fp=fopen(fname,"r"))==NULL)
		CHECKMSG(0,"Nema datoteke!");
	if(!(fscanf(fp,"%d %d",&rows,&cols)!=EOF))
		CHECK(0);
	Free();
	Take();
	// citamo od vrha prema dnu
	for(int r=rows-1; r>=0; r--)
		for(int c=0; c<cols; c++)
		{	if(!(fscanf(fp," %d ",&value)!=EOF))
				CHECK(0);
			field[r][c] = value;
		}
	fclose(fp);
	// odredi visinu stupaca
	for(int c=0; c<cols; c++)
	{	int h;
		for(h=0; h<rows && field[h][c] != EMPTY; h++)
			NULL;
		height[c] = h;
	}
	return true;
}

void Board::Save(const char* fname)
{
	FILE *fp;
	if((fp=fopen(fname,"w+"))==NULL)
		CHECKMSG(0,"Ne mogu otvoriti datoteku!");
	fprintf(fp,"%d %d\n",rows,cols);
	for(int i=rows-1;i>=0;i--)
	{	for(int j=0;j<cols;j++)
			fprintf(fp," %d ", field[i][j]);
		fprintf(fp,"\n");
	}
	fclose(fp);
}