/******************************************************************
 *
 * A simple PGM reader/writer crafted for 5kk73 GPU assignment
 *
 *****************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "stereo.h"

extern "C"
int readPGM(unsigned char **image, char *filename,		\
	    int *imageWidth, int *imageHeight, int *grayMax);

extern "C"
int writePGM(unsigned char *image, char *filename,		\
	     int imageWidth, int imageHeight, int grayMax);

int readPGM(unsigned char **image, char *filename,		\
	    int *imageWidth, int *imageHeight, int *grayMax){
  
  int PGM_HEADER_LINES = 3;
  
  FILE *input;
  
  int headerLines = 1;
  int scannedLines = 0;
  
  int counter = 0;

  // read header strings
  char *lineBuffer = (char *) malloc (LINE_BUFFER_SIZE+1);
  char *split;
  // format
  char *format = (char *) malloc (LINE_BUFFER_SIZE+1) ;
  // PGM binary
  char P5[]="P5";
  // comments
  char comments[LINE_BUFFER_SIZE+1];

  //////////////////////////////////////////////////////////////////
  //
  // Open the input PGM file
  //
  //////////////////////////////////////////////////////////////////
  
  input=fopen(filename,"r");

  //////////////////////////////////////////////////////////////////
  //
  // Read the intput PGM file header
  //
  //////////////////////////////////////////////////////////////////

  printf("\nRead image file: %s\n", filename);

  // read header strings
  while (scannedLines < headerLines) {
    fgets(lineBuffer, LINE_BUFFER_SIZE, input);
    // if not comments
    if(lineBuffer[0] != '#') {
      scannedLines = scannedLines + 1;
      // read the format
      if(scannedLines == 1) {
	split = strtok(lineBuffer, " \n");
	strcpy(format,split);
	if (strcmp(format,P5) == 0){
	  printf("FORMAT: %s\n",format);
	  headerLines = PGM_HEADER_LINES;
	} else{
	  printf("Only PGM P5 format is support.\n");
	}
      }
      // read width and height
      if (scannedLines == 2) {
	split = strtok (lineBuffer, " \n");
	*imageWidth = atoi(split);
	printf("WIDTH: %d, ", *imageWidth);
	split = strtok (NULL, " \n");
	*imageHeight = atoi(split);
	printf("HEIGHT: %d\n", *imageHeight);
      }
      // read maximum gray value
      if (scannedLines == 3) {
	split = strtok (lineBuffer, " \n");
	*grayMax = atoi(split);
	printf("GRAYMAX: %d\n", *grayMax);
      }
    } else{
      strcpy(comments,lineBuffer);
      printf("comments: %s", comments);
    }
  }

  //////////////////////////////////////////////////////////////////
  //
  // memory allocation for iamge array
  //
  //////////////////////////////////////////////////////////////////

  *image = (unsigned char*) malloc				\
    ( (*imageWidth) * (*imageHeight) * sizeof(unsigned char) );
  counter = fread(*image, sizeof(unsigned char),		\
		  (*imageWidth) * (*imageHeight), input);
  printf("pixels read: %d\n", counter);
  printf("\n");
  
  //////////////////////////////////////////////////////////////////
  //
  // Close the input PGM file and free line buffer
  //
  //////////////////////////////////////////////////////////////////
  
  fclose(input);
  free(lineBuffer);
  free(format);

  return 0;

}


int writePGM(unsigned char *image, char *filename,		\
	     int imageWidth, int imageHeight, int grayMax){
  
  FILE *output;
  int counter = 0;

  //////////////////////////////////////////////////////////////////
  //
  // Open output PGM file
  //
  //////////////////////////////////////////////////////////////////

  output=fopen(filename,"w");

  //////////////////////////////////////////////////////////////////
  //
  // Write output PGM file header
  //
  //////////////////////////////////////////////////////////////////
  
  fprintf(output, "%s\n", "P5");
  fprintf(output, "%s\n", "# Created by 5kk73.");
  fprintf(output, "%d %d\n", imageWidth, imageHeight);
  fprintf(output, "%d\n", grayMax);

  //////////////////////////////////////////////////////////////////
  //
  // Write output PGM gray value
  //
  //////////////////////////////////////////////////////////////////
  
  counter = fwrite(image, sizeof(unsigned char),	\
		   imageWidth * imageHeight, output);
  
  //////////////////////////////////////////////////////////////////
  //
  // Close the input PGM file
  //
  //////////////////////////////////////////////////////////////////
  
  fclose(output);

  return 0;

}
