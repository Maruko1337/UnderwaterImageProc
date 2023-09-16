#include <stdio.h>
#include <string.h> 
int mai()
{
   char Name[100];   
   char Address[100];
   int salary
   FILE *fpw;
   fpw=fopen("e:\\newfile.txt","w);
   if(fpw == NULL)
   {
      printf("Error");   
   return            
   }
   printf("Enter any Name: ");
   scanf("%s",Name);
    printf("Enter any Address: ");
   scanf("%s",&Address); 
   printf("Enter any Salary: ");
   scanf("%s",&salary); 
   fprintf(fpw,"%s\n",Name);
   fprintf(fpw,"%s\n",Address);
   fprintf(fpw,"%d\n",salary);
   close(*fpw);
   eturn 0;
}
