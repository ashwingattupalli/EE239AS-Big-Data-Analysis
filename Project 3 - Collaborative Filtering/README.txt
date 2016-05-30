### EE239AS - Special Topics in Signals & Systems - Project 3 ###

### Team Members ###
Mansee Jadhav - 204567818
Mauli Shah - 004567942
Ronak Sumbaly - 604591897

### COLLABORATIVE FILTERING ###
The following README contains the requirements and steps that are required to execute Project 3. 
The structure of the folders have also be explained. 

### IMPLEMENTATION ###
## Dependencies ##
a. numpy v1.10.4
b. scipy v0.16.1
c. matplotlib v1.5.1
d. sklearn v0.17

## Usage ##
       
Usage: python project3.py -q <question number>

<question number> : 1, 2, 3, 4, 5
                     
## Examples ##
If you want to run Question 1 (Total Least Squared Error),
$ cd Scripts 
$ python project3.py -q 1

If you want to run Question 5 (Recommender System),
$ cd Scripts
$ python project3.py -q 5

## Description ##
- Import the entire zip file containing all the folders (Dataset, Graphs, Scripts) into an IDE (e.g PyCharm).
- Run project3.py with configurations as specified above.
- Graphs are displayed as a separate window.
- Output of each question is displayed in the console output.

## Folder Structure ##
- Dataset : Initially empty. Will contain pickle files of Pre-processed data & LSI matrices after first execution.
- Graphs : Comprises of all graphs that are plotted for all questions.
- Scripts : Contains all the python scripts required to execute the project.
            - project3.py : Main file that is executed to get results
            - execute.py : Comprises for code for each question in the project.
            - utility.py : Imported by execute.py comprises of generic functions to execute
                           * Load Dataset          		
                           * Perform Classification   	* Calculate error
                           * Non-matrix factorization     	* Plot ROC curve