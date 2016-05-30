### EE239AS - Special Topics in Signals & Systems - Project 2 ###

### Team Members ###
Mansee Jadhav - 204567818
Mauli Shah - 004567942
Ronak Sumbaly - 604591897

### CLASSIFICATION ANALYSIS ###
The following README contains the requirements and steps that are required to execute Project 2. 
The structure of the folders have also be explained. 

### IMPLEMENTATION ###
## Dependencies ##
a. numpy v1.10.4
b. scipy v0.16.1
c. matplotlib v1.5.1
d. sklearn v0.17
e. nlkt v3.0

## Usage ##
       
Usage: python project2.py -q <question number>

<question number> : a, b, c, d, e, h, i, j
                     
## Examples ##
If you want to run Question a (Plot Histograms and Count of Documents),
$ cd Scripts 
$ python project2.py -q a

If you want to run Question h (Perform Logistic Regression),
$ cd Scripts
$ python project2.py -q h  

## Description ##
- Import the entire zip file containing all the folders (Dataset, Graphs, Scripts) into an IDE (e.g PyCharm).
- Run project2.py with configurations as specified above.
- Graphs are displayed as a separate window.
- Output of each question is displayed in the console output.

## Folder Structure ##
- Dataset : Initially empty. Will contain pickle files of Pre-processed data & LSI matrices after first execution.
- Graphs : Comprises of all graphs that are plotted for all questions.
- Scripts : Contains all the python scripts required to execute the project.
            - project2.py : Main file that is executed to get results
            - execute.py : Comprises for code for each question in the project.
            - utility.py : Imported by execute.py comprises of generic functions to execute
                           * Load Dataset          		      * Model dataset
                           * Pre-process Dataset   	          * Perform LSI decomposition
                           * Calculate Classifier Statistics  * Plot ROC curve

## NOTE ##
- Execution time for Question (c) is around 10 mins as the TFxICF stats are being calculated for each term in the document