### EE239AS - Special Topics in Signals & Systems - Project 1 ###

### Team Members ###
Mansee Jadhav - 204567818
Mauli Shah - 004567942
Ronak Sumbaly - 604591897

### REGRESSION ANALYSIS ###
The following README contains the requirements and steps that are required to execute Project 1. 
The structure of the folders have also be explained. 

### IMPLEMENTATION ###
## Dependencies ##
a. pandas v0.17.0
b. numpy v1.10.4
c. scipy v0.16.1
d. matplotlib v1.5.1
e. sklearn v0.17
f. statsmodel v0.6.1
g. pybrain v0.3

## Usage ##
       
Usage: python project1.py -q <question number>

<question number> : Network - 1 / 2a / 2b / 2c / 3
                    Housing - 4a / 4b / 5
                     
## Examples ##
If you want to run Question 1 (Plot Relationships Network Dataset),
$ cd Scripts 
$ python project1.py -q 1

If you want to run Question 4a (Linear Regression Housing Dataset),
$ cd Scripts
$ python project1.py -q 4a  

## Description ##
- Import the entire zip file containing all the folders (Dataset, Graphs, Scripts) into an IDE (e.g PyCharm).
- Run project1.py with configurations as specified above.
- Graphs are displayed and saved in the Graphs folders automatically.
  * Note. It is important to maintain the structure of the folders for the purposes of proper execution of the project 
          and in-order for graphs to be saved in the right location.  
- Output of each question is displayed in the console output.

## Folder Structure ##
- Dataset : Contains both the network-backup and boston-housing csv data files required for data analysis. 
- Graphs : Sub-divided into Network (Question 1, 2a, 2b, 2c, 3) and Housing (Question 4a, 4b, 5) 
           with each folder containing respective folders for their question number.
- Scripts : Contains all the python scripts required to execute the project.
            - project1.py : Main file that is executed to get results
            - network.py : Executes questions relating to the network-backup dataset. 
            - housing.py : Executes questions relating to the boston-housing dataset.
            - utility.py : Imported by both network.py/housing.py comprises of generic functions to execute
                           * Linear Regression          * Polynomial Regression
                           * Random Forest Regression   * Ridge Regression
                           * Neural Network Regression  * Lasso Regularization

## NOTE ##
- Execution time for Neural Networks can be upto 5 mins due to training of model for 100 epochs. 