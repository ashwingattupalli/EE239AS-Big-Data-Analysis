import copy
import execute
import argparse
import warnings
import logging as logger

warnings.filterwarnings("ignore")

logger.basicConfig(level=logger.INFO,format='%(message)s')

if __name__ == "__main__":
    # add parser arguments to enable question wise execution
    parser = argparse.ArgumentParser()
    parser.add_argument("--ques", "-q", help="execute question number (a,b,c,d,e)")
    args = parser.parse_args()
    args_dict = vars(copy.deepcopy(args))

    ques = args_dict['ques']

    # execute the corresponding question as per command line input

    if ques == '1':     execute.question_1()
    elif ques == '2':   execute.question_2()
    elif ques == '3':   execute.question_3()
    elif ques == '4':   execute.question_4()
    elif ques == '5':   execute.question_5()
    else:               logger.info("Input correct question number as command-line argument")
