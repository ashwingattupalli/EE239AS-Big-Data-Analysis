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
    parser.add_argument("--ques", "-q", help="execute question number (a,b,c,d,e,f,g,h,i)")
    args = parser.parse_args()
    args_dict = vars(copy.deepcopy(args))

    ques = args_dict['ques']

    # execute the corresponding question as per command line input

    if ques == 'a':     execute.question_a()
    elif ques == 'b':   execute.question_b()
    elif ques == 'c':   execute.question_c()
    elif ques == 'd':   execute.question_d()
    elif ques == 'e':   execute.question_e()
    elif ques == 'f':   execute.question_f()
    elif ques == 'g':   execute.question_g()
    elif ques == 'h':   execute.question_h()
    elif ques == 'i':   execute.question_i()
    else:               logger.info("Input correct question number as command-line argument")



