from Insurance.logger import logging
from Insurance.exception import InsuranceException
import os
import sys

def test_logger_exception():
    try:
        logging.info("Starting the test logger and exception!")
        result = 3 / 0
        print(result)
        logging.info("Ending point of the test logger and exception!")
    except Exception as e:
        logging.debug(str(e))
        raise InsuranceException(e, sys)
    

if __name__ == "__main__":
    try:
        test_logger_exception()
    except Exception as e:
        print(e)